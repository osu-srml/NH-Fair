"""LLaVA-NeXT: local Hugging Face or OpenAI-compatible gateway."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from release_benchmark.methods.lvlm.backend_config import (
    llm_gateway_base_url,
    resolve_vlm_backend,
)
from release_benchmark.methods.lvlm.gateway import OpenAIGatewayClient
from release_benchmark.methods.lvlm.llm_utils import (
    build_conversation,
    build_conversation_open,
)
from release_benchmark.methods.lvlm.zeroshot_common import (
    append_outputs_from_generated,
    build_prompt_text,
    finalize_zeroshot_metrics,
    max_output_tokens,
    print_gateway_banner,
    save_generation_artifacts,
)
from release_benchmark.utils.common import AverageMeter


def _parse_llava_answers(generated_texts: list[str]) -> list[str]:
    final_answers = []
    for text in generated_texts:
        t = (text or "").lower()
        if "[/inst]" in t:
            final_answers.append(text.split("[/inst]")[-1].strip())
        elif "assistant" in t:
            ans = text.split("assistant")[-1].strip().replace(":", "").strip()
            final_answers.append(ans)
        else:
            final_answers.append(text.strip())
    return final_answers


class llava_next(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.task_mode = getattr(args, "task_mode", "classification")
        self.backend = resolve_vlm_backend(args)

        if self.backend == "gateway":
            self.processor = None
            self.model = None
            print_gateway_banner(llm_gateway_base_url(args))
            self.gateway = OpenAIGatewayClient(args)
            self._text = build_prompt_text(args)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(
                args.model, padding_side="left"
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self.gateway = None
            if self.task_mode == "open_generation":
                conv = build_conversation_open(
                    args.dataset,
                    prompt_style=args.prompt_style,
                    task_attr=args.ta,
                    sensitive_attr=args.sa,
                )
            else:
                conv = build_conversation(
                    args.dataset,
                    prompt_style=args.prompt_style,
                    task_attr=args.ta,
                    sensitive_attr=args.sa,
                )
            self.prompt = self.processor.apply_chat_template(
                conv, add_generation_prompt=True
            )
        print(f"LLaVA-NeXT backend={self.backend} task_mode={self.task_mode}")

    def validate(self, val_set, epoch, args):
        return self._evaluate(val_set, epoch, args, mode="validation")

    def test(self, test_set, epoch, args):
        return self._evaluate(test_set, epoch, args, mode="test")

    def _evaluate(self, dataset, epoch, args, mode="test"):
        dataset.transform = None
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        tol_generated_texts = []
        loss_meter = AverageMeter()

        if self.backend == "transformers":
            self.model.eval()

        with torch.inference_mode():
            for data, target, sensitive_attr in tqdm(dataset):
                if self.backend == "transformers":
                    inputs = self.processor(
                        text=[self.prompt for _ in range(len(data))],
                        images=data,
                        return_tensors="pt",
                        padding=True,
                    ).to(args.device)
                    mnt = 64 if self.task_mode == "classification" else 1280
                    generated_ids = self.model.generate(**inputs, max_new_tokens=mnt)
                    generated_texts = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    output_text = _parse_llava_answers(generated_texts)
                else:
                    txt = self.gateway.generate_from_pil(
                        data[0],
                        self._text,
                        max_tokens=max_output_tokens(self.task_mode, args),
                        model=getattr(args, "model", None),
                    )
                    output_text = _parse_llava_answers([txt]) if txt else []

                if not output_text or output_text[0] in (None, "None", "Unknown", ""):
                    continue
                ok = append_outputs_from_generated(
                    output_text, self.task_mode, args, tol_output, tol_generated_texts
                )
                if not ok:
                    continue
                tol_target += target.cpu().numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().numpy().tolist()

        if not tol_output:
            return loss_meter.avg, {}

        log_dict = finalize_zeroshot_metrics(
            tol_output,
            tol_target,
            tol_sensitive,
            tol_index,
            tol_generated_texts,
            args,
            self.task_mode,
            self.num_classes,
            args.sensitive_attributes,
            mode,
            epoch,
            loss_meter,
        )

        if tol_generated_texts and self.task_mode != "classification":
            save_generation_artifacts(
                args,
                self.task_mode,
                mode,
                epoch,
                tol_generated_texts,
                tol_target,
                tol_sensitive,
                np.asarray(tol_output),
                self.num_classes,
                gateway_extra={"llm_gateway_url": llm_gateway_base_url(args)}
                if self.backend == "gateway"
                else None,
            )
        return loss_meter.avg, log_dict
