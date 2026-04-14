"""Llama Vision (3.2 / 4 Scout): local Hugging Face or OpenAI-compatible gateway."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor

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


def _is_llama4(model_name: str) -> bool:
    lowered = model_name.lower().replace("-", "").replace("_", "")
    return "llama4" in lowered


def _load_model(model_name: str, **kwargs):
    if _is_llama4(model_name):
        from transformers import Llama4ForConditionalGeneration

        return Llama4ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    from transformers import MllamaForConditionalGeneration

    return MllamaForConditionalGeneration.from_pretrained(model_name, **kwargs)


class llama(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.task_mode = getattr(args, "task_mode", "classification")
        self.backend = resolve_vlm_backend(args)
        self._is_llama4 = _is_llama4(getattr(args, "model", ""))

        if self.backend == "gateway":
            self.processor = None
            self.model = None
            print_gateway_banner(llm_gateway_base_url(args))
            self.gateway = OpenAIGatewayClient(args)
            self._text = build_prompt_text(args)
        else:
            self.processor = AutoProcessor.from_pretrained(
                args.model, padding_side="left"
            )
            self.model = _load_model(
                args.model, device_map="auto", torch_dtype=torch.bfloat16
            )
            self.gateway = None
            if self.task_mode == "open_generation":
                self.text = build_conversation_open(
                    args.dataset,
                    prompt_style=args.prompt_style,
                    task_attr=args.ta,
                    sensitive_attr=args.sa,
                    textonly=True,
                )
            else:
                self.text = build_conversation(
                    args.dataset,
                    prompt_style=args.prompt_style,
                    task_attr=args.ta,
                    sensitive_attr=args.sa,
                    textonly=True,
                )
        family = "Llama4" if self._is_llama4 else "Llama"
        print(f"{family} backend={self.backend} task_mode={self.task_mode}")

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
                    batch_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img1},
                                {"type": "text", "text": self.text},
                            ],
                        }
                        for img1 in data
                    ]
                    chat_kwargs = dict(
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        processor_kwargs={"padding": True},
                    )
                    inputs = self.processor.apply_chat_template(
                        batch_messages, **chat_kwargs
                    ).to(args.device)
                    mnt = 64 if self.task_mode == "classification" else 1280
                    outputs = self.model.generate(**inputs, max_new_tokens=mnt)
                    output_text = self.processor.batch_decode(
                        outputs[:, inputs["input_ids"].shape[-1] :],
                        skip_special_tokens=True,
                    )
                else:
                    txt = self.gateway.generate_from_pil(
                        data[0],
                        self._text,
                        max_tokens=max_output_tokens(self.task_mode, args),
                        model=getattr(args, "model", None),
                    )
                    output_text = [txt] if txt else []

                if not output_text:
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
