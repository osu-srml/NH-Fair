"""BLIP-2 feature extractor via Salesforce LAVIS.

Install `salesforce-lavis` with the `lavis` optional extra (see `pyproject.toml`); that
pins `transformers` for LAVIS and conflicts with the `llm` extra. Use CUDA PyTorch for GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

from release_benchmark.methods.vlm.vlm_utils import set_matching_prompt
from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter


class blip2(nn.Module):
    def __init__(self, args):

        super().__init__()
        # LAVIS moves weights to ``args.device`` and returns matching preprocessors.
        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name="blip2_feature_extractor",
                model_type="pretrain",
                is_eval=True,
                device=args.device,
            )
        )
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.text = set_matching_prompt(args)
        self.optimizer = None
        self.fe_scheduler = None
        self.text_inputs = [self.text_processors["eval"](txt) for txt in self.text]

    def compute_similarity(self, data):
        sample = {"image": data, "text_input": self.text_inputs}

        features_image = self.model.extract_features(
            sample, mode="image"
        ).image_embeds_proj
        features_text = self.model.extract_features(
            sample, mode="text"
        ).text_embeds_proj

        similarity, _ = torch.max((features_image @ features_text[:, 0, :].t()), dim=1)

        return similarity

    def validate(self, val_set, epoch, args):
        return self._evaluate(val_set, epoch, args, mode="validation")

    def test(self, test_set, epoch, args):
        return self._evaluate(test_set, epoch, args, mode="test")

    def _evaluate(self, dataset, epoch, args, mode="test"):
        # Clear transforms from the shared benchmark dataset so inputs follow this method's path.
        dataset.transform = None
        self.model.eval()

        eval_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in tqdm(dataset):
                data, target = data.to(args.device), target.to(args.device)
                similarity = self.compute_similarity(data)
                torch.argmax(similarity, dim=1)
                output = similarity.float()

                prob = F.softmax(output, dim=-1)

                loss = self.criterion(output, target)
                eval_loss.update(loss.item())

                tol_output += prob.cpu().numpy().tolist()
                tol_target += target.cpu().numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().numpy().tolist()

        log_dict, _t_predictions, _aucs_subgroup = calculate_metrics(
            tol_output,
            tol_target,
            tol_sensitive,
            tol_index,
            args.sensitive_attributes,
            num_class=args.num_classes,
        )

        print(
            f"\n##################################### {mode.capitalize()} {epoch} #######################################"
        )
        print(log_dict, "\n")
        return eval_loss.avg, log_dict
