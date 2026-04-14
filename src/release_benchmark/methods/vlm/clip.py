"""OpenAI CLIP zero-shot classifier (``zeroshot`` CLI).

``train`` / ``fine_tune_step`` raise; former finetuning logic lives in ``_legacy_*`` helpers
and ``setoptimizer`` / ``configure_finetuning`` for reference only.
"""

import clip as openclip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.methods.vlm.vlm_utils import set_matching_prompt
from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.utils.common import AverageMeter


class clip(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.model = self.setmodel(args)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.text_tokens = self.set_prompts(args)
        self.optimizer = None
        self.fe_scheduler = None

    def setoptimizer(self, model, args):
        fe_scheduler = None
        if args.optim == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
            )
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
            )

        if args.optim == "sgd":
            fe_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.StepLR_size, gamma=args.gamma
            )
        return optimizer, fe_scheduler

    def set_prompts(self, args):
        texts = set_matching_prompt(args)
        return openclip.tokenize(texts).to(args.device)

    def setmodel(self, args):

        if "resnet" in args.model:
            model, _preprocess = openclip.load("RN50", device=args.device)
        elif "vitb16" in args.model:
            model, _preprocess = openclip.load("ViT-B/16", device=args.device)
        else:
            raise NotImplementedError
        return model

    def configure_finetuning(self):
        # Legacy finetuning recipe (not reachable via train CLI).
        for name, param in self.model.named_parameters():
            if "text_projection" in name or "logit_scale" in name or "text" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train(self, train_loader, epoch, args):
        raise RuntimeError(
            "CLIP finetuning is disabled; use zero-shot: "
            "`python -m release_benchmark.cli.zeroshot --method clip ...`."
        )

    def fine_tune_step(self, data, target):
        raise RuntimeError(
            "CLIP finetuning is disabled; use zero-shot: "
            "`python -m release_benchmark.cli.zeroshot --method clip ...`."
        )

    def _legacy_train_one_epoch(self, train_loader, epoch, args):
        """Former supervised CLIP epoch (not called). Needs ``self.optimizer`` from ``setoptimizer``."""
        self.model.train()
        device = args.device

        if not args.no_progress:
            p_bar = tqdm(range(len(train_loader)))
        total_loss = AverageMeter()

        for batch_idx, (data, target, _sensitive_attr) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loss = self._legacy_finetune_step(data, target)
            total_loss.update(loss)
            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch + 1}/{args.epochs:4}. Iter: {batch_idx + 1:4}/{len(train_loader):4}. Loss: {total_loss.avg:.4f}.  "
                )
                p_bar.update()
        if not args.no_progress:
            p_bar.close()
        print(f"Average loss for epoch {epoch}: {total_loss.avg}")
        return total_loss.avg

    def _legacy_finetune_step(self, data, target):
        """Former CLIP finetune step (not called). Requires ``self.optimizer``."""
        image_features = self.model.encode_image(data)
        text_features = self.model.encode_text(self.text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T

        if self.num_classes == 1:
            target = target.float()
            similarity = similarity.squeeze()
            loss = self.criterion(similarity, target)
        else:
            loss = self.criterion(similarity, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def validate(self, val_loader, epoch, args):
        model = self.model
        model.eval()
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        corrent_torch = 0
        with torch.no_grad():
            for data, target, sensitive_attr in tqdm(val_loader):
                data, target = data.to(args.device), target.to(args.device)
                image_features = model.encode_image(data)
                text_features = model.encode_text(self.text_tokens)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # print(text_features)
                similarity = image_features @ text_features.T  # [batch_size, 2]
                predictions = torch.argmax(
                    similarity, dim=1
                )  # Return the predicted class for each image (0 or 1).
                output = similarity.float()
                corrent_torch += (predictions == target).sum()
                if self.num_classes == 1:  # BCE Loss
                    target = target.float()
                    output = output.squeeze()
                    prob = F.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                loss = self.criterion(output, target)
                try:
                    val_loss.update(loss.item())
                except Exception:
                    val_loss.update(loss.mean().item())

                tol_output += prob.cpu().data.numpy().tolist()
                tol_target += target.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()
        log_dict, _t_predictions, _aucs_subgroup = calculate_metrics(
            tol_output,
            tol_target,
            tol_sensitive,
            tol_index,
            args.sensitive_attributes,
            num_class=args.num_classes,
        )

        print(
            f"#####################################validation {epoch}#######################################"
        )
        print(log_dict, "\n")
        return val_loss.avg, log_dict

    def test(self, test_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()
        test_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in test_loader:
                data = data.to(device)
                # print(data.shape)
                image_features = model.encode_image(data)
                text_features = model.encode_text(self.text_tokens)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = image_features @ text_features.T  # [batch_size, 2]
                torch.argmax(
                    similarity, dim=1
                )  # Return the predicted class for each image (0 or 1).
                output = similarity.to(torch.float32)

                # print(self.num_classes)
                if self.num_classes == 1:  # BCE Loss
                    target = target.float()
                    output = output.squeeze()
                    prob = F.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                tol_output += prob.cpu().data.numpy().tolist()
                tol_target += target.cpu().data.numpy().tolist()
                tol_sensitive += sensitive_attr.cpu().data.numpy().tolist()

        log_dict, _t_predictions, _aucs_subgroup = calculate_metrics(
            tol_output,
            tol_target,
            tol_sensitive,
            tol_index,
            args.sensitive_attributes,
            num_class=args.num_classes,
        )
        print(
            "\n#####################################Test#######################################"
        )
        print(log_dict, "\n")
        return test_loss.avg, log_dict
