import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.utils.common import AverageMeter

from .erm import erm


class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self, features, labels, sensitive_labels, group_norm, method, epoch, mask=None
    ):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: target classes of shape [bsz].
            sensitive_labels: sensitive attributes of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            sensitive_mask = mask.clone()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
            sensitive_mask = (
                torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)
            )
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        # compute log_prob
        if method == "FSCL":
            mask = mask * logits_mask
            logits_mask_fair = logits_mask * (~mask.bool()).float() * sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum = exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum + ((exp_logits_sum == 0) * 1))

        elif method == "SupCon":
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        elif method == "FSCL*":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask = mask * logits_mask

            logits_mask_fair = logits_mask * sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum = exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum + ((exp_logits_sum == 0) * 1))

        elif method == "SimCLR":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # apply group normalization
        if group_norm == 1:
            mean_log_prob_pos = (
                (mask * log_prob) / ((mask * sensitive_mask).sum(1))
            ).sum(1)

        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # apply group normalization
        if group_norm == 1:
            C = loss.size(0) / 8
            norm = 1 / (((mask * sensitive_mask).sum(1) + 1).float())
            loss = (loss * norm) * C

        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ProjectionHead(nn.Module):
    """MLP projection head matching the original FSCL architecture."""

    def __init__(self, dim_in, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)


class fscl(erm):
    def __init__(self, args):
        super().__init__(args)
        self.cl_loss = FairSupConLoss(temperature=0.1)
        self.mode = "cl_training"
        self.local_max_patience = args.max_patience
        args.max_patience = 100
        self.local_patience = 0

        dim_in = self.model.fc.in_features
        self.projection_head = ProjectionHead(dim_in).to(args.device)

        cl_params = list(self.model.backbone.parameters()) + list(
            self.projection_head.parameters()
        )
        self.optimizer, self.fe_scheduler = self._make_optimizer(cl_params, args)
        self.optimizer_classifier = self._make_optimizer(
            self.model.fc.parameters(), args
        )[0]
        self.last_loss = 1e10

    def _make_optimizer(self, params, args):
        fe_scheduler = None
        if args.optim == "adam":
            optimizer = torch.optim.Adam(
                params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
            )
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
            )

        if args.optim == "sgd":
            fe_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.StepLR_size, gamma=args.gamma
            )
        return optimizer, fe_scheduler

    def train(self, train_loader, epoch, args):
        if epoch >= args.epochs - 60:
            self.mode = "classification"
        model = self.model
        device = args.device
        model.train()
        if not args.no_progress:
            p_bar = tqdm(range(len(train_loader)))
        total_loss = AverageMeter()
        for batch_idx, (data, target, sensitive_attr) in enumerate(train_loader):
            data_view1, data_view2 = data
            data_view1, data_view2, target, sensitive_attr = (
                data_view1.to(device),
                data_view2.to(device),
                target.to(device),
                sensitive_attr.to(device),
            )

            if self.mode == "cl_training":
                self.optimizer.zero_grad()
                inputs = torch.cat([data_view1, data_view2], dim=0)
                _, backbone_features = model.forward_return_feature(inputs)
                features = self.projection_head(backbone_features)
                bsz = target.shape[0]
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = self.cl_loss(
                    features, target, sensitive_attr, args.group_norm, "FSCL", epoch
                )
                loss.backward()
                self.optimizer.step()

            else:
                self.optimizer_classifier.zero_grad()
                with torch.no_grad():
                    _, features = model.forward_return_feature(data_view1)
                output = model.fc(features)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer_classifier.step()

            total_loss.update(loss.item())
            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch + 1}/{args.epochs:4}. Iter: {batch_idx + 1:4}/{len(train_loader):4}. Loss: {total_loss.avg:.4f}.  "
                )
                p_bar.update()
        if not args.no_progress:
            p_bar.close()
        print(f"Average loss for epoch {epoch}: {total_loss.avg}")
        if self.mode == "cl_training":
            if total_loss.avg > self.last_loss:
                self.local_patience += 1
            else:
                self.local_patience = 0
            self.last_loss = total_loss.avg
            if self.local_patience > self.local_max_patience:
                self.mode = "classification"
                args.max_patience = self.local_max_patience

        return total_loss.avg
