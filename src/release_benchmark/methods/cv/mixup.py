import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from release_benchmark.utils.common import AverageMeter

from .erm import erm


class mixup(erm):
    def __init__(self, args):

        super().__init__(args)
        self.model = self.setmodel(args)
        self.optimizer, self.fe_scheduler = self.setoptimizer(self.model, args)
        if args.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.num_classes >= 2:
            self.criterion = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes
        self.args = args

    def mixup_data(self, x, y, alpha=1.0, device="cuda:0"):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]

        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        if self.args.num_classes == 1:
            return lam * criterion(pred, y_a.float()) + (1 - lam) * criterion(
                pred, y_b.float()
            )
        elif self.args.num_classes >= 2:
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def mixup_data_by_sensitive(self, x, y, sensitive_attr, alpha=1.0, device="cuda:0"):
        """
        Efficient Mixup for inputs with the same target but different sensitive attributes using torch operations.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        x.size(0)

        # Broadcast y and sensitive_attr for pairwise comparison
        y_expand = y.unsqueeze(0)  # Shape: [1, batch_size]
        sensitive_expand = sensitive_attr.unsqueeze(0)  # Shape: [1, batch_size]

        # Create a mask for valid pairs: same target, different sensitive attribute
        pair_mask = (y_expand == y_expand.t()) & (
            sensitive_expand != sensitive_expand.t()
        )  # Shape: [batch_size, batch_size]

        # Remove diagonal to exclude self-pairs
        pair_mask.fill_diagonal_(True)

        # Find at least one valid pair for each sample
        indices = torch.multinomial(
            pair_mask.float(), num_samples=1, replacement=True
        ).squeeze(1)  # Shape: [batch_size]

        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b, lam

    def train(self, train_loader, epoch, args):
        model = self.model
        device = args.device
        model.train()
        if not args.no_progress:
            p_bar = tqdm(range(len(train_loader)))
        total_loss = AverageMeter()
        for batch_idx, (data, target, sensitive_attr) in enumerate(train_loader):
            data, target, sensitive_attr = (
                data.to(device),
                target.to(device),
                sensitive_attr.to(device),
            )
            if args.mixup_mode == "random":
                inputs, targets_a, targets_b, lam = self.mixup_data(
                    data, target, args.mixup_alpha, device
                )
            elif args.mixup_mode == "group":
                inputs, targets_a, targets_b, lam = self.mixup_data_by_sensitive(
                    data, target, sensitive_attr, args.mixup_alpha, device
                )
            else:
                raise NotImplementedError

            output = model(inputs)
            self.optimizer.zero_grad()
            if args.num_classes == 1:
                target = target.float()
            output = output.squeeze()

            loss = self.criterion(
                output, target
            ) + args.mixup_lam * self.mixup_criterion(
                self.criterion, output, targets_a, targets_b, lam
            )

            loss.backward()
            self.optimizer.step()
            total_loss.update(loss.item())
            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch + 1}/{args.epochs:4}. Iter: {batch_idx + 1:4}/{len(train_loader):4}. Loss: {total_loss.avg:.4f}.  "
                )
                p_bar.update()
        if not args.no_progress:
            p_bar.close()
        print(f"Average loss for epoch {epoch}: {total_loss.avg}")
        return total_loss.avg
