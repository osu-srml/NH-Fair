import torch
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.utils.common import AverageMeter

from .erm import erm


class DiffEOdd(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, s, y_gt):
        if self.num_classes == 2:
            y_pred = y_pred[:, 1]
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        y_pred_y1 = y_pred[y_gt == 1]
        s_y1 = s[y_gt == 1]

        y0 = y_pred_y1[s_y1 == 0]
        y1 = y_pred_y1[s_y1 == 1]
        reg_loss_y1 = torch.abs(torch.mean(y0) - torch.mean(y1))

        y_pred_y0 = y_pred[y_gt == 0]
        s_y0 = s[y_gt == 0]

        y0 = y_pred_y0[s_y0 == 0]
        y1 = y_pred_y0[s_y0 == 1]
        reg_loss_y0 = torch.abs(torch.mean(y0) - torch.mean(y1))

        reg_loss = reg_loss_y1 + reg_loss_y0
        return reg_loss


class DiffEOdd_multiclass(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, s, y_gt):
        reg_loss = 0.0

        for c in range(self.num_classes):
            preds_c = y_pred[:, c]
            mask = y_gt == c
            if mask.sum() == 0:
                continue
            preds_c = preds_c[mask]
            s_c = s[mask]

            group0 = preds_c[s_c == 0]
            group1 = preds_c[s_c == 1]

            if group0.numel() == 0 or group1.numel() == 0:
                continue

            diff = torch.abs(torch.mean(group0) - torch.mean(group1))
            reg_loss += diff

        return reg_loss


class DiffEOpp(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, s, y_gt):
        if self.num_classes == 2:
            y_pred = y_pred[:, 1]
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        y_pred = y_pred[y_gt == 1]
        s = s[y_gt == 1]

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        reg_loss = torch.abs(torch.mean(y0) - torch.mean(y1))
        return reg_loss


class DiffDP(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, s):
        if self.num_classes == 2:
            y_pred = y_pred[:, 1]
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        reg_loss = torch.abs(torch.mean(y0) - torch.mean(y1))
        return reg_loss


class DiffDP_multiclass(
    torch.nn.Module
):  # https://www.jmlr.org/papers/volume25/23-0322/23-0322.pdf
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, s):
        # y_pred: [bs, num_class]

        mask0 = (s == 0).unsqueeze(1)  # shape: [bs, 1]
        mask1 = (s == 1).unsqueeze(1)

        epsilon = 1e-8

        mean0 = (y_pred * mask0).sum(dim=0) / (mask0.sum(dim=0).float() + epsilon)
        mean1 = (y_pred * mask1).sum(dim=0) / (mask1.sum(dim=0).float() + epsilon)

        reg_loss_per_class = torch.abs(mean0 - mean1)

        final_loss = torch.max(reg_loss_per_class)
        return final_loss


class gapreg(erm):
    def __init__(self, args):
        super().__init__(args)
        metric = args.diff_metric
        if metric == "eop":
            metric = "opp"
        elif metric == "eod":
            metric = "odd"
        self.diff_metric = metric

        if self.diff_metric == "dp":
            self.fair_loss = DiffDP(num_classes=args.num_classes)
            if args.num_classes > 2:
                self.fair_loss = DiffDP_multiclass()
        if self.diff_metric == "opp":
            self.fair_loss = DiffEOpp(num_classes=args.num_classes)
            if args.num_classes > 2:
                raise NotImplementedError
        if self.diff_metric == "odd":
            self.fair_loss = DiffEOdd(num_classes=args.num_classes)
            if args.num_classes > 2:
                self.fair_loss = DiffEOdd_multiclass(num_classes=args.num_classes)

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
            output = model(data)
            self.optimizer.zero_grad()

            if args.num_classes == 1:
                output = output.squeeze()
                target = target.float()
                prob = torch.sigmoid(output).flatten()
            else:
                prob = F.softmax(output, dim=-1)

            if self.diff_metric == "dp":
                fairloss = args.diff_lambda * self.fair_loss(prob, sensitive_attr)
            else:
                fairloss = args.diff_lambda * self.fair_loss(
                    prob, sensitive_attr, target
                )

            loss = self.criterion(output, target) + fairloss
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
