import torch
import torch.nn.functional as F
from torch.func import vmap
from tqdm import tqdm

from release_benchmark.utils.common import AverageMeter

from .erm import erm


class MaxCDFdp(torch.nn.Module):
    def __init__(self, temperature, num_classes=2):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, y_pred, s):
        if self.num_classes == 2:
            y_pred = y_pred[:, 1]
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        probing_points = (
            torch.linspace(
                torch.min(y_pred).item(), torch.max(y_pred).item(), steps=100
            )
            .reshape(-1, 1)
            .to(y_pred.device)
        )
        diff0 = probing_points - y_pred[s == 0]
        diff1 = probing_points - y_pred[s == 1]
        diff0 = torch.sigmoid(self.temperature * diff0)
        diff1 = torch.sigmoid(self.temperature * diff1)
        cdf0 = torch.mean(diff0, axis=1)
        cdf1 = torch.mean(diff1, axis=1)
        delta_ecdf = torch.abs(cdf0 - cdf1)

        return torch.max(delta_ecdf)


class MaxCDFdp_multiclass(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_pred, s):
        num_class = y_pred.shape[1]
        losses = []
        for i in range(num_class):
            preds = y_pred[:, i]
            min_val = torch.min(preds).item()
            max_val = torch.max(preds).item()
            probing_points = (
                torch.linspace(min_val, max_val, steps=100)
                .reshape(-1, 1)
                .to(y_pred.device)
            )
            preds0 = preds[s == 0]
            preds1 = preds[s == 1]
            if preds0.numel() == 0 or preds1.numel() == 0:
                continue
            diff0 = probing_points - preds0
            diff1 = probing_points - preds1
            diff0 = torch.sigmoid(self.temperature * diff0)
            diff1 = torch.sigmoid(self.temperature * diff1)

            cdf0 = torch.mean(diff0, dim=1)
            cdf1 = torch.mean(diff1, dim=1)
            delta_ecdf = torch.abs(cdf0 - cdf1)
            loss_i = torch.max(delta_ecdf)
            losses.append(loss_i)
        if len(losses) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        final_loss = torch.stack(losses).max()
        return final_loss


class MaxCDFdp_multiclass_vmap(
    torch.nn.Module
):  # if pytorch version doesn't support vmap, use MaxCDFdp_multiclass instead
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def compute_loss_for_class(self, preds, s, temperature):
        min_val = preds.min()
        max_val = preds.max()
        linspace = torch.linspace(0.0, 1.0, steps=100, device=preds.device).reshape(
            -1, 1
        )
        probing_points = min_val + linspace * (max_val - min_val)

        preds0 = preds[s == 0]
        preds1 = preds[s == 1]
        if preds0.numel() == 0 or preds1.numel() == 0:
            return torch.tensor(0.0, device=preds.device)
        diff0 = probing_points - preds0
        diff1 = probing_points - preds1
        diff0 = torch.sigmoid(temperature * diff0)
        diff1 = torch.sigmoid(temperature * diff1)
        cdf0 = diff0.mean(dim=1)
        cdf1 = diff1.mean(dim=1)
        delta_ecdf = torch.abs(cdf0 - cdf1)
        return delta_ecdf.max()

    def forward(self, y_pred, s):
        y_pred_t = y_pred.T

        def loss_fn(preds):
            return self.compute_loss_for_class(preds, s, self.temperature)

        losses = vmap(loss_fn)(y_pred_t)
        return losses.max()


class mcdp(erm):
    def __init__(self, args):
        super().__init__(args)
        self.fair_loss = MaxCDFdp(args.diff_temperature, num_classes=args.num_classes)

        self.num_classes = args.num_classes

        if args.num_classes > 2:
            self.fair_loss = MaxCDFdp_multiclass_vmap(args.diff_temperature)

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
            fairloss = args.diff_lambda * self.fair_loss(prob, sensitive_attr)

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
