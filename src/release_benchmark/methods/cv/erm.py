import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from release_benchmark.metrics.fairness_metrics import calculate_metrics
from release_benchmark.model import (
    Vanilla_ResNet18,
    Vanilla_ViT,
)
from release_benchmark.utils.common import AverageMeter

_OPTIM_REGISTRY = {
    "adam": lambda params, args: torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
    ),
    "sgd": lambda params, args: torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    ),
    "adamw": lambda params, args: torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
    ),
    "adagrad": lambda params, args: torch.optim.Adagrad(
        params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
    ),
    "adadelta": lambda params, args: torch.optim.Adadelta(
        params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7
    ),
}


class erm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = self.setmodel(args)
        self.optimizer, self.fe_scheduler = self.setoptimizer(self.model, args)
        self.args = args
        self.num_classes = args.num_classes
        self.criterion = (
            nn.BCEWithLogitsLoss() if args.num_classes == 1 else nn.CrossEntropyLoss()
        )

    def setmodel(self, args):
        in_channel = 3
        if "resnet" in args.model:
            model = Vanilla_ResNet18(
                num_classes=args.num_classes,
                in_channel=in_channel,
                pretrain=args.pretrain,
                freeze_layers=args.freeze,
                model=args.model,
            )
        elif args.model == "vit":
            model = Vanilla_ViT(
                num_classes=args.num_classes,
                in_channel=in_channel,
                pretrain=args.pretrain,
            )
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        return model.to(args.device)

    def setoptimizer(self, model, args):
        factory = _OPTIM_REGISTRY.get(args.optim)
        if factory is None:
            raise ValueError(
                f"Unsupported optimizer: {args.optim}. "
                f"Choose from {list(_OPTIM_REGISTRY)}"
            )
        optimizer = factory(model.parameters(), args)
        fe_scheduler = None
        if args.optim == "sgd":
            fe_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.StepLR_size, gamma=args.gamma
            )
        return optimizer, fe_scheduler

    def validate(self, val_loader, epoch, args):
        model = self.model
        device = args.device
        model.eval()
        val_loss = AverageMeter()
        tol_output, tol_target, tol_sensitive, tol_index = [], [], [], []
        with torch.no_grad():
            for data, target, sensitive_attr in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if self.num_classes == 1:
                    target = target.float()
                    output = output.squeeze()
                    prob = torch.sigmoid(output).flatten()
                else:
                    prob = F.softmax(output, dim=-1)
                loss = self.criterion(output, target)
                if loss.dim() > 0:
                    val_loss.update(loss.mean().item())
                else:
                    val_loss.update(loss.item())

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
                data, target = data.to(device), target.to(device)
                output = model(data)

                if args.num_classes == 1:
                    target = target.float()
                    prob = torch.sigmoid(output).flatten()
                    output = output.squeeze()
                else:
                    prob = F.softmax(output, dim=-1)

                loss = self.criterion(output, target)
                if loss.dim() > 0:
                    test_loss.update(loss.mean().item())
                else:
                    test_loss.update(loss.item())

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
                target = target.float()
            output = output.squeeze()
            loss = self.criterion(output, target)
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
