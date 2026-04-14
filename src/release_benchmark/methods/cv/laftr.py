# Adapted from MEDFAIR: https://github.com/ys-zong/MEDFAIR/tree/main/models/LAFTR

import torch
import torch.nn as nn
from tqdm import tqdm

from release_benchmark.model import (
    Vanilla_ResNet18,
    Vanilla_ViT,
)
from release_benchmark.utils.common import AverageMeter

from .erm import erm


class LAFTRModel(nn.Module):
    """Wraps a backbone with an adversarial discriminator for LAFTR training."""

    def __init__(self, backbone, adversary_size=128, model_var="laftr-dp"):
        super().__init__()
        self.backbone = backbone
        self.fc = backbone.fc
        self.num_classes = backbone.fc.out_features
        self.sens_classes = 2  # we only have two sensitive attributes
        hidden_size = backbone.fc.in_features
        self.model_var = model_var
        if self.model_var != "laftr-dp":  # P(sensitive_attribute | features, label)
            self.adv_neurons = [
                hidden_size + self.num_classes,
                adversary_size,
                self.sens_classes - 1,
            ]
        else:  # P(sensitive_attribute | features), no label
            self.adv_neurons = [hidden_size, adversary_size, self.sens_classes - 1]

        self.num_adversaries_layers = len(self.adv_neurons)
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for one class label.
        self.discriminator = nn.ModuleList(
            [
                nn.Linear(self.adv_neurons[i], self.adv_neurons[i + 1])
                for i in range(self.num_adversaries_layers - 1)
            ]
        )

    def forward(self, x, target=None):
        logits, features = self.backbone.forward_return_feature(x)
        if target is None:
            return logits

        if (
            self.model_var != "laftr-dp"
        ):  # P(sensitive_attribute | features, label), use one-hot encoding for label since we have multiple classes
            target_onehot = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).float()
            features = torch.cat(
                [features, target_onehot],
                dim=1,
            )
        for hidden in self.discriminator:
            features = hidden(features)

        A_logits = torch.squeeze(features)
        return logits, A_logits


class laftr(erm):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.setmodel(args)
        self.net = [
            param
            for name, param in self.model.named_parameters()
            if "discriminator" not in name
        ]
        self.optimizer, self.optimizer_disc, self.fe_scheduler = (
            self.setoptimizer_laftr(self.model, args)
        )
        self.model_var = args.model_var

    def setmodel(self, args):
        in_channel = 3
        if "resnet" in args.model:
            backbone = Vanilla_ResNet18(
                num_classes=args.num_classes,
                in_channel=in_channel,
                pretrain=args.pretrain,
                freeze_layers=args.freeze,
                model=args.model,
            )
        elif args.model == "vit":
            backbone = Vanilla_ViT(
                num_classes=args.num_classes,
                in_channel=in_channel,
                pretrain=args.pretrain,
            )
        model = LAFTRModel(backbone, adversary_size=128, model_var=args.model_var)
        model = model.to(args.device)
        return model

    def setoptimizer_laftr(self, model, args):
        fe_scheduler = None

        if args.optim == "adam":
            optimizer = torch.optim.Adam(
                self.net, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer_disc = torch.optim.Adam(
                model.discriminator.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                self.net, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
            )
            optimizer_disc = torch.optim.SGD(
                model.discriminator.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "adamw":
            optimizer = torch.optim.AdamW(
                self.net, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer_disc = torch.optim.AdamW(
                model.discriminator.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError(f"unimplemented optimizer: {args.optim}")

        if args.optim == "sgd":
            fe_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.StepLR_size, gamma=args.gamma
            )
        return optimizer, optimizer_disc, fe_scheduler

    def get_AYweights(self, data):
        A_weights, Y_weights, AY_weights = (
            data.get_A_proportions(),
            data.get_Y_proportions(),
            data.get_AY_proportions(),
        )
        return A_weights, Y_weights, AY_weights

    def get_weighted_aud_loss(self, L, X, Y, A, A_wts, Y_wts, AY_wts):
        """Returns weighted discriminator loss"""
        if self.model_var == "laftr-dp":
            A0_wt = A_wts[0]
            A1_wt = A_wts[1]
            wts = A0_wt * (1 - A) + A1_wt * A
            wtd_L = L * torch.squeeze(wts)
        elif (  # does not fit multiclass classification
            self.model_var == "laftr-eqodd"
            or self.model_var == "laftr-eqopp0"
            or self.model_var == "laftr-eqopp1"
        ):
            A0_Y0_wt = AY_wts[0][0]
            A0_Y1_wt = AY_wts[0][1]
            A1_Y0_wt = AY_wts[1][0]
            A1_Y1_wt = AY_wts[1][1]

            if self.model_var == "laftr-eqodd":
                wts = (
                    A0_Y0_wt * (1 - A) * (1 - Y)
                    + A0_Y1_wt * (1 - A) * (Y)
                    + A1_Y0_wt * (A) * (1 - Y)
                    + A1_Y1_wt * (A) * (Y)
                )
            elif self.model_var == "laftr-eqopp0":
                wts = A0_Y0_wt * (1 - A) * (1 - Y) + A1_Y0_wt * (A) * (1 - Y)
            elif self.model_var == "laftr-eqopp1":
                wts = A0_Y1_wt * (1 - A) * (Y) + A1_Y1_wt * (A) * (Y)

            wtd_L = L * torch.squeeze(wts)
        else:
            raise Exception("Wrong model name")
            exit(0)

        return wtd_L

    def l1_loss(self, y, y_logits):
        """Returns l1 loss"""
        y_hat = torch.sigmoid(y_logits)
        return torch.squeeze(torch.abs(y - y_hat))

    def train(self, train_loader, epoch, args):
        A_weights, Y_weights, AY_weights = self.get_AYweights(train_loader.dataset)

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
            Y_logits, A_logits = model(data, target)
            output = Y_logits
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            if self.num_classes == 1:  # BCE Loss
                target = target.float()
                output = output.squeeze()

            class_loss = args.class_coeff * self.criterion(output, target)
            aud_loss = -args.fair_coeff * self.l1_loss(sensitive_attr, A_logits)
            weighted_aud_loss = self.get_weighted_aud_loss(
                aud_loss, data, target, sensitive_attr, A_weights, Y_weights, AY_weights
            )
            weighted_aud_loss = torch.mean(weighted_aud_loss)
            loss = class_loss + weighted_aud_loss

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.net, 5.0)
            for i in range(args.aud_steps):
                if i != args.aud_steps - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(), 5.0
                )
                self.optimizer_disc.step()

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
