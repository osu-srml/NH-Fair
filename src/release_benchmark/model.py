"""Backbone model definitions for the NH-Fair benchmark."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import BasicBlock, Bottleneck

# ---------------------------------------------------------------------------
# ResNet (custom copy supporting weight init control)
# ---------------------------------------------------------------------------


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: type[BasicBlock | Bottleneck],
    layers: list[int],
    weights: WeightsEnum | None,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = ResNet(block, layers, **kwargs)
    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )
    return model


# ---------------------------------------------------------------------------
# Lookup table for torchvision ResNet constructors
# ---------------------------------------------------------------------------

_RESNET_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def _build_resnet_backbone(model_name: str, pretrain: bool):
    builder = _RESNET_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Unknown ResNet variant: {model_name}")
    weights = "IMAGENET1K_V1" if pretrain else None
    return builder(weights=weights)


# ---------------------------------------------------------------------------
# Vanilla ResNet wrapper (supports 18/34/50/101/152)
# ---------------------------------------------------------------------------


class Vanilla_ResNet18(nn.Module):
    def __init__(
        self,
        num_classes=7,
        in_channel=3,
        pretrain=False,
        freeze_layers=0,
        model="resnet18",
    ):
        super().__init__()
        backbone = _build_resnet_backbone(model, pretrain)
        self.backbone = backbone
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = backbone.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        if in_channel == 1:
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.freeze_layers(freeze_layers)

    def freeze_layers(self, freeze_layers):
        layers = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    def forward_return_feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        logits = self.fc(feature)
        return logits, feature

    def forward_return_feature_mixup(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ---------------------------------------------------------------------------
# Per-group classifiers
# ---------------------------------------------------------------------------


class Decoupled_ResNet18(nn.Module):
    def __init__(
        self,
        num_classes=7,
        in_channel=3,
        pretrain=False,
        sensitive_attributes=2,
        model="resnet18",
    ):
        super().__init__()
        backbone = _build_resnet_backbone(model, pretrain)
        self.backbone = backbone
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = backbone.fc.in_features
        self.decoupled_classifiers = nn.ModuleDict()
        self.sensitive_attributes = sensitive_attributes
        self.num_classes = num_classes
        for i in range(sensitive_attributes):
            self.decoupled_classifiers[str(i)] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, sa):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        fe_outputs = torch.zeros(x.size(0), self.num_classes).to(x.device)
        for i in range(self.sensitive_attributes):
            current_idx = (sa == i).nonzero().squeeze()
            if current_idx.numel() > 0:
                fe_outputs[current_idx] = self.decoupled_classifiers[str(i)](
                    x[current_idx]
                )
        return fe_outputs


# ---------------------------------------------------------------------------
# ViT wrapper
# ---------------------------------------------------------------------------


class Vanilla_ViT(nn.Module):
    def __init__(self, num_classes=7, in_channel=3, pretrain=False):
        super().__init__()
        vit_b_16 = models.vit_b_16(pretrained=pretrain)
        self.vit_b_16 = vit_b_16
        self.patch_embed = vit_b_16.conv_proj
        self.cls_token = vit_b_16.class_token
        self.encoder = vit_b_16.encoder
        num_ftrs = vit_b_16.heads.head.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def _extract_features(self, x: torch.Tensor):
        x = self.vit_b_16._process_input(x)
        n = x.shape[0]
        batch_class_token = self.cls_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor):
        feature = self._extract_features(x)
        return self.fc(feature)

    def forward_return_feature(self, x: torch.Tensor):
        feature = self._extract_features(x)
        logits = self.fc(feature)
        return logits, feature
