"""
EfficientNet-B0 embedding backbone + ArcFace head for VFRW Stage C.

Switched from a plain linear classifier to ArcFace after Check B showed the flat-softmax
approach plateauing around 30% top-5 on the hand-picked confusable-font subset (well below
the 50% pre-committed threshold), while a broad 500-class random subset was still improving
normally. That split result is the specific signature of a margin problem, not a capacity
problem: plain softmax only needs *some* positive margin to classify correctly, so it has no
training pressure to widen the decision boundary between near-identical fonts once average
accuracy looks fine. ArcFace adds an explicit angular margin to the true class's logit during
training, forcing the backbone to learn representations where confusable classes are pushed
apart by a real geometric gap. Same technique used in face recognition for the analogous
problem (huge class count, many near-identical identities).

Freeze/unfreeze helpers implement the same 2-phase progressive-unfreezing plan as before:
phase 1 trains only the ArcFace head (backbone fully frozen), phase 2 additionally unfreezes
the last N MBConv blocks.
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """Standard ArcFace head: cosine similarity between L2-normalized features and L2-normalized
    class weight vectors, with an additive angular margin m applied to the true class during
    training (label given) and no margin at inference (label=None, plain scaled cosine
    similarity -- used for eval/top-k ranking, where there's no "true class" to margin against).
    s scales the cosine values into a range cross-entropy can actually push gradients through;
    without it, cosine similarity's [-1, 1] range makes the softmax dangerously close to
    uniform regardless of how separated the classes actually are.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        # if cosine is already so far off (> pi - m) that adding the margin would wrap around
        # past pi, fall back to a linear penalty instead of the wrapped (wrong-direction) value
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class FontEmbeddingModel(nn.Module):
    """Backbone (num_classes=0 -> returns the pooled 1280-d feature vector directly, no linear
    head from timm) feeding into an ArcMarginProduct head. forward() takes labels only during
    training; eval/inference calls it with labels=None for plain cosine-similarity ranking."""

    def __init__(self, num_classes: int, pretrained: bool = True, embedding_dim: int = 1280,
                 arc_s: float = 30.0, arc_m: float = 0.30):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        self.classifier = ArcMarginProduct(embedding_dim, num_classes, s=arc_s, m=arc_m)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features, labels)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return FontEmbeddingModel(num_classes, pretrained=pretrained)


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: everything except the ArcFace head is frozen."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """Phase 2: unfreeze the head plus the last n_blocks of the backbone's stages (timm's
    EfficientNet exposes stages as backbone.blocks, a Sequential of 7 stages for B0). Early
    stages stay frozen -- they encode generic edge/texture features that transfer fine from
    ImageNet; font discrimination lives in the later, more shape-specific stages."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    total_stages = len(model.backbone.blocks)
    unfreeze_from = max(0, total_stages - n_blocks)
    for stage_idx in range(unfreeze_from, total_stages):
        for param in model.backbone.blocks[stage_idx].parameters():
            param.requires_grad = True

    # conv_head/bn2 sit between the last block and the pooled feature output -- unfreeze them
    # too when any backbone blocks are trainable, otherwise the head is fine-tuning on frozen
    # features right up until the last block, which is an awkward halfway point.
    for name, param in model.named_parameters():
        if name.startswith("backbone.conv_head") or name.startswith("backbone.bn2"):
            param.requires_grad = True


def freeze_bn_stats(model: nn.Module) -> None:
    """model.train() puts every submodule into training mode regardless of requires_grad --
    freezing a layer's parameters does NOT stop its BatchNorm from recomputing per-batch
    statistics and updating running_mean/running_var. For a frozen backbone that's wrong: it
    should use the fixed pretrained running stats, not noisy current-batch statistics, which
    at small batch sizes can hit a near-zero-variance channel and produce NaN outputs. Call
    this every time after model.train() in the training loop, for both phases -- it puts
    BatchNorm layers belonging to any currently-frozen parameters back into eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            frozen = all(not p.requires_grad for p in module.parameters(recurse=False)) \
                if list(module.parameters(recurse=False)) else True
            if frozen:
                module.eval()


def param_groups_for_phase(model: nn.Module, lr_head: float, lr_backbone: float | None) -> list[dict]:
    """Discriminative LR: head gets lr_head, any trainable backbone params get lr_backbone.
    Only includes params with requires_grad=True, so call this after freeze/unfreeze."""
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier")]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")]

    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        assert lr_backbone is not None, "backbone has trainable params but no lr_backbone given"
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
