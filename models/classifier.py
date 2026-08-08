"""
EfficientNet-B0 embedding backbone + hierarchical family/role ArcFace heads for VFRW Stage C.

History: plain softmax plateaued at 30% top-5 on the confusable-font subset. Switched to a
single flat ArcFace head over all classes, which only nudged it to 33.9% top-5, still far below
the 50% threshold. A confusion-matrix diagnostic on that result showed the real story: the true
FAMILY appeared in the top-5 predicted families 69.3% of the time, and the true ROLE appeared in
the top-5 predicted roles 89.3% of the time -- both individually strong -- but getting family
AND role exactly right simultaneously, within 5 guesses out of a flat 20-way (or 3407-way, same
result either way) space, was the hard part. That's the textbook signature for a hierarchical
classifier: decompose one hard joint problem into two problems the model has already mostly
solved independently.

Architecture: one shared backbone embedding feeds two separate ArcFace heads --
  - family_classifier: ArcMarginProduct over every distinct font family in the dataset (~1976
    families). This is the "which typeface" problem -- genuinely distinct visual entities.
  - role_classifier: ArcMarginProduct over just 4 classes (Regular/Bold/Italic/BoldItalic),
    trained GLOBALLY across every family's images, not per-family. This turns "distinguish bold
    from regular" from a low-data problem scattered across thousands of narrow per-family
    buckets into one well-populated 4-way problem with the full dataset behind it.

Final (family, role) prediction combines both heads' scores at inference (see
training/train.py's hierarchical evaluation) rather than predicting the flat class directly.

Freeze/unfreeze helpers implement the same 2-phase progressive-unfreezing plan as before:
phase 1 trains only the two heads (backbone fully frozen), phase 2 additionally unfreezes the
last N MBConv blocks. Both heads share the "classifier" name prefix (family_classifier /
role_classifier) so the existing name-based freeze/unfreeze/param-group logic needs no changes
beyond that shared prefix.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# timm is imported lazily, inside FontEmbeddingModel.__init__, not here at module level. On
# Windows, DataLoader worker processes (spawn, not fork) re-run this module's top-level imports
# even though workers only ever touch the Dataset class, never the model -- a module-level timm
# import was dragging its full transitive dependency chain (huggingface_hub, filelock, etc.)
# into every worker's memory for nothing, which combined with a tight system RAM budget produced
# a real MemoryError when num_workers was raised above 0.


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
    head from timm) feeding into two separate ArcMarginProduct heads: family and role. forward()
    takes labels only during training (family_labels, role_labels both required together, or
    both None for eval); eval/inference calls it with labels=None for plain cosine-similarity
    ranking on both heads."""

    def __init__(self, num_families: int, num_roles: int = 4, pretrained: bool = True,
                 embedding_dim: int = 1280, arc_s: float = 30.0, arc_m: float = 0.30):
        super().__init__()
        import timm  # see the module-level comment on the lazy import
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        self.family_classifier = ArcMarginProduct(embedding_dim, num_families, s=arc_s, m=arc_m)
        self.role_classifier = ArcMarginProduct(embedding_dim, num_roles, s=arc_s, m=arc_m)

    def forward(self, x: torch.Tensor, family_labels: torch.Tensor | None = None,
                role_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        family_logits = self.family_classifier(features, family_labels)
        role_logits = self.role_classifier(features, role_labels)
        return family_logits, role_logits


def build_model(num_families: int, num_roles: int = 4, pretrained: bool = True) -> nn.Module:
    return FontEmbeddingModel(num_families, num_roles, pretrained=pretrained)


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: everything except the two ArcFace heads is frozen."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("family_classifier") or name.startswith("role_classifier")


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """Phase 2: unfreeze both heads plus the last n_blocks of the backbone's stages (timm's
    EfficientNet exposes stages as backbone.blocks, a Sequential of 7 stages for B0). Early
    stages stay frozen -- they encode generic edge/texture features that transfer fine from
    ImageNet; font discrimination lives in the later, more shape-specific stages."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("family_classifier") or name.startswith("role_classifier")

    total_stages = len(model.backbone.blocks)
    unfreeze_from = max(0, total_stages - n_blocks)
    for stage_idx in range(unfreeze_from, total_stages):
        for param in model.backbone.blocks[stage_idx].parameters():
            param.requires_grad = True

    # conv_head/bn2 sit between the last block and the pooled feature output -- unfreeze them
    # too when any backbone blocks are trainable, otherwise the heads are fine-tuning on frozen
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
    """Discriminative LR: both heads get lr_head, any trainable backbone params get lr_backbone.
    Only includes params with requires_grad=True, so call this after freeze/unfreeze."""
    is_head = lambda n: n.startswith("family_classifier") or n.startswith("role_classifier")
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and is_head(n)]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not is_head(n)]

    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        assert lr_backbone is not None, "backbone has trainable params but no lr_backbone given"
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
