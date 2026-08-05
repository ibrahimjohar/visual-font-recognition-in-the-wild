"""
EfficientNet-B0 classifier for VFRW Stage C. Pretrained backbone via timm, head replaced for
the font class count. Freeze/unfreeze helpers implement the 2-phase progressive-unfreezing
plan: phase 1 trains only the head (backbone fully frozen, no backward graph through it),
phase 2 additionally unfreezes the last N MBConv blocks.
"""

import timm
import torch.nn as nn


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: everything except the classifier head is frozen."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """Phase 2: unfreeze the head plus the last n_blocks of model.blocks (timm's EfficientNet
    exposes stages as model.blocks, a Sequential of 7 stages for B0). Early stages stay frozen
    -- they encode generic edge/texture features that transfer fine from ImageNet; font
    discrimination lives in the later, more shape-specific stages."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    total_stages = len(model.blocks)
    unfreeze_from = max(0, total_stages - n_blocks)
    for stage_idx in range(unfreeze_from, total_stages):
        for param in model.blocks[stage_idx].parameters():
            param.requires_grad = True

    # conv_head/bn2 sit between the last block and the classifier -- unfreeze them too when
    # any backbone blocks are trainable, otherwise the head is fine-tuning on frozen features
    # right up until the last block, which is an awkward halfway point.
    for name, param in model.named_parameters():
        if name.startswith("conv_head") or name.startswith("bn2"):
            param.requires_grad = True


def param_groups_for_phase(model: nn.Module, lr_head: float, lr_backbone: float | None) -> list[dict]:
    """Discriminative LR: head gets lr_head, any trainable backbone params get lr_backbone.
    Only includes params with requires_grad=True, so call this after freeze/unfreeze."""
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]

    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        assert lr_backbone is not None, "backbone has trainable params but no lr_backbone given"
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
