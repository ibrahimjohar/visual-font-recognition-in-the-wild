"""
Loads data/synthetic via its manifest.csv rather than ImageFolder auto-discovery, so class
indices are stable and reproducible across resumed runs regardless of directory listing order,
and so train/val splits use the manifest's own split column instead of re-deriving one.

Returns (family_idx, role_idx) pairs per sample, not a single flat class index -- matches the
hierarchical family/role classifier in models/classifier.py. See that module's docstring for
why: a flat 3407-way (or 20-way, for the confusable smoke-test subset) head plateaued well
below its confusable-subset threshold even with an ArcFace margin, while a confusion-matrix
diagnostic showed the model was already getting family right 69% of the time and role right 89%
of the time in its top-5 guesses -- the joint prediction was the hard part, not either piece
alone. Decomposing into two heads lets each learn its own, easier problem directly.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent

# ImageNet normalization -- matches the pretrained EfficientNet-B0 weights' expected input.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# light live augmentation only -- the dataset already bakes in rotation/perspective/blur/noise
# diversity at generation time (see data_gen/render.py); duplicating that here is redundant,
# not additive. This is just enough jitter to avoid the model memorizing exact pixel layouts.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_class_list(manifest_csv: str) -> list[str]:
    """Sorted, deterministic class_name -> index mapping. Kept for anything that still wants a
    flat (family, role) label (e.g. the hard-subset selection logic, which operates on class
    names), but training itself now uses load_family_role_lists below."""
    classes = set()
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            classes.add(row["class_name"])
    return sorted(classes)


def load_family_role_lists(manifest_csv: str) -> tuple[list[str], list[str]]:
    """Sorted, deterministic family list and role list, built once from the manifest so index
    assignment is identical across every process/resume that reads the same manifest file."""
    families, roles = set(), set()
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            families.add(row["family"])
            roles.add(row["role"])
    return sorted(families), sorted(roles)


class FontDataset(Dataset):
    def __init__(self, manifest_csv: str, data_dir: str, split: str,
                 family_to_idx: dict[str, int], role_to_idx: dict[str, int],
                 class_filter: set[str] | None = None, transform=None):
        """
        split: "train" or "val", matches the manifest's split column.
        class_filter: if given, only include rows whose class_name is in this set -- used by
        the hard-subset smoke test to train on a slice of classes without touching disk layout.
        """
        self.data_dir = Path(data_dir)
        self.family_to_idx = family_to_idx
        self.role_to_idx = role_to_idx
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)

        self.samples: list[tuple[str, int, int]] = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                if class_filter is not None and row["class_name"] not in class_filter:
                    continue
                self.samples.append((
                    row["path"],
                    self.family_to_idx[row["family"]],
                    self.role_to_idx[row["role"]],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        rel_path, family_idx, role_idx = self.samples[idx]
        img = Image.open(self.data_dir / rel_path).convert("RGB")
        return self.transform(img), family_idx, role_idx
