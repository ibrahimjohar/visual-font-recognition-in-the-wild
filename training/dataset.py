"""
Loads data/synthetic via its manifest.csv rather than ImageFolder auto-discovery, so class
indices are stable and reproducible across resumed runs regardless of directory listing order,
and so train/val splits use the manifest's own split column instead of re-deriving one.
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
    """Sorted, deterministic class_name -> index mapping, built once from the manifest so it's
    identical across every process/resume that reads the same manifest file."""
    classes = set()
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            classes.add(row["class_name"])
    return sorted(classes)


class FontDataset(Dataset):
    def __init__(self, manifest_csv: str, data_dir: str, split: str,
                 class_to_idx: dict[str, int], class_filter: set[str] | None = None,
                 transform=None):
        """
        split: "train" or "val", matches the manifest's split column.
        class_filter: if given, only include rows whose class_name is in this set -- used by
        the hard-subset smoke test to train on a slice of classes without touching disk layout.
        """
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)

        self.samples: list[tuple[str, int]] = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                if class_filter is not None and row["class_name"] not in class_filter:
                    continue
                self.samples.append((row["path"], self.class_to_idx[row["class_name"]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        img = Image.open(self.data_dir / rel_path).convert("RGB")
        return self.transform(img), label
