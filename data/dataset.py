"""
DeepTrace Dataset Pipeline
Handles multi-generator image dataset loading, augmentation, and splitting.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

CLASS_TO_IDX: Dict[str, int] = {
    "stable_diffusion": 0,
    "midjourney":       1,
    "dalle3":           2,
    "flux":             3,
    "real":             4,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

CATEGORY_NAMES = ["animals", "city", "food", "nature", "people"]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic transforms for production inference."""
    return get_val_transforms(image_size)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepTraceDataset(Dataset):
    """
    Loads images from a manifest CSV with columns:
        path, source, category, split

    Expects directory structure:
        data/raw/
            stable_diffusion/<category>/<image>.jpg
            midjourney/<category>/<image>.jpg
            dalle3/<category>/<image>.jpg
            flux/<category>/<image>.jpg
            real/<category>/<image>.jpg
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        data_root: str = "data",
    ):
        self.data_root = Path(data_root)
        self.transform = transform or get_val_transforms()

        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {manifest_path}")

        print(f"[DeepTraceDataset] split={split} | {len(self.df)} samples")
        print(self.df["source"].value_counts().to_string())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.data_root / row["path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open {img_path}: {e}")

        label = CLASS_TO_IDX[row["source"]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute per-class inverse-frequency weights for WeightedRandomSampler."""
        counts = self.df["source"].map(CLASS_TO_IDX).value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for use with WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        labels = self.df["source"].map(CLASS_TO_IDX).values
        return class_weights[labels]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    manifest_path: str,
    data_root: str = "data",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Returns {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    train_dataset = DeepTraceDataset(
        manifest_path, split="train",
        transform=get_train_transforms(image_size), data_root=data_root
    )
    val_dataset = DeepTraceDataset(
        manifest_path, split="val",
        transform=get_val_transforms(image_size), data_root=data_root
    )
    test_dataset = DeepTraceDataset(
        manifest_path, split="test",
        transform=get_val_transforms(image_size), data_root=data_root
    )

    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ---------------------------------------------------------------------------
# Manifest builder (run once to create the CSV from raw data directory)
# ---------------------------------------------------------------------------

def build_manifest(data_root: str = "data/raw", output_path: str = "data/manifest.csv",
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   seed: int = 42) -> pd.DataFrame:
    """
    Scans data/raw/<source>/<category>/<image> and creates a stratified manifest CSV.
    Run this once after downloading the dataset.
    """
    from sklearn.model_selection import train_test_split

    records = []
    data_path = Path(data_root)

    for source_dir in sorted(data_path.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        if source not in CLASS_TO_IDX:
            print(f"  Skipping unknown source directory: {source}")
            continue

        for category_dir in sorted(source_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name

            for img_file in sorted(category_dir.glob("*")):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    records.append({
                        "path": str(img_file.relative_to(Path(data_root).parent)),
                        "source": source,
                        "category": category,
                        "split": "train",   # will be overwritten below
                    })

    df = pd.DataFrame(records)
    print(f"Total images found: {len(df)}")
    print(df.groupby(["source", "category"]).size().to_string())

    # Stratified split by (source, category)
    strat_col = df["source"] + "_" + df["category"]

    train_idx, temp_idx = train_test_split(
        df.index, test_size=(val_ratio + test_ratio),
        stratify=strat_col, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=strat_col.iloc[temp_idx], random_state=seed
    )

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx,   "split"] = "val"
    df.loc[test_idx,  "split"] = "test"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nManifest saved to {output_path}")
    print(df["split"].value_counts().to_string())
    return df
