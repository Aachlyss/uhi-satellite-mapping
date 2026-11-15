"""
Step 1b: PyTorch Dataset + DataLoaders for the UHI regression task.

`UHIDataset` loads the pre-cropped .npz patches produced by `src/preprocess.py`.
The optical inputs are already Min-Max scaled into [0, 1] (surface reflectance).
The target temperature is stored in Celsius and Min-Max normalised to [0, 1] on
the fly using the global min/max from dataset/stats.json, so a Sigmoid-headed
U-Net can predict it. Use `denormalize_temp` to map predictions back to Celsius.

Run a quick sanity check:  python -m src.dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from . import config as C


def load_stats() -> dict:
    if not C.STATS_PATH.exists():
        raise FileNotFoundError(
            f"{C.STATS_PATH} not found. Run `python -m src.preprocess` first."
        )
    with open(C.STATS_PATH) as f:
        return json.load(f)


class UHIDataset(Dataset):
    """Reads 4-channel optical patches (x) and 1-channel temperature maps (y)."""

    def __init__(
        self,
        patch_paths,
        temp_min: float,
        temp_max: float,
        normalize_target: bool = True,
    ):
        self.patch_paths = [Path(p) for p in patch_paths]
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.temp_range = max(self.temp_max - self.temp_min, 1e-6)
        self.normalize_target = normalize_target

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int):
        with np.load(self.patch_paths[idx]) as data:
            x = data["x"].astype(np.float32)   # [4, H, W] reflectance in [0, 1]
            y = data["y"].astype(np.float32)   # [H, W] Celsius

        if self.normalize_target:
            y = (y - self.temp_min) / self.temp_range  # -> [0, 1]

        x = torch.from_numpy(x)                        # [4, H, W]
        y = torch.from_numpy(y).unsqueeze(0)           # [1, H, W]
        return x, y

    def denormalize_temp(self, y_norm):
        """Map a normalised [0, 1] target/prediction back to degrees Celsius."""
        return y_norm * self.temp_range + self.temp_min


def get_dataloaders(
    batch_size: int = C.BATCH_SIZE,
    val_split: float = C.VAL_SPLIT,
    num_workers: int = C.NUM_WORKERS,
    normalize_target: bool = True,
):
    """Build train/validation DataLoaders from the generated patches."""
    stats = load_stats()
    patch_paths = sorted(C.PATCH_DIR.glob("*.npz"))
    if not patch_paths:
        raise FileNotFoundError(
            f"No patches in {C.PATCH_DIR}. Run `python -m src.preprocess` first."
        )

    train_paths, val_paths = train_test_split(
        patch_paths, test_size=val_split, random_state=C.RANDOM_SEED, shuffle=True
    )

    train_ds = UHIDataset(train_paths, stats["temp_min"], stats["temp_max"], normalize_target)
    val_ds = UHIDataset(val_paths, stats["temp_min"], stats["temp_max"], normalize_target)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    print(f"Patches: {len(patch_paths)} | train: {len(train_ds)} | val: {len(val_ds)}")
    print(f"Temp range: {stats['temp_min']:.2f} C .. {stats['temp_max']:.2f} C")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"\nBatch sanity check:")
    print(f"  x: {tuple(x.shape)}  dtype={x.dtype}  min={x.min():.3f} max={x.max():.3f}")
    print(f"  y: {tuple(y.shape)}  dtype={y.dtype}  min={y.min():.3f} max={y.max():.3f}")
