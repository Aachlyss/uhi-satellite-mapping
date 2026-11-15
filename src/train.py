"""
Step 3: Training loop for the UHI U-Net (image-to-image temperature regression).

- Device   : Apple MPS > CUDA > CPU (via config.get_device()).
- Optimizer: Adam.
- Loss     : MSE on the Min-Max normalised [0, 1] target (switch to L1 with
             --loss l1 if preferred).
- Metric   : Mean Absolute Error (MAE) reported in real degrees Celsius. Because
             the target is linearly Min-Max scaled, MAE in Celsius is just the
             normalised MAE multiplied by (temp_max - temp_min).
- Output   : saves the best epoch (lowest validation MAE) to best_model.pth.

Run:  python -m src.train                  # defaults
      python -m src.train --epochs 50 --loss l1
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from . import config as C
from .dataset import get_dataloaders, load_stats
from .model import UNet

BEST_MODEL_PATH = C.PROJECT_ROOT / "best_model.pth"


def _run_epoch(model, loader, criterion, device, temp_range, optimizer=None):
    """Run one epoch. If optimizer is given -> train; else -> evaluate."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_abs_err = 0.0   # accumulated absolute error in Celsius
    n_pixels = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_px = y.numel()
        total_loss += loss.item() * batch_px
        # |pred - target| is in normalised units; * temp_range -> Celsius.
        total_abs_err += (pred - y).abs().sum().item() * temp_range
        n_pixels += batch_px

    avg_loss = total_loss / n_pixels
    mae_celsius = total_abs_err / n_pixels
    return avg_loss, mae_celsius


def train(epochs: int = 30, lr: float = 1e-3, loss_name: str = "mse"):
    device = C.get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(normalize_target=True)
    stats = load_stats()
    temp_range = max(stats["temp_max"] - stats["temp_min"], 1e-6)
    print(f"Temp range for MAE: {temp_range:.2f} C\n")

    model = UNet(in_channels=4, out_channels=1, final_activation="sigmoid").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() if loss_name == "l1" else nn.MSELoss()
    print(f"Optimizer: Adam(lr={lr}) | Loss: {loss_name.upper()}\n")

    best_mae = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = _run_epoch(
            model, train_loader, criterion, device, temp_range, optimizer
        )
        val_loss, val_mae = _run_epoch(
            model, val_loader, criterion, device, temp_range, optimizer=None
        )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "train_mae_c": train_mae, "val_mae_c": val_mae}
        )

        flag = ""
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_mae_c": val_mae,
                    "temp_min": stats["temp_min"],
                    "temp_max": stats["temp_max"],
                    "in_channels": 4,
                    "final_activation": "sigmoid",
                },
                BEST_MODEL_PATH,
            )
            flag = "  <- best (saved)"

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss {train_loss:.5f}  val_loss {val_loss:.5f} | "
            f"train_MAE {train_mae:5.2f} C  val_MAE {val_mae:5.2f} C{flag}"
        )

    with open(C.PROJECT_ROOT / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val MAE: {best_mae:.2f} C")
    print(f"Saved best model -> {BEST_MODEL_PATH}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the UHI U-Net.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["mse", "l1"], default="mse")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(epochs=args.epochs, lr=args.lr, loss_name=args.loss)
