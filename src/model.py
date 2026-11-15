"""
Step 2: U-Net architecture for image-to-image temperature regression.

Input : 4 channels (Red, Green, Blue, Near-Infrared), 256 x 256
Output: 1 channel  (predicted land surface temperature map)

This is a classic U-Net (encoder / bottleneck / decoder with skip connections).
Because the target temperature is Min-Max scaled into [0, 1] during Step 1, the
final layer uses a Sigmoid activation so outputs stay in that range; Step 3 then
de-normalises predictions back to degrees Celsius for interpretable MAE.
Set `final_activation="identity"` if you ever train on raw (unscaled) Celsius.

Run a quick shape/device check:  python -m src.model
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2, the core block of the U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale with maxpool then a DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscale, concatenate the matching skip connection, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels = channels coming from the deeper layer (before concat).
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation we have (in_channels // 2 + skip_channels) channels.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if odd input sizes caused a 1-pixel mismatch (robustness).
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for 4-channel optical -> 1-channel temperature regression."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_features: int = 64,
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        f = base_features

        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        # Bottleneck
        self.down4 = Down(f * 8, f * 16)
        # Decoder
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)
        # Regression head (1x1 conv -> single channel)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

        if final_activation == "sigmoid":
            self.final = nn.Sigmoid()
        elif final_activation in ("identity", "linear", None):
            self.final = nn.Identity()
        else:
            raise ValueError(f"Unknown final_activation: {final_activation}")

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)      # bottleneck
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.final(x)


if __name__ == "__main__":
    device = C.get_device()
    print(f"Using device: {device}")

    model = UNet(in_channels=4, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    dummy = torch.randn(2, 4, C.PATCH_SIZE, C.PATCH_SIZE, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  min={out.min():.3f} max={out.max():.3f}")
