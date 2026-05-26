"""The segmentation CNN: a compact U-Net that maps one eye's raw RGB to
per-pixel class logits (0 background, 1 grass blade, 2 banana).

This is the learned perception layer of the heuristic+CNN controller. It is
imported by both the trainer (`train_seg_cnn.py`) and the controller
integration, so the architecture lives in one place.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 3  # 0 background, 1 grass blade, 2 banana


class _ConvBlock(nn.Module):
    """(Conv 3x3 - BN - ReLU) x2, spatial size preserved."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SegCNN(nn.Module):
    """Compact 3-level U-Net.

    Input  : (B, in_ch, H, W) float in [0, 1]  — one eye's RGB.
    Output : (B, n_classes, H, W) logits.

    H, W must be divisible by 8. At deployment we run it at 128x128.
    """

    def __init__(self, in_ch: int = 3, n_classes: int = N_CLASSES,
                 base: int = 32):
        super().__init__()
        self.config = dict(in_ch=in_ch, n_classes=n_classes, base=base)

        self.enc1 = _ConvBlock(in_ch, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)                       # (base,   H,   W)
        e2 = self.enc2(self.pool(e1))           # (2base,  H/2, W/2)
        e3 = self.enc3(self.pool(e2))           # (4base,  H/4, W/4)
        b = self.bottleneck(self.pool(e3))      # (8base,  H/8, W/8)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                    # (n_classes, H, W)


def save_seg_cnn(model: SegCNN, path: str):
    torch.save({"state_dict": model.state_dict(), "config": model.config}, path)


def load_seg_cnn(path: str, device: str = "cpu") -> SegCNN:
    ckpt = torch.load(path, map_location=device)
    model = SegCNN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
