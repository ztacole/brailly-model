"""BraillyNet: anchor-free (CenterNet-style) Braille cell detector + dot classifier.

Input : RGB image, float32 in [0, 1], shape (B, 3, H, W), H and W multiples of 32.
Output: tensor (B, 11, H/4, W/4)
    [0]     cell-centre heatmap logit
    [1:7]   dot 1..6 logits (read at detected centres)
    [7:9]   centre sub-pixel offset (x, y) in output-stride units
    [9:11]  log cell size (w, h) in output-stride units

Predicting the six dots as independent binary outputs (instead of 63 classes)
lets the model generalise to rare patterns and keeps the head tiny.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STRIDE = 4
OUT_CH = 11
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _cbr(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BraillyNet(nn.Module):
    def __init__(self, pretrained: bool = True, width: int = 64):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.mobilenet_v3_large(weights=weights).features
        self.s4, self.s8, self.s16, self.s32 = f[:4], f[4:7], f[7:13], f[13:16]
        self.lat32 = _cbr(160, width, 1)
        self.lat16 = _cbr(112, width, 1)
        self.lat8 = _cbr(40, width, 1)
        self.lat4 = _cbr(24, width, 1)
        self.smooth16 = _cbr(width, width)
        self.smooth8 = _cbr(width, width)
        self.smooth4 = _cbr(width, width)
        self.head = nn.Sequential(_cbr(width, width), _cbr(width, width), nn.Conv2d(width, OUT_CH, 1))
        nn.init.constant_(self.head[-1].bias, 0.0)
        with torch.no_grad():
            self.head[-1].bias[0] = -2.19  # prior p=0.1 for the heatmap
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        c4 = self.s4(x)
        c8 = self.s8(c4)
        c16 = self.s16(c8)
        c32 = self.s32(c16)
        p = self.lat32(c32)
        p = self.smooth16(self.lat16(c16) + F.interpolate(p, size=c16.shape[-2:], mode="nearest"))
        p = self.smooth8(self.lat8(c8) + F.interpolate(p, size=c8.shape[-2:], mode="nearest"))
        p = self.smooth4(self.lat4(c4) + F.interpolate(p, size=c4.shape[-2:], mode="nearest"))
        return self.head(p)
