"""Datasets (synthetic + real annotated photos) and training-target encoding."""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import STRIDE
from .synth import SynthConfig, render_sample


# ---------------------------------------------------------------- targets
def encode_targets(cells: np.ndarray, h: int, w: int, stride: int = STRIDE):
    """cells (N,5) x0,y0,x1,y1,mask in input px -> dense targets at stride."""
    oh, ow = h // stride, w // stride
    heat = np.zeros((oh, ow), np.float32)
    dots = np.zeros((6, oh, ow), np.float32)
    dmask = np.zeros((oh, ow), np.float32)
    reg = np.zeros((4, oh, ow), np.float32)     # off_x, off_y, log_w, log_h
    rmask = np.zeros((oh, ow), np.float32)
    ys, xs = np.mgrid[0:oh, 0:ow].astype(np.float32)
    for x0, y0, x1, y1, m in cells:
        m = int(m)
        cx, cy = (x0 + x1) / 2 / stride, (y0 + y1) / 2 / stride
        bw, bh = max(1e-2, (x1 - x0) / stride), max(1e-2, (y1 - y0) / stride)
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < ow and 0 <= iy < oh):
            continue
        sig = max(0.5, 0.12 * min(bw, bh) * 1.6)
        r = int(math.ceil(3 * sig))
        xa, xb, ya, yb = max(0, ix - r), min(ow, ix + r + 1), max(0, iy - r), min(oh, iy + r + 1)
        g = np.exp(-((xs[ya:yb, xa:xb] - ix) ** 2 + (ys[ya:yb, xa:xb] - iy) ** 2) / (2 * sig * sig))
        heat[ya:yb, xa:xb] = np.maximum(heat[ya:yb, xa:xb], g)
        bits = np.array([(m >> k) & 1 for k in range(6)], np.float32)
        dx0, dx1, dy0, dy1 = max(0, ix - 1), min(ow, ix + 2), max(0, iy - 1), min(oh, iy + 2)
        dots[:, dy0:dy1, dx0:dx1] = bits[:, None, None]
        dmask[dy0:dy1, dx0:dx1] = 1.0
        reg[:, iy, ix] = (cx - ix, cy - iy, math.log(bw), math.log(bh))
        rmask[iy, ix] = 1.0
    return heat, dots, dmask, reg, rmask


def to_tensor_sample(img_bgr: np.ndarray, cells: np.ndarray):
    h, w = img_bgr.shape[:2]
    heat, dots, dmask, reg, rmask = encode_targets(cells, h, w)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (x, torch.from_numpy(heat), torch.from_numpy(dots), torch.from_numpy(dmask),
            torch.from_numpy(reg), torch.from_numpy(rmask))


# ---------------------------------------------------------------- synthetic
class SynthDataset(Dataset):
    def __init__(self, length: int, size: int = 384, seed: int = 0, cfg: Optional[SynthConfig] = None):
        self.length = length
        self.cfg = cfg or SynthConfig(out_h=size, out_w=size)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = random.Random((self.seed * 1_000_003 + self.epoch) * 10_000_019 + i)
        img, cells = render_sample(rng, self.cfg)
        return to_tensor_sample(img, cells)


# ---------------------------------------------------------------- real data
def read_csv_annotation(path: str, w: int, h: int) -> np.ndarray:
    """Angelina-style CSV: left;top;right;bottom;label  (coords in [0,1])."""
    rows = []
    if not os.path.exists(path):
        return np.zeros((0, 5), np.float32)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(";")
            if len(p) < 5:
                continue
            l, t, r, b = map(float, p[:4])
            m = int(p[4])
            if 1 <= m <= 63:
                rows.append((l * w, t * h, r * w, b * h, m))
    return np.float32(rows).reshape(-1, 5)


def read_image_list(list_file: str) -> List[str]:
    base = os.path.dirname(os.path.abspath(list_file))
    out = []
    with open(list_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if line:
                out.append(os.path.join(base, line))
    return out


def load_real(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    stem = os.path.splitext(img_path)[0]
    return img, read_csv_annotation(stem + ".csv", w, h)


class RealDataset(Dataset):
    """Random crops of annotated photos with geometric/photometric augmentation.

    Expects lists of image paths, each with an Angelina-format ``.csv`` next to
    it (images without a csv are treated as negatives, e.g. non-Braille photos).
    """

    def __init__(self, images: Sequence[str], size: int = 384, crops_per_image: int = 8,
                 target_cell_h: Tuple[float, float] = (16, 70), seed: int = 0):
        self.images = list(images)
        self.size = size
        self.cpi = crops_per_image
        self.th = target_cell_h
        self.seed = seed
        self.epoch = 0
        self._cache = {}

    def __len__(self) -> int:
        return len(self.images) * self.cpi

    def _get(self, idx: int):
        if idx not in self._cache:
            img, cells = load_real(self.images[idx])
            # keep memory bounded: store downscaled to long side 1600
            s = min(1.0, 1600 / max(img.shape[:2]))
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                cells = cells.copy()
                cells[:, :4] *= s
            self._cache[idx] = (img, cells)
        return self._cache[idx]

    def __getitem__(self, i: int):
        rng = random.Random((self.seed * 7919 + self.epoch) * 1_000_003 + i)
        img, cells = self._get(i // self.cpi)
        S = self.size
        med_h = float(np.median(cells[:, 3] - cells[:, 1])) if len(cells) else 40.0
        scale = math.exp(rng.uniform(math.log(self.th[0]), math.log(self.th[1]))) / max(med_h, 1)
        ang = rng.uniform(-6, 6)
        h, w = img.shape[:2]
        # pick crop centre: mostly on a labelled cell
        if len(cells) and rng.random() < 0.85:
            c = cells[rng.randrange(len(cells))]
            cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        M = cv2.getRotationMatrix2D((cx, cy), ang, scale)
        M[0, 2] += S / 2 - cx + rng.uniform(-S / 4, S / 4)
        M[1, 2] += S / 2 - cy + rng.uniform(-S / 4, S / 4)
        out = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if len(cells):
            b = cells
            corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1).reshape(-1, 2)
            tc = (np.c_[corners, np.ones(len(corners))] @ M.T).reshape(-1, 4, 2)
            x0, y0, x1, y1 = tc[..., 0].min(1), tc[..., 1].min(1), tc[..., 0].max(1), tc[..., 1].max(1)
            # shrink rotated-box growth back to the original aspect
            ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
            bw, bh = (b[:, 2] - b[:, 0]) * scale, (b[:, 3] - b[:, 1]) * scale
            ncells = np.stack([ccx - bw / 2, ccy - bh / 2, ccx + bw / 2, ccy + bh / 2, b[:, 4]], 1)
            keep = (ccx >= 0) & (ccx < S) & (ccy >= 0) & (ccy < S)
            ncells = ncells[keep].astype(np.float32)
        else:
            ncells = np.zeros((0, 5), np.float32)
        out = photometric(out, rng)
        return to_tensor_sample(out, ncells)


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * rng.uniform(0.6, 1.3) + rng.uniform(-25, 25)
    x *= np.float32([rng.uniform(0.9, 1.1) for _ in range(3)])
    if rng.random() < 0.3:
        k = rng.uniform(0.5, 0.9)
        x = x * k + x.mean() * (1 - k)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.5))
    x += np.random.default_rng(rng.getrandbits(32)).standard_normal(x.shape).astype(np.float32) * rng.uniform(0, 6)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        _, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.1:
        x = cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return x
