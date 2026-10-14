"""Decode raw network output into a list of detected Braille cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .model import STRIDE


@dataclass
class Cell:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    dot_probs: np.ndarray  # (6,)

    @property
    def mask(self) -> int:
        return int(sum(1 << k for k in range(6) if self.dot_probs[k] > 0.5))

    @property
    def box(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _maxpool3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="constant", constant_values=-np.inf)
    h, w = a.shape
    m = a.copy()
    for dy in range(3):
        for dx in range(3):
            m = np.maximum(m, p[dy:dy + h, dx:dx + w])
    return m


def decode_output(out: np.ndarray, score_thr: float = 0.3, scale: float = 1.0) -> List[Cell]:
    """out: (11, H/4, W/4) raw logits. ``scale`` maps model-input px -> original px."""
    heat = _sigmoid(out[0])
    peaks = (heat == _maxpool3(heat)) & (heat >= score_thr)
    ys, xs = np.nonzero(peaks)
    cells: List[Cell] = []
    for y, x in zip(ys, xs):
        dp = _sigmoid(out[1:7, y, x])
        if dp.max() <= 0.5:  # a cell must have at least one raised dot
            continue
        ox, oy = out[7, y, x], out[8, y, x]
        w, h = np.exp(np.clip(out[9:11, y, x], -3, 6))
        cells.append(Cell(cx=(x + ox) * STRIDE * scale, cy=(y + oy) * STRIDE * scale,
                          w=float(w) * STRIDE * scale, h=float(h) * STRIDE * scale,
                          score=float(heat[y, x]), dot_probs=dp))
    return _suppress_overlaps(cells)


def _suppress_overlaps(cells: List[Cell]) -> List[Cell]:
    """Remove duplicate detections whose centres are closer than ~half a cell."""
    if len(cells) < 2:
        return cells
    cells = sorted(cells, key=lambda c: -c.score)
    pts = np.array([[c.cx, c.cy] for c in cells])
    wh = np.array([[c.w, c.h] for c in cells])
    keep = np.ones(len(cells), bool)
    for i in range(len(cells)):
        if not keep[i]:
            continue
        d = np.abs(pts[i + 1:] - pts[i])
        # Neighbouring cells are >= ~1.5 cell-widths apart (pitch 6+ mm vs 4 mm
        # cell), so anything closer is a competing hypothesis for the same cell
        # (typically a "half cell" shifted by one dot column).
        dup = (d[:, 0] < 0.75 * wh[i, 0]) & (d[:, 1] < 0.5 * wh[i, 1])
        keep[i + 1:][dup] = False
    return [c for c, k in zip(cells, keep) if k]
