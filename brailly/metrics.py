from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class GTCell:
    cx: float
    cy: float
    w: float
    h: float
    mask: int


def gt_from_array(cells: np.ndarray) -> List[GTCell]:
    return [GTCell((a[0] + a[2]) / 2, (a[1] + a[3]) / 2, a[2] - a[0], a[3] - a[1], int(a[4])) for a in cells]


def match_cells(pred: Sequence, gt: Sequence[GTCell]):
    """Greedy centre matching. Returns (n_matched, n_correct_pattern)."""
    if not pred or not gt:
        return 0, 0
    P = np.array([[p.cx, p.cy] for p in pred])
    G = np.array([[g.cx, g.cy] for g in gt])
    tol = np.array([[0.5 * g.w, 0.4 * g.h] for g in gt])
    used = np.zeros(len(pred), bool)
    matched = correct = 0
    order = np.argsort([-p.score for p in pred]) if hasattr(pred[0], "score") else range(len(pred))
    gt_taken = np.zeros(len(gt), bool)
    for i in order:
        d = np.abs(G - P[i]) / tol
        ok = (d[:, 0] < 1) & (d[:, 1] < 1) & ~gt_taken
        if not ok.any():
            continue
        dist = np.where(ok, np.hypot(d[:, 0], d[:, 1]), np.inf)
        j = int(np.argmin(dist))
        gt_taken[j] = True
        used[i] = True
        matched += 1
        correct += int(pred[i].mask == gt[j].mask)
    return matched, correct


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
