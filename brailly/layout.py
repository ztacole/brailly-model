"""Reading-order reconstruction: cells -> lines (with spaces) -> paragraphs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .braille import BrailleCodec, cells_to_unicode


@dataclass
class Line:
    items: list                    # detected cells, left-to-right
    cells: List[Optional[int]]     # masks with None for blank cells
    left: float = 0.0              # rotated x of first cell
    right: float = 0.0
    y: float = 0.0                 # rotated mean y
    braille: str = ""
    text: str = ""


@dataclass
class Paragraph:
    lines: List[Line]
    text: str = ""
    kind: str = "paragraph"        # "paragraph" | "heading"


@dataclass
class Layout:
    angle_deg: float
    pitch: float
    line_spacing: float
    lines: List[Line] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def braille(self) -> str:
        return "\n".join(l.braille for l in self.lines)


def estimate_angle(pts: np.ndarray, med_w: float) -> float:
    """Dominant text-line angle (radians) from nearest horizontal neighbours."""
    if len(pts) < 3:
        return 0.0
    angs, wts = [], []
    for i in range(len(pts)):
        d = pts - pts[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        dist[i] = np.inf
        cand = np.where((dist < 3.0 * med_w) & (np.abs(d[:, 0]) > np.abs(d[:, 1]) * 1.5) & (d[:, 0] > 0))[0]
        if len(cand):
            j = cand[np.argmin(dist[cand])]
            angs.append(math.atan2(d[j, 1], d[j, 0]))
            wts.append(1.0)
    if not angs:
        return 0.0
    return float(np.median(angs))


def build_layout(items: Sequence, codec: Optional[BrailleCodec] = None,
                 dehyphenate: bool = False) -> Layout:
    """``items``: objects with cx, cy, w, h, mask attributes."""
    codec = codec or BrailleCodec()
    items = [c for c in items if c.mask]
    if not items:
        return Layout(0.0, 0.0, 0.0)
    pts = np.array([[c.cx, c.cy] for c in items], np.float64)
    med_w = float(np.median([c.w for c in items]))
    med_h = float(np.median([c.h for c in items]))
    ang = estimate_angle(pts, med_w)
    ca, sa = math.cos(-ang), math.sin(-ang)
    rx = pts[:, 0] * ca - pts[:, 1] * sa
    ry = pts[:, 0] * sa + pts[:, 1] * ca

    # --- greedy left-to-right line tracking (tolerant to curvature/perspective)
    order = np.argsort(rx)
    tracks: List[List[int]] = []
    for i in order:
        best, best_d = -1, 0.5 * med_h
        for t, tr in enumerate(tracks):
            j = tr[-1]
            gap = rx[i] - rx[j]
            if gap > 25 * med_w:  # too far to be the same line continuation
                continue
            # predict y from the last two members (local slope)
            yp = ry[j]
            dyv = abs(ry[i] - yp)
            if dyv < best_d:
                best, best_d = t, dyv
        if best >= 0:
            tracks[best].append(i)
        else:
            tracks.append([i])

    # merge tracks that are the same text line split by a long gap
    tracks.sort(key=lambda tr: np.mean(ry[tr]))
    merged: List[List[int]] = []
    for tr in tracks:
        if merged:
            prev = merged[-1]
            if abs(np.median(ry[tr]) - np.median(ry[prev])) < 0.35 * med_h and \
                    (rx[tr].min() > rx[prev].max() or rx[prev].min() > rx[tr].max()):
                merged[-1] = sorted(prev + tr, key=lambda k: rx[k])
                continue
        merged.append(tr)
    tracks = [sorted(tr, key=lambda k: rx[k]) for tr in merged]

    # --- cell pitch from consecutive gaps
    gaps = np.concatenate([np.diff(rx[tr]) for tr in tracks if len(tr) > 1] or [np.array([])])
    if len(gaps):
        base = np.median(gaps)
        small = gaps[gaps < 1.4 * base]
        pitch = float(np.median(small)) if len(small) else float(base)
    else:
        pitch = med_w * 1.6
    pitch = max(pitch, med_w * 1.05)

    lines: List[Line] = []
    for tr in tracks:
        cells: List[Optional[int]] = []
        for a, k in enumerate(tr):
            if a:
                n_blank = int(round((rx[k] - rx[tr[a - 1]]) / pitch)) - 1
                cells += [None] * max(0, n_blank)
            cells.append(items[k].mask)
        ln = Line(items=[items[k] for k in tr], cells=cells, left=float(rx[tr[0]]),
                  right=float(rx[tr[-1]]), y=float(np.mean(ry[tr])))
        ln.braille = cells_to_unicode(cells)
        ln.text = codec.decode(cells)
        lines.append(ln)
    lines.sort(key=lambda l: l.y)

    ys = np.array([l.y for l in lines])
    spacing = float(np.median(np.diff(ys))) if len(ys) > 1 else med_h * 1.6
    lay = Layout(angle_deg=math.degrees(ang), pitch=pitch, line_spacing=spacing, lines=lines)
    lay.paragraphs = group_paragraphs(lines, pitch, spacing, dehyphenate)
    return lay


def group_paragraphs(lines: List[Line], pitch: float, spacing: float, dehyphenate: bool = False) -> List[Paragraph]:
    if not lines:
        return []
    full = [l for l in lines if len(l.cells) >= 4] or lines
    margin_l = float(np.percentile([l.left for l in full], 10))
    margin_r = float(np.percentile([l.right for l in full], 90))
    width = max(margin_r - margin_l, pitch)

    paras: List[Paragraph] = []
    cur: List[Line] = []
    prev: Optional[Line] = None
    for ln in lines:
        indent = (ln.left - margin_l) / pitch
        rgap = (margin_r - ln.right) / pitch
        heading = indent >= 3 and rgap >= 3 and abs(indent - rgap) <= 3 and width / pitch > 12
        new_para = prev is None
        if prev is not None:
            if ln.y - prev.y > 1.6 * spacing:
                new_para = True
            elif indent >= 1.5:
                new_para = True
            elif (margin_r - prev.right) / pitch > 6 and prev.text.rstrip()[-1:] in ".?!:\u201d)":
                new_para = True
            elif paras and paras[-1].kind == "heading" and not cur:
                new_para = True
        if heading:
            if cur:
                paras.append(Paragraph(cur))
            paras.append(Paragraph([ln], kind="heading"))
            cur = []
        else:
            if new_para and cur:
                paras.append(Paragraph(cur))
                cur = []
            cur.append(ln)
        prev = ln
    if cur:
        paras.append(Paragraph(cur))

    for p in paras:
        txt = ""
        for ln in p.lines:
            t = ln.text.strip()
            if not t:
                continue
            if not txt:
                txt = t
            elif txt.endswith("-"):
                txt = (txt[:-1] if dehyphenate else txt) + t
            else:
                txt += " " + t
        p.text = txt
    return [p for p in paras if p.text]
