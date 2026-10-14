"""Procedural generator of photo-like Braille page images with cell labels.

Each sample simulates a smartphone photo of an embossed Braille page:

* physical page geometry (dot spacing, cell pitch, line pitch, margins,
  paragraph indents, centred titles, blank lines)
* embossed dots rendered as a height map and shaded with a random light
  source (highlight + cast shadow), optional *interpoint* (back side) dots that
  appear as pits and must NOT be detected, optional ink-printed Braille
* interlinear printed Latin text, paper texture, backgrounds, page edges
* perspective / rotation, illumination gradients, blur, noise, JPEG artefacts

Labels: ``cells`` float32 array (N, 5) = x0, y0, x1, y1, mask (1..63) in
output-image pixels. Blank cells (spaces) are not objects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .braille import BrailleCodec
from .textgen import random_paragraph

_codec = BrailleCodec()


@dataclass
class SynthConfig:
    out_h: int = 384
    out_w: int = 384
    dot_px_min: float = 4.5      # dot spacing (centre-to-centre) in output px
    dot_px_max: float = 17.0
    p_interpoint: float = 0.5
    p_ink_text: float = 0.3
    p_ink_braille: float = 0.07
    p_page_edge: float = 0.5
    p_random_cells: float = 0.25
    p_negative: float = 0.03
    max_rotation: float = 8.0
    max_persp: float = 0.07


def _loguniform(rng: random.Random, a: float, b: float) -> float:
    return math.exp(rng.uniform(math.log(a), math.log(b)))


def _lowfreq_noise(h: int, w: int, rng: np.random.Generator, scale: int) -> np.ndarray:
    sh, sw = max(2, h // scale), max(2, w // scale)
    n = rng.standard_normal((sh, sw)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


def _background(h: int, w: int, rng: random.Random, nrng: np.random.Generator) -> np.ndarray:
    base = np.array([rng.uniform(0, 255) for _ in range(3)], np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    for _ in range(rng.randint(0, 8)):
        col = tuple(float(rng.uniform(0, 255)) for _ in range(3))
        if rng.random() < 0.5:
            p1 = (rng.randint(-w // 2, w), rng.randint(-h // 2, h))
            p2 = (p1[0] + rng.randint(10, w), p1[1] + rng.randint(10, h))
            cv2.rectangle(img, p1, p2, col, -1)
        else:
            c = (rng.randint(0, w), rng.randint(0, h))
            cv2.ellipse(img, c, (rng.randint(5, w // 2), rng.randint(5, h // 2)),
                        rng.uniform(0, 180), 0, 360, col, -1)
    img += _lowfreq_noise(h, w, nrng, rng.randint(8, 64))[..., None] * rng.uniform(0, 25)
    t = rng.random()
    if t < 0.3:  # wood grain
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, math.pi)
        u = xx * math.cos(a) + yy * math.sin(a)
        u += _lowfreq_noise(h, w, nrng, rng.randint(16, 64)) * rng.uniform(5, 30)
        grain = np.sin(u / rng.uniform(1.5, 8)) + 0.5 * np.sin(u / rng.uniform(0.7, 2.0))
        img += grain[..., None] * rng.uniform(5, 30) * np.float32([1, 0.8, 0.6])
    elif t < 0.65:  # fabric / carpet / speckle: high-frequency dot-like texture
        s = rng.uniform(1, 4)
        n = nrng.standard_normal((max(2, int(h / s)), max(2, int(w / s)), 3)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
        img += n * rng.uniform(10, 50)
    if rng.random() < 0.4:  # scattered blobs that look like dots
        for _ in range(rng.randint(50, 800)):
            c = (rng.randint(0, w), rng.randint(0, h))
            col = tuple(float(v) for v in (base + rng.uniform(-80, 80)))
            cv2.circle(img, c, rng.randint(1, 5), col, -1, cv2.LINE_AA)
    return np.clip(img, 0, 255)


def _layout_cells(rng: random.Random, n_cols: int, n_lines: int) -> List[List[Optional[int]]]:
    """Return list of lines; each line is a list of cells (None = blank)."""
    lines: List[List[Optional[int]]] = []
    while len(lines) < n_lines:
        r = rng.random()
        if r < rng.choice([0.0, 0.25]) * 0.5 or (r < 0.15 and rng.random() < 0.3):
            lines.append([])  # blank line
            continue
        if r < 0.22:  # centred title
            words = _codec.encode(random_paragraph(rng))[: rng.randint(4, max(5, n_cols - 4))]
            pad = max(0, (n_cols - len(words)) // 2)
            lines.append([None] * pad + words)
            continue
        if rng.random() < 0.1:  # random cells: covers rarely used patterns
            line: List[Optional[int]] = []
            while len(line) < n_cols:
                line.append(rng.randint(1, 63) if rng.random() > 0.2 else None)
            lines.append(line)
            continue
        text = random_paragraph(rng)
        cells = _codec.encode(text)
        # split into words and wrap
        words: List[List[int]] = []
        cur: List[int] = []
        for c in cells:
            if c is None:
                words.append(cur)
                cur = []
            else:
                cur.append(c)
        if cur:
            words.append(cur)
        indent = rng.choice([0, 2, 2, 2, 4])
        line = [None] * indent
        for w in words:
            w = w[:n_cols]
            if len(line) + len(w) + (1 if line and line[-1] is not None else 0) > n_cols:
                lines.append(line)
                if len(lines) >= n_lines:
                    break
                line = []
            if line and line[-1] is not None:
                line.append(None)
            line += w
        if line and len(lines) < n_lines:
            lines.append(line)
    return lines[:n_lines]


def _random_cell_lines(rng: random.Random, n_cols: int, n_lines: int, p_blank: float = 0.2):
    return [[(rng.randint(1, 63) if rng.random() > p_blank else None) for _ in range(n_cols)]
            for _ in range(n_lines)]


def render_sample(rng: random.Random, cfg: SynthConfig = SynthConfig()) -> Tuple[np.ndarray, np.ndarray]:
    nrng = np.random.default_rng(rng.getrandbits(32))
    H, W = cfg.out_h, cfg.out_w

    # ---------------------------------------------------------- geometry (px)
    d = _loguniform(rng, cfg.dot_px_min, cfg.dot_px_max)          # dot spacing
    cell_pitch = d * rng.uniform(2.35, 2.75)                       # ~6.0-6.6 mm / 2.5 mm
    line_pitch = d * rng.uniform(3.9, 4.5)
    ink_text = rng.random() < cfg.p_ink_text
    if ink_text:
        line_pitch = d * rng.uniform(5.5, 7.5)                     # room for interlinear text
    r_dot = d * rng.uniform(0.24, 0.36)

    margin = int(max(H, W) * 0.25)
    PH, PW = H + 2 * margin, W + 2 * margin                        # page-plane canvas

    page_x0, page_y0, page_x1, page_y1 = 0, 0, PW, PH
    if rng.random() < cfg.p_page_edge:
        page_x0 = rng.randint(0, margin + W // 2)
        page_y0 = rng.randint(0, margin + H // 3)
        page_x1 = PW - rng.randint(0, margin + W // 3)
        page_y1 = PH - rng.randint(0, margin + H // 3)
        if page_x1 - page_x0 < W // 3:
            page_x1 = min(PW, page_x0 + W // 2)
        if page_y1 - page_y0 < H // 3:
            page_y1 = min(PH, page_y0 + H // 2)

    text_x0 = page_x0 + rng.uniform(0.5, 4) * cell_pitch
    text_y0 = page_y0 + rng.uniform(0.3, 3) * line_pitch - rng.uniform(0, line_pitch)
    n_cols = max(1, int((page_x1 - text_x0 - d) / cell_pitch) - rng.randint(0, 3))
    n_lines = max(1, int((page_y1 - text_y0 - 2 * d) / line_pitch))

    negative = rng.random() < cfg.p_negative
    if negative:
        lines = []
    elif rng.random() < cfg.p_random_cells:
        lines = _random_cell_lines(rng, n_cols, n_lines, rng.uniform(0.0, 0.4))
    else:
        lines = _layout_cells(rng, n_cols, n_lines)

    jitter = d * rng.uniform(0.0, 0.07)
    line_tilt = rng.uniform(-0.004, 0.004)  # slight per-page drift of lines
    ink_braille = rng.random() < cfg.p_ink_braille

    height = np.zeros((PH, PW), np.float32)
    ink = np.zeros((PH, PW), np.float32)
    boxes: List[Tuple[float, float, float, float, int]] = []
    rint = lambda v: int(round(v * 16))
    shift = 4
    amp_base = rng.uniform(0.7, 1.0)

    for li, line in enumerate(lines):
        y = text_y0 + li * line_pitch
        for ci, m in enumerate(line):
            if not m:
                continue
            x = text_x0 + ci * cell_pitch
            yy = y + x * line_tilt
            if x + d > page_x1 or yy + 2 * d > page_y1:
                continue
            for k in range(6):
                if not (m >> k) & 1:
                    continue
                dx = d * (k // 3)
                dy = d * (k % 3)
                px = x + dx + nrng.normal(0, jitter)
                py = yy + dy + nrng.normal(0, jitter)
                rr = r_dot * rng.uniform(0.85, 1.15)
                if ink_braille:
                    cv2.circle(ink, (rint(px), rint(py)), rint(rr), 1.0, -1, cv2.LINE_AA, shift)
                else:
                    cv2.circle(height, (rint(px), rint(py)), rint(rr),
                               amp_base * rng.uniform(0.55, 1.15), -1, cv2.LINE_AA, shift)
            boxes.append((x - r_dot, yy - r_dot, x + d + r_dot, yy + 2 * d + r_dot, m))

    # interpoint: back-side dots appear as pits, offset from the front grid
    if rng.random() < cfg.p_interpoint and not ink_braille:
        back = np.zeros_like(height)
        ox = rng.uniform(0.2, 0.8) * cell_pitch
        oy = rng.uniform(0.25, 0.75) * line_pitch
        for li, line in enumerate(_random_cell_lines(rng, n_cols + 1, n_lines + 1, 0.3)):
            y = text_y0 + li * line_pitch - oy
            for ci, m in enumerate(line):
                if not m:
                    continue
                x = text_x0 + ci * cell_pitch - ox
                for k in range(6):
                    if (m >> k) & 1:
                        cv2.circle(back, (rint(x + d * (k // 3)), rint(y + d * (k % 3))),
                                   rint(r_dot * rng.uniform(0.85, 1.1)),
                                   amp_base * rng.uniform(0.4, 1.0), -1, cv2.LINE_AA, shift)
        height -= back * rng.uniform(0.5, 1.0)

    # ------------------------------------------------------------ shading
    sig = max(0.6, r_dot * rng.uniform(0.35, 0.6))
    height = cv2.GaussianBlur(height, (0, 0), sig)
    height += _lowfreq_noise(PH, PW, nrng, max(2, int(d))) * rng.uniform(0.0, 0.03)
    relief = rng.uniform(0.5, 1.6) / max(1.0, sig)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * relief * 4
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * relief * 4
    az = rng.uniform(0, 2 * math.pi)
    if rng.random() < 0.6:  # light usually from above / top-left in handheld photos
        az = rng.uniform(math.pi * 1.0, math.pi * 2.0)
    el = math.radians(rng.uniform(20, 70))  # angle from surface normal
    lx, ly, lz = math.sin(el) * math.cos(az), math.sin(el) * math.sin(az), math.cos(el)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    ndotl = (-gx * lx - gy * ly + lz) * nz
    diffuse = np.clip(ndotl, 0, 1) / max(lz, 1e-3)
    amb = rng.uniform(0.15, 0.5)
    shade = amb + (1 - amb) * diffuse
    spec = np.clip(ndotl, 0, 1) ** rng.uniform(8, 40) * rng.uniform(0, 0.25)
    # cast shadow: shifted height opposite to light direction
    sh_len = sig * rng.uniform(0.5, 2.0)
    M = np.float32([[1, 0, lx / max(math.sin(el), 0.2) * sh_len], [0, 1, ly / max(math.sin(el), 0.2) * sh_len]])
    shadow = cv2.warpAffine(np.clip(height, 0, None), M, (PW, PH)) - np.clip(height, 0, None)
    shade *= 1 - np.clip(shadow, 0, 1) * rng.uniform(0.0, 0.5)

    paper = np.array([rng.uniform(200, 255), rng.uniform(195, 250), rng.uniform(185, 245)], np.float32)
    if rng.random() < 0.15:
        paper *= rng.uniform(0.6, 0.9)
    albedo = np.ones((PH, PW), np.float32)
    albedo *= 1 + _lowfreq_noise(PH, PW, nrng, rng.randint(16, 128)) * rng.uniform(0, 0.05)
    albedo *= 1 + nrng.standard_normal((PH, PW)).astype(np.float32) * rng.uniform(0, 0.02)

    if ink_braille:
        albedo *= 1 - cv2.GaussianBlur(ink, (0, 0), 0.5) * rng.uniform(0.5, 0.95)

    # interlinear printed text (ink)
    if ink_text and lines:
        tmp = np.zeros((PH, PW), np.uint8)
        font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                           cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX])
        fscale = d * rng.uniform(0.8, 1.4) / 22.0
        for li in range(len(lines)):
            y = int(text_y0 + li * line_pitch + 2 * d + (line_pitch - 2 * d) * rng.uniform(0.6, 0.85))
            txt = random_paragraph(rng)[: n_cols * 2]
            txt = "  ".join(txt.upper()) if rng.random() < 0.4 else txt
            cv2.putText(tmp, txt, (int(text_x0), y), font, fscale, 255, max(1, int(fscale * 1.5)), cv2.LINE_AA)
        albedo *= 1 - tmp.astype(np.float32) / 255 * rng.uniform(0.3, 0.75)

    if rng.random() < 0.3:  # dirt, punched holes, pencil marks (not Braille)
        specks = np.zeros((PH, PW), np.float32)
        for _ in range(rng.randint(3, 40)):
            c = (rng.randint(page_x0, max(page_x0 + 1, page_x1)), rng.randint(page_y0, max(page_y0 + 1, page_y1)))
            if rng.random() < 0.5:
                cv2.circle(specks, c, max(1, int(r_dot * rng.uniform(0.5, 2.0))), rng.uniform(0.3, 1), -1, cv2.LINE_AA)
            else:
                e = (c[0] + rng.randint(-int(4 * d), int(4 * d)), c[1] + rng.randint(-int(4 * d), int(4 * d)))
                cv2.line(specks, c, e, rng.uniform(0.2, 0.8), rng.randint(1, 2), cv2.LINE_AA)
        albedo *= 1 - cv2.GaussianBlur(specks, (0, 0), 0.7) * 0.8

    page = (albedo * shade + spec)[..., None] * paper[None, None, :]

    # page alpha and soft drop shadow
    alpha = np.zeros((PH, PW), np.float32)
    alpha[page_y0:page_y1, page_x0:page_x1] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)

    # ------------------------------------------------------------ projection
    ang = math.radians(rng.uniform(-cfg.max_rotation, cfg.max_rotation))
    src = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
    c, s = math.cos(ang), math.sin(ang)
    ctr = np.float32([PW / 2, PH / 2])
    dst = []
    for p in src:
        q = p - ctr
        q = np.float32([c * q[0] - s * q[1], s * q[0] + c * q[1]])
        q += np.float32([nrng.uniform(-1, 1), nrng.uniform(-1, 1)]) * cfg.max_persp * max(PW, PH)
        dst.append(q + np.float32([W / 2, H / 2]) + np.float32([nrng.uniform(-0.1, 0.1) * W, nrng.uniform(-0.1, 0.1) * H]))
    Hm = cv2.getPerspectiveTransform(src, np.float32(dst))

    bg = _background(H, W, rng, nrng)
    warped = cv2.warpPerspective(page, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    walpha = cv2.warpPerspective(alpha, Hm, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    if rng.random() < 0.5:  # page drop shadow on background
        sh = cv2.GaussianBlur(walpha[..., 0], (0, 0), rng.uniform(2, 8))[..., None]
        bg *= 1 - sh * rng.uniform(0.1, 0.5)
    img = warped * walpha + bg * (1 - walpha)

    # ------------------------------------------------------------ photometric
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gdir = rng.uniform(0, 2 * math.pi)
    grad = ((xx - W / 2) * math.cos(gdir) + (yy - H / 2) * math.sin(gdir)) / max(H, W)
    img *= (1 + grad * rng.uniform(-0.8, 0.8))[..., None]
    if rng.random() < 0.5:
        rad = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (0.5 * (H * H + W * W))
        img *= (1 - rad * rng.uniform(0, 0.5))[..., None]
    img *= np.float32([rng.uniform(0.93, 1.07) for _ in range(3)])
    img *= rng.uniform(0.45, 1.25)
    if rng.random() < 0.3:  # contrast reduction (haze / overexposure)
        k = rng.uniform(0.4, 0.9)
        img = img * k + img.mean() * (1 - k)
    img = np.clip(img, 0, 255)
    gamma = rng.uniform(0.7, 1.4)
    img = 255.0 * (img / 255.0) ** gamma

    br = rng.random()
    if br < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, max(0.4, d * 0.18)))
    elif br < 0.45:
        k = max(3, int(d * rng.uniform(0.2, 0.6)) | 1)
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kern = cv2.warpAffine(kern, rot, (k, k))
        kern /= max(kern.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kern)
    img += nrng.standard_normal(img.shape).astype(np.float32) * rng.uniform(0, 8)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 95)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.15:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------ labels
    out = []
    if boxes:
        b = np.float32(boxes)
        corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1)  # N,4,2
        tc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Hm).reshape(-1, 4, 2)
        x0, y0 = tc[..., 0].min(1), tc[..., 1].min(1)
        x1, y1 = tc[..., 0].max(1), tc[..., 1].max(1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        keep = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        out = np.stack([x0, y0, x1, y1, b[:, 4]], 1)[keep]
    cells = np.float32(out).reshape(-1, 5)
    return img, cells  # img is BGR uint8
