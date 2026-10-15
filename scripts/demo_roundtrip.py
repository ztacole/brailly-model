#!/usr/bin/env python
"""Uji ujung-ke-ujung: teks Indonesia -> halaman Braille -> foto -> baca -> teks.

Merender halaman Braille berisi teks yang sudah diketahui (dengan pencahayaan,
derau, dan kemiringan seperti foto ponsel), membacanya kembali dengan model,
lalu melaporkan ketepatan per karakter. Berguna untuk memastikan seluruh rantai
(deteksi -> tata letak -> paragraf -> terjemahan) bekerja pada bahasa Indonesia.

python scripts/demo_roundtrip.py --model models/brailly.onnx
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.braille import BrailleCodec  # noqa: E402
from brailly.metrics import levenshtein  # noqa: E402
from brailly.pipeline import BraillyReader, ReaderConfig  # noqa: E402

TEKS = [
    "Kemerdekaan Indonesia",
    "",
    "  Pada tanggal 17 Agustus 1945, bangsa",
    "Indonesia memproklamasikan kemerdekaan.",
    "Peristiwa itu menjadi tonggak sejarah",
    "yang sangat penting bagi rakyat.",
    "",
    "  Sejak saat itu, rakyat Indonesia terus",
    "belajar dan bekerja untuk membangun",
    "negeri yang adil dan makmur.",
    "",
    "Apakah kita sudah menghargai jasa para",
    "pahlawan? Mari kita renungkan bersama.",
]


def render_page(lines, codec, d=11.0, rng=None, rotate=1.5):
    """Render halaman Braille timbul dengan tata letak tetap (bukan acak)."""
    rng = rng or random.Random(0)
    pitch = d * 2.5
    line_pitch = d * 4.2
    cell_rows = [codec.encode(t.strip()) if t.strip() else [] for t in lines]
    indents = [(len(t) - len(t.lstrip())) for t in lines]
    n_cols = max((len(c) + i) for c, i in zip(cell_rows, indents))
    margin = d * 4
    W = int(margin * 2 + n_cols * pitch + d)
    H = int(margin * 2 + len(lines) * line_pitch)

    height = np.zeros((H, W), np.float32)
    rint, shift = lambda v: int(round(v * 16)), 4
    for li, (row, ind) in enumerate(zip(cell_rows, indents)):
        y = margin + li * line_pitch
        for ci, m in enumerate(row):
            if not m:
                continue
            x = margin + (ci + ind) * pitch
            for k in range(6):
                if (m >> k) & 1:
                    cv2.circle(height, (rint(x + d * (k // 3)), rint(y + d * (k % 3))),
                               rint(d * 0.3), 1.0, -1, cv2.LINE_AA, shift)

    # bayangan timbul: arah cahaya dari kiri atas
    height = cv2.GaussianBlur(height, (0, 0), d * 0.16)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * 9
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * 9
    el, az = math.radians(45), math.radians(250)
    lx, ly, lz = math.sin(el) * math.cos(az), math.sin(el) * math.sin(az), math.cos(el)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nl = (-gx * lx - gy * ly + lz) * nz
    shade = 0.35 + 0.65 * np.clip(nl, 0, 1) / lz
    img = np.clip(shade[..., None] * np.float32([238, 240, 243]), 0, 255)

    # derau, kemiringan, kompresi -> mendekati foto nyata
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rotate, 1.0)
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img *= (1 - ((xx - W * 0.3) ** 2 + (yy - H * 0.2) ** 2) / (6.0 * (H * H + W * W)))[..., None]
    img += np.random.default_rng(1).standard_normal((H, W, 3)).astype(np.float32) * 3
    img = np.clip(img, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/brailly.onnx")
    ap.add_argument("--save-image", default=None)
    ap.add_argument("--dot-px", type=float, default=11.0)
    args = ap.parse_args()

    codec = BrailleCodec()
    img = render_page(TEKS, codec, d=args.dot_px)
    if args.save_image:
        cv2.imwrite(args.save_image, img)
        print(f"halaman uji disimpan: {args.save_image} {img.shape[1]}x{img.shape[0]}")

    hasil = BraillyReader(args.model, ReaderConfig()).read(img)

    harapan = [t.strip() for t in TEKS if t.strip()]
    didapat = [l["text"].strip() for l in hasil["lines"]]
    print(f"\n{'=' * 62}\nPER BARIS\n{'=' * 62}")
    cocok = 0
    for i, h in enumerate(harapan):
        g = didapat[i] if i < len(didapat) else ""
        ok = g == h
        cocok += ok
        print(f"{'OK ' if ok else 'BEDA'} | {h}")
        if not ok:
            print(f"     | {g}")

    gab_h, gab_g = " ".join(harapan), " ".join(didapat)
    cer = levenshtein(gab_g, gab_h) / max(1, len(gab_h))
    print(f"\n{'=' * 62}\nPARAGRAF HASIL BACA\n{'=' * 62}")
    for p in hasil["paragraphs"]:
        print(f"[{p['kind']}] {p['text']}")
    print(f"\nbaris tepat {cocok}/{len(harapan)}   CER {cer:.4f}   "
          f"sel {hasil['meta']['num_cells']}   sudut {hasil['meta']['angle_deg']}derajat")
    return 0 if cer < 0.02 else 1


if __name__ == "__main__":
    sys.exit(main())
