#!/usr/bin/env python
"""Bandingkan varian model (ONNX/TFLite, fp32/int8): akurasi, ukuran, kecepatan.

Semua varian diukur pada daftar foto yang sama dan lewat jalur pasca-pemrosesan
yang sama, sehingga perbedaannya murni berasal dari model.

python scripts/bench_backends.py --list data/uploaded/test.txt \
    --model onnx-fp32=models/brailly.onnx --model tflite-int8=models/brailly_int8.tflite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.layout import build_layout  # noqa: E402
from brailly.metrics import gt_from_array, levenshtein, match_cells  # noqa: E402
from brailly.pipeline import BraillyReader, ReaderConfig  # noqa: E402


def baca_daftar(list_file: str):
    base = os.path.dirname(os.path.abspath(list_file))
    with open(list_file, encoding="utf-8") as f:
        return [os.path.join(base, l.strip().replace("\\", "/")) for l in f if l.strip()]


def baca_anotasi(path: str, w: int, h: int) -> np.ndarray:
    csv = os.path.splitext(path)[0] + ".csv"
    rows = []
    if os.path.exists(csv):
        with open(csv, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split(";")
                if len(p) >= 5 and 1 <= int(p[4]) <= 63:
                    rows.append((float(p[0]) * w, float(p[1]) * h,
                                 float(p[2]) * w, float(p[3]) * h, int(p[4])))
    return np.float32(rows).reshape(-1, 5)


def f1(p, r):
    return 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="nama=path, dapat diulang (mis. tflite-int8=models/m.tflite)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = baca_daftar(args.list)
    varian = [m.split("=", 1) for m in args.model]
    hasil = {}
    for nama, mp in varian:
        reader = BraillyReader(mp, ReaderConfig())
        n_pred = n_gt = matched = correct = b_err = b_len = 0
        t0 = time.time()
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            gt = gt_from_array(baca_anotasi(p, img.shape[1], img.shape[0]))
            pred = reader.detect_cells(img)
            m, c = match_cells(pred, gt)
            n_pred += len(pred); n_gt += len(gt); matched += m; correct += c
            lp, lg = build_layout(pred, reader.codec), build_layout(gt, reader.codec)
            b_err += levenshtein(lp.braille, lg.braille); b_len += len(lg.braille)
        detik = (time.time() - t0) / max(1, len(paths))
        sp, sr = correct / max(1, n_pred), correct / max(1, n_gt)
        dp, dr = matched / max(1, n_pred), matched / max(1, n_gt)
        hasil[nama] = {
            "ukuran_mb": round(os.path.getsize(mp) / 1e6, 1),
            "det_f1": round(f1(dp, dr), 4),
            "sym_f1": round(f1(sp, sr), 4),
            "braille_cer": round(b_err / max(1, b_len), 4),
            "detik_per_foto": round(detik, 3),
        }
        print(f"{nama:14s} {hasil[nama]}", flush=True)

    lebar = max(len(n) for n in hasil) + 1
    print(f"\n{'varian':<{lebar}} {'MB':>5} {'det F1':>7} {'sym F1':>7} {'CER':>7} {'detik':>7}")
    for n, r in hasil.items():
        print(f"{n:<{lebar}} {r['ukuran_mb']:>5} {r['det_f1']:>7} {r['sym_f1']:>7} "
              f"{r['braille_cer']:>7} {r['detik_per_foto']:>7}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(hasil, f, indent=2)


if __name__ == "__main__":
    main()
