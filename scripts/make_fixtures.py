#!/usr/bin/env python
"""Hasilkan fixture emas untuk menguji port Kotlin.

Keluaran Python di sini menjadi acuan kebenaran: implementasi Kotlin wajib
menghasilkan hal yang sama persis. Tanpa fixture ini, port Kotlin hanya
"tampak benar" tanpa bisa dibuktikan.

Tiga berkas dihasilkan:

  codec_cases.json   teks <-> braille: 64 bitmask + kalimat uji
  layout_cases.json  sel (posisi + mask) -> baris, spasi, paragraf, teks
  decoder_case.json  tensor keluaran model kecil -> daftar sel

python scripts/make_fixtures.py --out ../brailly/app/src/test/resources/fixtures
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.braille import BrailleCodec, mask_to_dots  # noqa: E402
from brailly.layout import build_layout  # noqa: E402
from brailly.metrics import gt_from_array  # noqa: E402
from brailly.pipeline import BraillyReader, ReaderConfig  # noqa: E402

KALIMAT = [
    "Saya membaca buku di perpustakaan.",
    "Pada tahun 1945 Indonesia merdeka!",
    "Apakah kamu sudah makan?",
    "Anak-anak bermain di taman; mereka senang.",
    "Harga 2,5 kilo adalah 12.500 rupiah.",
    "BRAILLE adalah huruf (titik timbul).",
    "Kelas 3a sudah pulang.",
    "Ia berkata: \u201cAyo belajar\u201d dengan semangat.",
    "Kemerdekaan Indonesia",
    "ada 89.",
]


def fixture_codec(codec: BrailleCodec) -> dict:
    return {
        "catatan": "acuan dari brailly/braille.py (Python)",
        "mask_ke_dots": {str(m): mask_to_dots(m) for m in range(64)},
        "encode": [
            {"teks": t, "sel": [(-1 if c is None else int(c)) for c in codec.encode(t)]}
            for t in KALIMAT
        ],
        "decode": [
            {"sel": [(-1 if c is None else int(c)) for c in codec.encode(t)], "teks": t}
            for t in KALIMAT
        ],
    }


def fixture_layout(paths, codec: BrailleCodec, batas: int) -> dict:
    kasus = []
    for p in paths[:batas]:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        csv = os.path.splitext(p)[0] + ".csv"
        baris = []
        with open(csv, encoding="utf-8") as f:
            for line in f:
                q = line.strip().split(";")
                if len(q) >= 5 and 1 <= int(q[4]) <= 63:
                    baris.append((float(q[0]) * w, float(q[1]) * h,
                                  float(q[2]) * w, float(q[3]) * h, int(q[4])))
        if not baris:
            continue
        sel = gt_from_array(np.float32(baris).reshape(-1, 5))
        lay = build_layout(sel, codec)
        kasus.append({
            "nama": os.path.basename(p),
            "masukan": [[round(float(c.cx), 3), round(float(c.cy), 3), round(float(c.w), 3),
                         round(float(c.h), 3), int(c.mask)] for c in sel],
            "harapan": {
                "sudut_derajat": round(lay.angle_deg, 4),
                "jarak_sel": round(lay.pitch, 4),
                "baris": [{"sel": [(-1 if m is None else int(m)) for m in l.cells],
                           "teks": l.text} for l in lay.lines],
                "paragraf": [{"jenis": pr.kind, "teks": pr.text} for pr in lay.paragraphs],
            },
        })
    return {"catatan": "acuan dari brailly/layout.py (Python)", "kasus": kasus}


def fixture_decoder(model: str) -> dict:
    """Potongan tensor keluaran model asli + sel yang harus dihasilkan."""
    from brailly.postprocess import decode_output

    reader = BraillyReader(model, ReaderConfig())
    rng = random.Random(0)
    from brailly.synth import SynthConfig, render_sample
    img, _ = render_sample(rng, SynthConfig(out_h=1024, out_w=768, dot_px_min=11, dot_px_max=11.5,
                                            p_page_edge=0, p_random_cells=0, p_negative=0,
                                            p_ink_text=0, max_rotation=2))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    keluaran = reader.backend(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]))[0]
    # ambil potongan agar fixture tetap kecil
    pot = keluaran[:, 40:104, 30:94].astype(np.float32)
    sel = decode_output(pot, 0.3, scale=1.0)
    return {
        "catatan": "acuan dari brailly/postprocess.py (Python). Tensor dipotong agar kecil.",
        "bentuk": list(pot.shape),
        "ambang": 0.3,
        "tensor": [round(float(v), 4) for v in pot.ravel()],
        "harapan": [{"cx": round(float(c.cx), 3), "cy": round(float(c.cy), 3),
                     "w": round(float(c.w), 3), "h": round(float(c.h), 3),
                     "mask": int(c.mask), "skor": round(float(c.score), 4)} for c in sel],
    }


def fixture_preprocess() -> dict:
    """Gambar kecil berukuran ganjil + tensor yang harus dihasilkan.

    Ukuran dipilih agar rasio skalanya pecahan, sehingga penyusutan area dan
    pelapisan tepi (BORDER_REPLICATE) ikut teruji.
    """
    from brailly.preprocess import resize_pad  # noqa: F401  (memastikan modul ada)

    w, h = 53, 37
    target_h, target_w = 32, 32
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)  # BGR

    skala = min(target_h / h, target_w / w)
    kecil = cv2.resize(img, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    dilapisi = cv2.copyMakeBorder(kecil, 0, target_h - kecil.shape[0], 0,
                                  target_w - kecil.shape[1], cv2.BORDER_REPLICATE)
    rgb = cv2.cvtColor(dilapisi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1))

    # piksel dikirim sebagai ARGB, menyamai Bitmap.getPixels() di Android
    argb = []
    for y in range(h):
        for x in range(w):
            b, g, r = (int(v) for v in img[y, x])
            argb.append((0xFF << 24) | (r << 16) | (g << 8) | b)
    return {
        "catatan": "acuan dari jalur pra-pemrosesan brailly/pipeline.py",
        "lebar": w,
        "tinggi": h,
        "target_tinggi": target_h,
        "target_lebar": target_w,
        "skala": round(float(skala), 8),
        "lebar_kecil": int(kecil.shape[1]),
        "tinggi_kecil": int(kecil.shape[0]),
        "piksel_argb": argb,
        "tensor": [round(float(v), 6) for v in tensor.ravel()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default="/projects/data/AngelinaDataset/uploaded/test.txt")
    ap.add_argument("--model", default="models/brailly.tflite")
    ap.add_argument("--layout-cases", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    codec = BrailleCodec()
    base = os.path.dirname(os.path.abspath(args.list))
    paths = [os.path.join(base, l.strip().replace("\\", "/"))
             for l in open(args.list, encoding="utf-8") if l.strip()]

    berkas = {
        "codec_cases.json": fixture_codec(codec),
        "layout_cases.json": fixture_layout(paths, codec, args.layout_cases),
        "decoder_case.json": fixture_decoder(args.model),
        "preprocess_case.json": fixture_preprocess(),
    }
    for nama, isi in berkas.items():
        path = os.path.join(args.out, nama)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(isi, f, ensure_ascii=False, separators=(",", ":"))
        print(f"{nama}: {os.path.getsize(path) / 1024:.0f} KB")
    n = len(berkas["layout_cases.json"]["kasus"])
    print(f"kasus layout: {n}, kasus codec: {len(KALIMAT)}, "
          f"sel pada kasus decoder: {len(berkas['decoder_case.json']['harapan'])}")


if __name__ == "__main__":
    main()
