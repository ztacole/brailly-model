#!/usr/bin/env python
"""Ekspor checkpoint ke TFLite (LiteRT) langsung dari PyTorch, lalu verifikasi.

Ekspor dilakukan dari checkpoint PyTorch memakai litert-torch (dahulu
ai-edge-torch) -- BUKAN konversi ONNX->TFLite yang rawan rusak.

TFLite memerlukan bentuk masukan tetap, jadi ukurannya ditentukan di sini. Untuk
ponsel ini justru menguntungkan: bentuk statis diperlukan delegate GPU/NPU, dan
deteksi cukup satu lintasan bila jarak foto dipandu overlay kamera.

Mode kuantisasi:
  none          fp32, paling akurat, ~12 MB
  dynamic-int8  bobot int8 per-kanal, aktivasi fp32. Tanpa kalibrasi, ~3,5 MB.
                Aktivasi tidak dipotong, jadi akurasinya paling terjaga.
  static-int16  bobot int8, aktivasi int16. Perlu kalibrasi.
  static-int8   bobot dan aktivasi int8. Paling kecil dan paling cepat di NPU,
                tetapi logit heatmap (jangkauan +-15) mudah terpotong.

python scripts/export_tflite.py --ckpt runs/ft/best.pt --out models/brailly.tflite
python scripts/export_tflite.py --ckpt runs/ft/best.pt --out models/brailly_int8.tflite \
    --quant-mode dynamic-int8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.model import BraillyNet  # noqa: E402

MODE = ("none", "dynamic-int8", "static-int16", "static-int8")


def muat_model(ckpt: str) -> BraillyNet:
    model = BraillyNet(pretrained=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    return model


def contoh_data(n: int, h: int, w: int, seed: int = 0):
    """Data representatif untuk kalibrasi: halaman Braille sintetis."""
    import random

    import cv2

    from brailly.synth import SynthConfig, render_sample

    rng = random.Random(seed)
    cfg = SynthConfig(out_h=h, out_w=w, dot_px_min=7, dot_px_max=14)
    keluar = []
    for _ in range(n):
        img, _ = render_sample(rng, cfg)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        keluar.append(rgb.transpose(2, 0, 1)[None])
    return keluar


def kuantisasi(path_fp32: str, path_out: str, mode: str, kalib, nama_masukan: str):
    from ai_edge_quantizer import quantizer, recipe

    resep = {
        "dynamic-int8": recipe.dynamic_wi8c_afp32,
        "static-int16": recipe.static_wi8_ai16,
        "static-int8": recipe.static_wi8_ai8,
    }[mode]
    qt = quantizer.Quantizer(path_fp32, resep())
    if mode.startswith("static"):
        print(f"kalibrasi dengan {len(kalib)} halaman sintetis ...")
        # calibrate() mengharapkan data per-signature: {nama_signature: [{masukan: data}]}
        data = {"serving_default": [{nama_masukan: x} for x in kalib]}
        hasil = qt.quantize(qt.calibrate(data))
    else:
        hasil = qt.quantize()
    hasil.export_model(path_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--quant-mode", choices=MODE, default="none")
    ap.add_argument("--calib", type=int, default=24, help="jumlah halaman sintetis untuk kalibrasi")
    args = ap.parse_args()

    assert args.height % 32 == 0 and args.width % 32 == 0, "tinggi/lebar harus kelipatan 32"
    model = muat_model(args.ckpt)
    contoh = (torch.rand(1, 3, args.height, args.width),)

    import litert_torch

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    antara = args.out if args.quant_mode == "none" else args.out + ".fp32.tmp"
    litert_torch.convert(model, contoh).export(antara)

    if args.quant_mode != "none":
        from ai_edge_litert.interpreter import Interpreter
        # kalibrasi memakai nama masukan pada signature ('args_0'),
        # bukan nama tensor ('serving_default_args_0')
        sig = Interpreter(model_path=antara).get_signature_list()["serving_default"]
        nama = sig["inputs"][0]
        kalib = contoh_data(args.calib, args.height, args.width) if args.quant_mode.startswith("static") else []
        kuantisasi(antara, args.out, args.quant_mode, kalib, nama)
        os.remove(antara)

    print(f"tersimpan {args.out}  {os.path.getsize(args.out) / 1e6:.1f} MB  "
          f"masukan {args.width}x{args.height}  mode {args.quant_mode}")

    # ---- verifikasi terhadap PyTorch
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=args.out)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    skala, nol = out["quantization"]
    print(f"masukan {inp['shape']} {inp['dtype'].__name__} -> "
          f"keluaran {out['shape']} {out['dtype'].__name__}"
          + (f" (skala {skala:.6f}, titik-nol {nol})" if skala else ""))

    selisih = []
    for x in contoh_data(3, args.height, args.width, seed=777):
        with torch.no_grad():
            acuan = model(torch.from_numpy(x)).numpy()
        interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
        interp.invoke()
        got = interp.get_tensor(out["index"]).astype(np.float32)
        if skala:
            got = (got - nol) * skala
        selisih.append(float(np.abs(acuan - got).max()))
    print(f"selisih logit maks vs PyTorch: {max(selisih):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
