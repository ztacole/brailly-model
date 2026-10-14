#!/usr/bin/env python
"""Export a checkpoint to ONNX (dynamic H/W, multiples of 32) and verify it.

python scripts/export_onnx.py --ckpt runs/synth/best.pt --out models/brailly.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.model import BraillyNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = BraillyNet(pretrained=False)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    x = torch.rand(1, 3, 640, 480)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(model, x, args.out, input_names=["image"], output_names=["maps"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "maps": {0: "batch", 2: "h4", 3: "w4"}},
                      opset_version=args.opset, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for shape in [(1, 3, 640, 480), (1, 3, 1024, 768)]:
        xi = torch.rand(*shape)
        with torch.no_grad():
            ref = model(xi).numpy()
        got = sess.run(None, {"image": xi.numpy()})[0]
        print(shape, "max abs diff", float(np.abs(ref - got).max()))
    print("saved", args.out, f"{os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
