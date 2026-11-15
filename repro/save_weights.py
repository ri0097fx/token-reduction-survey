#!/usr/bin/env python3
"""Usage: python repro/save_weights.py --out $VTR_WORK/weights"""
import argparse
import os

import timm
import torch

NAMES = ["deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in NAMES:
        path = os.path.join(args.out, name + ".pth")
        if os.path.exists(path):
            print("exists", path)
            continue
        model = timm.create_model(name + ".fb_in1k", pretrained=True)
        torch.save(model.state_dict(), path)
        print("saved", path)


if __name__ == "__main__":
    main()
