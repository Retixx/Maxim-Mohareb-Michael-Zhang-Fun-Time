#!/usr/bin/env python3
"""
Does Qwen3-14B FP16 fit a fleet of 1..6 A100s?

Answers it by running the REAL accelerate shard planner against a meta-device
model. No weights are downloaded or loaded, so this runs on any machine --
including a 4 GB laptop -- and still produces the actual per-GPU byte
allocation that `device_map="auto"` would produce on the fleet.

It also sizes the KV cache from the model's own config at the frozen production
batch size, because weights are not the binding constraint: the Extractor stage
runs per-document over ~1,600-token contexts at batch 32.

Usage:
    python fleet_fit.py                          # Qwen3-14B fp16, 40 GiB cards
    python fleet_fit.py --gpu-gib 80
    python fleet_fit.py --model Qwen/Qwen3-8B --dtype fp16
"""

from __future__ import annotations

import argparse
import sys

import torch
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM

GIB = 1024 ** 3


def bytes_per_param(dtype: str) -> float:
    return {"fp16": 2.0, "bf16": 2.0, "8bit": 1.0, "4bit": 0.55}[dtype]


def kv_cache_bytes(cfg, batch: int, seq: int, dtype_bytes: int = 2) -> int:
    """2 (K and V) * layers * kv_heads * head_dim * batch * seq * bytes."""
    layers = cfg.num_hidden_layers
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return 2 * layers * kv_heads * head_dim * batch * seq * dtype_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    ap.add_argument("--gpu-gib", type=float, default=40.0)
    ap.add_argument("--max-gpus", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32, help="frozen production batch_size")
    ap.add_argument("--seq", type=int, default=1600, help="approx extractor context length")
    ap.add_argument("--reserve-gib", type=float, default=2.0,
                    help="per-GPU headroom for activations/fragmentation/cuda ctx")
    args = ap.parse_args()

    print(f"model      : {args.model}")
    print(f"dtype      : {args.dtype}")
    print(f"fleet      : up to {args.max_gpus} x {args.gpu_gib:.0f} GiB")
    print(f"batch/seq  : {args.batch} x {args.seq}")
    print(f"reserve    : {args.reserve_gib:.1f} GiB/GPU\n")

    cfg = AutoConfig.from_pretrained(args.model)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    weight_gib = n_params * bytes_per_param(args.dtype) / GIB
    kv_gib = kv_cache_bytes(cfg, args.batch, args.seq) / GIB

    print(f"parameters      : {n_params/1e9:.2f} B")
    print(f"weights ({args.dtype:4s})  : {weight_gib:8.2f} GiB")
    print(f"KV cache        : {kv_gib:8.2f} GiB "
          f"({cfg.num_hidden_layers} layers, "
          f"{getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)} kv-heads, "
          f"head_dim {getattr(cfg,'head_dim',None) or cfg.hidden_size//cfg.num_attention_heads})")
    print(f"weights + KV    : {weight_gib + kv_gib:8.2f} GiB\n")

    no_split = getattr(model, "_no_split_modules", None) or []
    dtype = torch.float16 if args.dtype in ("fp16", "8bit", "4bit") else torch.bfloat16

    usable = args.gpu_gib - args.reserve_gib
    print(f"{'GPUs':>5s} {'usable':>9s} {'total':>9s} {'fits?':>7s}   shard plan")
    print("-" * 78)

    verdict_1 = None
    for n in range(1, args.max_gpus + 1):
        max_memory = {i: f"{usable:.2f}GiB" for i in range(n)}
        try:
            dmap = infer_auto_device_map(
                model, max_memory=max_memory,
                no_split_module_classes=no_split, dtype=dtype,
            )
        except Exception as e:
            print(f"{n:5d} {usable:8.1f}G {usable*n:8.1f}G {'ERROR':>7s}   {type(e).__name__}: {e}")
            continue

        spilled = sorted({str(v) for v in dmap.values() if v in ("cpu", "disk")})
        gpus_used = sorted({v for v in dmap.values() if isinstance(v, int)})
        weights_fit = not spilled
        # KV cache is allocated on whichever GPU holds the corresponding layers,
        # so it spreads across the shard set roughly in proportion to layers.
        per_gpu_total = (weight_gib + kv_gib) / max(1, len(gpus_used))
        runtime_fit = weights_fit and per_gpu_total <= usable

        mark = "YES" if runtime_fit else ("weights" if weights_fit else "NO")
        detail = (f"{len(gpus_used)} GPU(s), ~{per_gpu_total:.1f} GiB/GPU incl. KV"
                  if weights_fit else f"SPILLS TO {','.join(spilled)}")
        print(f"{n:5d} {usable:8.1f}G {usable*n:8.1f}G {mark:>7s}   {detail}")
        if n == 1:
            verdict_1 = runtime_fit

    print()
    total = weight_gib + kv_gib
    need = int(-(-total // usable))
    print(f"minimum GPUs for weights+KV at batch {args.batch}: {need} "
          f"x {args.gpu_gib:.0f} GiB (usable {usable:.1f} GiB each)")
    if verdict_1 is False and need <= args.max_gpus:
        print(f"=> single-GPU is NOT sufficient; a {need}-way shard is. "
              f"A {args.max_gpus}-GPU fleet is comfortably sufficient.")
    elif verdict_1:
        print("=> fits on a single GPU; sharding is optional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
