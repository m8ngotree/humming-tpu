"""Phase 1: how far is the current TPU AWQ path from roofline?

Paths, per (layer shape, M):
  bf16      plain bf16 weight einsum (what you'd run unquantized)
  xla_awq   tpu-inference's current AWQ path (XLA dequant + einsum)
  wna16_v0  our first Pallas kernel

Also reports, per shape, whether XLA's compiled xla_awq allocates a temp buffer
the size of a dequantized bf16 weight, a cheap signal that it materializes the
dequantized weight in HBM instead of fusing the dequant into the dot.

Usage (on a TPU VM):   uv run python bench/phase1_gap.py --chip v6e
CPU smoke test:        uv run python bench/phase1_gap.py --smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp

from humming_tpu import baselines, reference, roofline
from humming_tpu.config import WeightConfig
from humming_tpu.kernels import wna16

# Qwen3-8B linear layers (K, N); qkv and gate_up are the fused projections.
SHAPES = {
    "qkv": (4096, 6144),
    "o": (4096, 4096),
    "gate_up": (4096, 24576),
    "down": (12288, 4096),
}
MS = [1, 8, 32, 128, 512, 2048]
GROUP = 128


def time_us(fn, *args, iters: int = 50) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / iters * 1e6)
    return min(times)


def temp_bytes(fn, *args) -> int | None:
    try:
        return fn.lower(*args).compile().memory_analysis().temp_size_in_bytes
    except Exception:  # memory_analysis is backend-dependent
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chip", default="v6e", choices=list(roofline.CHIPS))
    p.add_argument("--smoke", action="store_true", help="tiny shapes, interpret mode, CPU")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    shapes, ms, iters = SHAPES, MS, 50
    interpret = False
    if args.smoke:
        shapes, ms, iters, interpret = {"tiny": (1024, 1024)}, [1, 16], 2, True
    chip = roofline.CHIPS[args.chip]
    print(f"devices: {jax.devices()}  chip model: {chip.name}")

    rows = []
    for name, (k, n) in shapes.items():
        cfg = WeightConfig(shape_k=k, shape_n=n, b_dtype="uint4", weight_scale_group_size=GROUP)
        w = jax.random.normal(jax.random.key(0), (k, n), jnp.float32) * 0.02
        q, s, z = reference.quantize(w, cfg)
        w_bf16 = w.astype(jnp.bfloat16)
        awq_args = baselines.prepare_xla_awq(q, s, z, GROUP)
        kq, ks, kz = wna16.prepare(q, s, z, cfg)
        del w, q

        for m in ms:
            x = jax.random.normal(jax.random.key(1), (m, k), jnp.float32).astype(jnp.bfloat16)
            v0 = jax.jit(lambda x, a, b, c: wna16.wna16_matmul(x, a, b, c, group=GROUP,
                                                               interpret=interpret))
            results = {
                "bf16": (time_us(baselines.bf16_matmul, x, w_bf16, iters=iters),
                         roofline.bound_us(m, None, k, n, chip)["bound_us"]),
                "xla_awq": (time_us(baselines.xla_awq, x, *awq_args, iters=iters),
                            roofline.bound_us(m, cfg, k, n, chip)["bound_us"]),
                "wna16_v0": (time_us(v0, x, kq, ks, kz, iters=iters),
                             roofline.bound_us(m, cfg, k, n, chip)["bound_us"]),
            }
            tmp = temp_bytes(baselines.xla_awq, x, *awq_args)
            materializes = tmp is not None and tmp >= k * n * 2 * 0.9
            for path, (t, bound) in results.items():
                row = dict(shape=name, k=k, n=n, m=m, path=path, us=round(t, 2),
                           bound_us=round(bound, 2), pct_roofline=round(100 * bound / t, 1),
                           speedup_vs_bf16=round(results["bf16"][0] / t, 2))
                if path == "xla_awq":
                    row["xla_temp_bytes"] = tmp
                    row["xla_materializes_bf16_weight"] = materializes
                rows.append(row)
                print(f"{name:8s} M={m:5d} {path:9s} {t:10.1f} us  bound {bound:8.1f} us  "
                      f"{row['pct_roofline']:5.1f}% roofline  {row['speedup_vs_bf16']:5.2f}x vs bf16"
                      + (f"  temp={tmp}" if path == "xla_awq" else ""))

    out = pathlib.Path(args.out or f"results/phase1_{args.chip}{'_smoke' if args.smoke else ''}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
