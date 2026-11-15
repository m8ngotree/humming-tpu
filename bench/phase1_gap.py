"""Phase 1: how far are XLA's quantized-weight paths from roofline on TPU?

Paths per (layer shape, M):
  bf16          plain bf16 weight
  xla_w8        int8 weight, per-channel scale, XLA dequant (simplest fusable case)
  xla_awq       tpu-inference's current AWQ path: uint4 + group zero/scale, XLA dequant
  wna16_v0      our first Pallas kernel, same uint4 format

For the XLA paths we also record compiled temp-buffer bytes: a temp the size of a
bf16 weight means XLA materialized the dequantized weight in HBM instead of
fusing dequant into the dot. --dump-hlo writes the optimized HLO for inspection.

TPU VM:   uv run python bench/phase1_gap.py --chip v6e --dump-hlo
CPU:      uv run python bench/phase1_gap.py --smoke
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp

from humming_tpu import baselines, reference, roofline
from humming_tpu.config import WeightConfig
from humming_tpu.kernels import wna16

# Qwen3-8B linear layers as (K, N); qkv and gate_up are fused projections.
SHAPES = {
    "qkv": (4096, 6144),
    "o": (4096, 4096),
    "gate_up": (4096, 24576),
    "down": (12288, 4096),
}
M_VALUES = (1, 8, 32, 128, 512, 2048)
GROUP = 128


def time_us(fn: Callable, *args, iters: int) -> float:
    """Best-of-3 mean wall time per call, after a compile/warmup call."""
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - start) / iters * 1e6)
    return best


def compiled(fn: Callable, *args):
    return fn.lower(*args).compile()


def temp_bytes(fn: Callable, *args) -> int | None:
    analysis = compiled(fn, *args).memory_analysis()
    return None if analysis is None else int(analysis.temp_size_in_bytes)


def scrub_paths(text: str) -> str:
    """Replace absolute source paths with repo-relative ones (HLO metadata embeds them)."""
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    return text.replace(repo_root + "/", "").replace(str(pathlib.Path.home()), "~")


def build_weights(k: int, n: int):
    cfg = WeightConfig(shape_k=k, shape_n=n, b_dtype="uint4", weight_scale_group_size=GROUP)
    w = jax.random.normal(jax.random.key(0), (k, n), jnp.float32) * 0.02
    q, s, z = reference.quantize(w, cfg)
    cfg_w8 = WeightConfig(
        shape_k=k, shape_n=n, b_dtype="int8", weight_scale_group_size=0, has_zero_point=False
    )
    q8, s8, _ = reference.quantize(w, cfg_w8)
    return {
        "cfg": cfg,
        "cfg_w8": cfg_w8,
        "bf16": (w.astype(jnp.bfloat16),),
        "xla_w8": (q8.astype(jnp.int8), s8.astype(jnp.bfloat16)),
        "xla_awq": baselines.prepare_xla_awq(q, s, z, GROUP),
        "wna16_v0": wna16.prepare(q, s, z, cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chip", default="v6e", choices=sorted(roofline.CHIPS))
    parser.add_argument("--smoke", action="store_true", help="tiny shape, interpret mode")
    parser.add_argument("--dump-hlo", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    shapes, m_values, iters, interpret = SHAPES, M_VALUES, 50, False
    if args.smoke:
        shapes, m_values, iters, interpret = {"tiny": (1024, 1024)}, (1, 16), 2, True
    chip = roofline.CHIPS[args.chip]
    tag = f"{args.chip}{'_smoke' if args.smoke else ''}"
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)
    print(f"jax {jax.__version__}  devices {jax.devices()}  roofline model {chip.name}")

    v0 = jax.jit(functools.partial(wna16.wna16_matmul, group=GROUP, interpret=interpret))
    paths = {
        "bf16": baselines.bf16_matmul,
        "xla_w8": baselines.xla_w8_channel,
        "xla_awq": baselines.xla_awq,
        "wna16_v0": v0,
    }

    rows = []
    for name, (k, n) in shapes.items():
        weights = build_weights(k, n)
        weight_cfg = {
            "bf16": None,
            "xla_w8": weights["cfg_w8"],
            "xla_awq": weights["cfg"],
            "wna16_v0": weights["cfg"],
        }
        for m in m_values:
            x = jax.random.normal(jax.random.key(1), (m, k), jnp.float32).astype(jnp.bfloat16)
            times = {p: time_us(fn, x, *weights[p], iters=iters) for p, fn in paths.items()}
            for path, fn in paths.items():
                bound = roofline.matmul_bound(m, k, n, weight_cfg[path], chip)
                row = {
                    "shape": name,
                    "k": k,
                    "n": n,
                    "m": m,
                    "path": path,
                    "us": round(times[path], 2),
                    "bound_us": round(bound.us, 2),
                    "pct_roofline": round(100 * bound.us / times[path], 1),
                    "speedup_vs_bf16": round(times["bf16"] / times[path], 2),
                }
                if path.startswith("xla_"):
                    tmp = temp_bytes(fn, x, *weights[path])
                    row["temp_bytes"] = tmp
                    row["materializes_weight"] = tmp is not None and tmp >= 0.9 * 2 * k * n
                rows.append(row)
                print(
                    f"{name:8s} M={m:5d} {path:9s} {row['us']:9.1f}us "
                    f"bound {row['bound_us']:8.1f}us {row['pct_roofline']:5.1f}% "
                    f"{row['speedup_vs_bf16']:5.2f}x-bf16 "
                    f"{'temp=' + str(row.get('temp_bytes')) if 'temp_bytes' in row else ''}"
                )
            if args.dump_hlo and m in (1, 512):
                for path in ("xla_w8", "xla_awq"):
                    hlo = scrub_paths(compiled(paths[path], x, *weights[path]).as_text())
                    (out_dir / f"hlo_{tag}_{name}_{path}_m{m}.txt").write_text(hlo)

    out = pathlib.Path(args.out or out_dir / f"phase1_{tag}.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
