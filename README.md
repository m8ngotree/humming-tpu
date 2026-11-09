# humming-tpu

Quantized GEMM kernels for TPU in Pallas, inspired by
[vllm-project/humming](https://github.com/vllm-project/humming) (Apache-2.0).
This is a separate project, not a fork: Humming is CUDA C++ and none of its code applies to TPU. What carries over is the design:
- one format spec for weight type, scale granularity, scale dtype and zero points (field names follow Humming's `LayerConfig`);
- an offline repack into a layout the kernel wants;
- tile configs chosen by M range.

## Why

On vLLM's TPU backend ([tpu-inference](https://github.com/vllm-project/tpu-inference)), as of 2026-11-08:
- The AWQ linear path has **no kernel**. It runs `(qweight - qzeros) * scales` and then a plain `einsum` (`layers/vllm/quantization/awq.py`).
- The Pallas `quantized_matmul` raises `NotImplementedError("zero_point is not supported.")`.
- Sub-8-bit formats other than int4 have no packed storage anywhere in the TPU stack (Qwix stores 3-bit weights in int4 containers).

Decode is HBM-bound, so a 4-bit weight should be close to 4× faster than bf16 at small M. The question is how much of that the current path captures.

## Layout

```
humming_tpu/
  config.py         WeightConfig: format spec (Humming LayerConfig subset), weight is [K, N]
  reference.py      fp32 quantize / dequantize / matmul, the source of truth
  baselines.py      bf16 matmul + exact copy of tpu-inference's AWQ path
  roofline.py       per-chip HBM / MXU bounds
  importers/awq.py  AutoAWQ qweight/qzeros/scales <-> standard format
  kernels/wna16.py  Pallas W{4,8}A16 group-quant matmul (v0)
bench/phase1_gap.py measures every path vs roofline, M = 1..2048, Qwen3-8B shapes
scripts/tpu.sh      spot TPU VM up / sync / run / fetch / down
```

```bash
uv sync && uv run pytest -q                 # CPU, Pallas in interpret mode
uv run python bench/phase1_gap.py --smoke   # CPU smoke test of the benchmark
```

## Roadmap

| Stage | What | Done when |
|---|---|---|
| 0 ✅ | Format spec, fp32 reference, AWQ importer, baseline copy, roofline, v0 kernel, CPU tests | 18 tests pass in interpret mode |
| 1 | **Measure the gap on v6e-1**: bf16 vs xla_awq vs wna16_v0, % of roofline, plus whether XLA materializes the dequantized weight | Table in `results/`. **Go/no-go:** if xla_awq is already ≥ 80% of roofline at decode, pivot to stage 4 (W4A8) |
| 2 | Make v0 fast at decode (M ≤ 64): tile sweep, scaling in output space (`Σ_g s_g·(x_g@q_g) − xsum@(s·z)`), bf16 VPU math on v6e, bigger weight DMAs, split K for small N | ≥ 75% of HBM roofline on Qwen3-8B shapes, M=1..32 |
| 3 | Prefill regime + per-chip M-range dispatch table; upstream PR into tpu-inference's AWQ/GPTQ path, with end-to-end vLLM numbers | PR open, write-up #1 |
| 4 | W4A8 / W8A8 on the int8 MXU (v5e/v6e): dynamic per-token activation quant | Prefill beats bf16 |
| 5 | MoE grouped GEMM (gmm-style, scalar-prefetched group offsets) for W4A16-zp / W4A8 / MXFP4 | Faster than tpu-inference gmm_v2 on Qwen3-30B-A3B |
| 6 | Format breadth: real packed int2/3/5/6 (custom uint32 layouts), small FP weights, e8m0/fp8 scales | Humming-style support matrix, each cell measured |

The current order puts measurement before optimization. The Stage 1 numbers may reorder stages 2–6.
