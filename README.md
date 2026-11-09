# humming-tpu

Quantized GEMM kernels for TPU in Pallas, inspired by
[vllm-project/humming](https://github.com/vllm-project/humming) (Apache-2.0).
This is a separate project, not a fork: Humming is CUDA C++ and none of its code applies to TPU. What carries over is the design:
- one format spec for weight type, scale granularity, scale dtype and zero points (field names follow Humming's `LayerConfig`);
- an offline repack into the layout the kernel wants;
- tile configs chosen by M range.

## Why

Measured on v6e ([results/phase1_v6e.md](results/phase1_v6e.md)):
- **vLLM's TPU AWQ path is slower than unquantized bf16: 0.38–0.79×.**
- XLA does not fuse group-wise zero-point dequant into the matmul. It writes the full dequantized bf16 weight first, to HBM on large layers, then runs the dot.
- tpu-inference's Pallas `quantized_matmul` rejects zero points.

A 4-bit weight should be close to 4× *faster* than bf16 at decode.

## Layout

```
humming_tpu/
  config.py         WeightConfig: format spec (Humming LayerConfig subset), weight is [K, N]
  reference.py      fp32 quantize / dequantize / matmul, the source of truth
  baselines.py      bf16, int8 per-channel, and an exact copy of tpu-inference's AWQ path
  roofline.py       per-chip HBM / MXU bounds
  importers/awq.py  AutoAWQ qweight/qzeros/scales <-> standard format
  kernels/wna16.py  Pallas W{4,8}A16 group-quant matmul (v0)
bench/phase1_gap.py   every path vs roofline, M = 1..2048, Qwen3-8B shapes, HLO dump
bench/hbm_bandwidth.py achievable HBM read bandwidth
scripts/tpu.sh        spot TPU VM up / sync / run / fetch / down
```

```bash
uv sync && uv run pytest -q                      # CPU, Pallas interpret mode
HUMMING_TPU_COMPILE=1 uv run pytest -q           # on a TPU VM: real Mosaic compile
uv run python bench/phase1_gap.py --smoke        # CPU smoke test of the benchmark
```

## Plan

The main lesson from Stage 1: on TPU, **the cost of dequant arithmetic on the VPU sets the speed, even at decode**. The VPU is TPU's vector unit, which is weak relative to the MXU. Our straightforward kernel moves 4× fewer bytes but runs no faster than bf16. Every stage below is organized around cutting VPU instructions per weight element.

| Stage | What | Done when |
|---|---|---|
| 0 ✅ | Format spec, fp32 reference, AWQ importer, baselines, roofline, v0 kernel | Tests pass in interpret mode and under Mosaic on v6e |
| 1 ✅ | Measure the gap on v6e | [Go](results/phase1_v6e.md): xla_awq is 0.38–0.79× bf16; v0 is VPU-bound at ~20% of roofline |
| 2 | **Cheap dequant.** Measured one change at a time with profiler device time: **(a)** apply scales and zeros in output space: `acc += (x_g @ q_g)·s_g`, with the zero term as `xsum @ (s·z)`, so dequant work is per output, not per weight; **(b)** offline repack plus a bit trick that turns int4 into bf16 in ~2 bitwise ops per 32-bit word (`0x4300 \| nibble` = 128+n; fold the −128 into the zero point), Marlin's idea mapped onto TPU's packed-bf16 vreg layout; **(c)** tile/DMA sweep and per-M dispatch | ≥ 70% of HBM roofline at M ≤ 32 on Qwen3-8B shapes (~3× bf16) |
| 3 | **Upstream.** tpu-inference issue with Stage 1 data, then a PR routing AWQ/GPTQ linears to the kernel; end-to-end vLLM numbers; write-up #1 | PR open |
| 4 | **Prefill: W4A8 on the int8 MXU.** At large M, even perfect bf16 dequant is VPU-bound; int4→int8 is cheaper and the int8 MXU runs at 2× bf16 | Prefill beats bf16 |
| 5 | **MoE grouped GEMM** with the same codec (W4A16-zp, W4A8, MXFP4) | Beats gmm_v2 on Qwen3-30B-A3B |
| 6 | **Format breadth, the Humming part:** generalize the Stage 2b repack + bit-trick codec to N-bit (true 3/5/6-bit packing) and small-FP weights | A support matrix with every cell measured |
