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

[docs/landscape.md](docs/landscape.md) lists everything Humming and DeepGEMM offer and which of it TPU already covers. The open space is Humming's: flexible low-bit **weight** formats.
[docs/hardware.md](docs/hardware.md) explains why each kernel needs a v5e/v6e variant and a v7x variant: v5e/v6e have int8/int4 MXUs, while v7x has fp8 and bf16 × fp8.

The main lesson from Stage 1: on TPU, **the cost of dequant arithmetic on the VPU sets the speed, even at decode**. Our straightforward kernel moves 4× fewer bytes but runs no faster than bf16. Every kernel is organized around cutting VPU instructions per weight element.

Done so far:
- **Stage 0:** format spec, fp32 reference, AWQ importer, baselines, roofline, v0 kernel; tests pass in interpret mode and under Mosaic on v6e.
- **Stage 1:** [the gap on v6e](results/phase1_v6e.md). tpu-inference's AWQ path is 0.38–0.79× bf16; v0 is VPU-bound at ~20% of roofline.

| Kernel | What | v5e / v6e path | v7x path | Done when |
|---|---|---|---|---|
| **K1** `wna16` dense | int4 + zero points (AWQ/GPTQ), then int8. **(a)** apply scales/zeros in output space: `acc += (x_g @ q_g)·s_g`, zero term `xsum @ (s·z)`; **(b)** offline repack + bit-trick decode (`0x4300 \| nibble` = 128+n); **(c)** tile sweep, per-M dispatch | bf16 MXU | bf16 or **bf16 × fp8** MXU (decode only to fp8), split across 2 TensorCores | ≥ 70% of HBM roofline at M ≤ 32 on both chips; tpu-inference PR |
| **K2** `wna8` / `wna4` dense, prefill | Weight × quantized-activation matmul, plus a fused activation-quantize kernel | W4A8 on int8 MXU, **W4A4 on int4 MXU** (4× bf16) | W4A8-fp8: int4 → e4m3 (exact), fp8 MXU (2× bf16) | Prefill beats bf16 |
| **K3** grouped MoE | K1/K2 over uneven expert groups | same | same | Beats `gmm_v2` on Qwen3-30B-A3B |
| **K4** packed low-bit | True int2/int3 (and ternary) packing, then int5/int6 | bf16 / int8 | bf16 / fp8 | 3-bit faster than 4-bit at decode |
| **K5** FP4 weights | MXFP4 (e8m0 scales), NVFP4 (two-level scales) | e2m1 × 2 → int8 (exact) | e2m1 → e4m3 (exact) | Faster than today's MXFP4 path |

Each kernel reuses the previous one's pieces (codec, scaling strategy, MXU path, tiling), so later formats are new codecs, not new kernels.

## License

Apache-2.0, see [LICENSE](LICENSE). `humming_tpu/baselines.py` reproduces a few lines of
[tpu-inference](https://github.com/vllm-project/tpu-inference) (Apache-2.0), attributed in place.
