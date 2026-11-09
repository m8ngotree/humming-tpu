# Stage 1: the quantized-weight gap on TPU v6e

v6e-1 spot (us-east1-d), jax 0.11.2, 2026-11-09. Raw data: `phase1_v6e.json`; optimized HLO: `hlo_v6e_*`.
Qwen3-8B linear shapes; uint4 weights use group 128 with zero points (the AWQ format).
Times are best-of-3 wall clock over 50 back-to-back calls. Roofline uses the 1640 GB/s spec.

## Speedup vs bf16 (higher is better; 4-bit ideal at decode is ~3.8×)

| Shape (K×N) | M | xla_w8 (int8, per-channel) | **xla_awq (tpu-inference today)** | wna16_v0 (ours) |
|---|---|---|---|---|
| gate_up 4096×24576 | 1 | 1.16× | **0.38×** | 0.91× |
| gate_up | 32 | 1.74× | **0.38×** | 0.85× |
| gate_up | 2048 | 0.72× | **0.58×** | 0.28× |
| down 12288×4096 | 1 | 1.35× | **0.44×** | 0.92× |
| down | 32 | 1.63× | **0.43×** | 0.88× |
| qkv 4096×6144 | 1 | 1.20× | **0.63×** | 0.94× |
| o 4096×4096 | 1 | 1.08× | **0.68×** | 0.97× |

## Findings

1. **tpu-inference's AWQ path is slower than unquantized bf16 at every M we measured, 0.38–0.79×.** Serving an AWQ checkpoint on TPU today buys memory capacity but loses speed.
2. **Cause: XLA does not fuse group-wise zero-point dequant into the dot.** The HLO has two fusions:
   - `convert_multiply_fusion` writes the whole dequantized `bf16[G,128,N]` weight;
   - a second fusion reads it back for the dot.

   On gate_up/down that buffer goes to HBM (201 MB / 101 MB temp), so the path reads 4 bits, writes 16 and reads 16 again. On qkv/o it fits in on-chip memory (`S(1)`), which is why those shapes lose less.
3. **XLA does fuse the simple case.** int8 with a per-channel scale is a single `multiply_reduce_fusion`, up to 1.8× faster than bf16 at decode. The failure is specific to group scales with zero points.
4. **Our v0 kernel is VPU-bound, not memory-bound.** It sits at a flat 19–26% of roofline from M=1 to 2048, roughly at bf16 speed at decode, so it already beats xla_awq by 1.3–2.4× there. Per weight element it does: int4→int32→f32 conversion, subtract the zero, multiply by the scale, then cast to bf16. That is too many VPU ops, so dequant arithmetic, not HBM, sets the time.
5. **Measurement caveat.** bf16 on qkv/o exceeds the spec roofline (up to 132%), and a read-bandwidth probe (`bench/hbm_bandwidth.py`) measured only 1.15–1.48 TB/s. Wall-clock percentages on small shapes are therefore unreliable. Stage 2 switches to profiler device time.

## Decision

**Go.** The headroom is about 4× on xla_awq at decode, and about 2.5–4× on our own v0. The core technical problem is **VPU instructions per weight element**, which shapes Stage 2 (see README).
