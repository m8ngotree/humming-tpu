# Target hardware

Specs come from JAX's own chip table (`jax/_src/tpu_info.py`, jax 0.11.2), except HBM bandwidth (public spec, measured where noted).
Per-chip numbers; v7x has 2 TensorCores per chip, and its per-core numbers are half.

| | v5e | v6e | **v7x** |
|---|---|---|---|
| TensorCores / chip | 1 | 1 | **2** |
| bf16 MXU | 197 TF | 920 TF | **2.31 PF** |
| int8 × int8 MXU | 394 T | 1.84 P | **none** |
| int4 × int4 MXU | 788 T | 3.68 P | **none** |
| fp8 MXU | none | e5m2 / e4m3b11 only, bf16 speed | **4.6 PF; e4m3fn and e5m2; also bf16 × fp8 mixed** |
| VMEM / core | 128 MiB | 128 MiB | **64 MiB** (tpu-inference caps it at 48) |
| HBM / chip | 16 GiB | 32 GiB | **192 GiB** |
| HBM bandwidth | 819 GB/s | 1640 GB/s spec, 1.15–1.48 TB/s measured | **7.37 TB/s** spec |
| Spot price (2026-11-15) | — | $0.24–0.71/hr | $4.12/hr (`tpu7x-standard-1t`, us-central1 / us-east1) |

## What this means for the kernels

- **Low-precision MXU paths differ by chip:**
  - **v5e/v6e:** the fast paths are **int8 and int4**, so W4A8 and W4A4 run on the integer MXU.
  - **v7x:** the fast path is **fp8**. W4A8 becomes int4 → fp8 e4m3, which is exact for every 4-bit integer, followed by an fp8 × fp8 matmul. tpu-inference already ships the symmetric version of this (#1841, #2236).
- **v7x multiplies bf16 × fp8 natively.** For W4A16 on v7x, weights can be decoded only as far as fp8, with no bf16 dequant. Whether int → fp8 conversion is cheaper on the VPU than int → bf16 needs measuring.
- **FP4 (e2m1) decodes exactly into both targets:**
  - into fp8 e4m3 on v7x;
  - with MXFP4's power-of-two scales folded in, into int8 on v6e. All e2m1 values × 2 are integers in [−12, 12].
- **On v7x, JAX lists each TensorCore as its own device.** A single-device kernel runs on half a chip: 3.69 TB/s and 1.15 PF bf16, which `roofline.CHIPS["v7x"]` uses. Using the whole chip means sharding across both devices, or the core-parallel APIs (`pltpu.get_tpu_info().num_cores`, `pltpu.TensorCoreMesh`) that some tokamax kernels use. A core also has half of v6e's VMEM, so tiles are smaller.
- **Decode is ~4.5× more bandwidth-hungry on v7x** (7.37 TB/s), so VPU dequant cost per byte matters even more there than on v6e.

## How we use the chips

- **CPU (interpret mode):** correctness while writing.
- **v6e-1 spot:** first Mosaic compile, learning, cheap iteration. Also the v5e/v6e kernel variants.
- **v7x `tpu7x-standard-1t` spot:** the frontier target. Every kernel gets a v7x variant, tuned and benchmarked there. This is a GCE machine type, not the TPU-VM API, so `scripts/tpu.sh` needs a v7x mode. Quota is unknown until the first create attempt.
