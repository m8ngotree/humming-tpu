# Humming and DeepGEMM: everything they offer

A catalog of both GPU libraries, read from source: Humming `main` 2026-11-13 and DeepGEMM `main` 2026-10-20.
The last section maps each item to its TPU status as of 2026-11-15.

---

## Humming (vllm-project/humming)

A JIT-compiled quantized GEMM library. One templated CUDA kernel family is specialized at runtime from three configs.

### Kernels

| Kernel | What it does |
|---|---|
| `HummingKernel` | The GEMM itself: dense or MoE, any supported weight × activation format. Three pipeline families: standard, warp-specialized and UMMA (Blackwell) |
| `ProcessInputKernel` / `ProcessInputScaleKernel` | Activation pre-pass. Quantizes inputs (per-tensor, per-token or per-group, dynamic or static) and can fuse an activation function (e.g. `silu(a)*b`), a Hadamard rotation, MoE scatter and zeroing of invalid rows. Runs as one fused pass or two phases (`collect_absmax` → `quantize`) |
| `QuantWeightKernel` | Quantizes a float weight into any supported format |
| `DequantKernel` | Quantized weight → float (reference and debug) |
| `PackWeightKernel` / `UnpackWeightKernel` | Packs N-bit values into 32-bit words and back, for any bit-width 1–8 |
| `RepackWeightKernel` | Offline rearrangement into the layout the GEMM wants (Marlin-style) |
| `ConvertWeightKernel` | Converts between weight formats (e.g. symmetric INT → `float8e0m7`) |
| `ProcessMxfp4W4A8Kernel` | Prepares MXFP4 weights for the FP8-activation path |
| `TopsBenchKernel` | Measures peak tensor-core throughput per dtype and MMA type |

### Data types

- **Integer weights:**
  - `uint1`–`uint8` (asymmetric, with zero point);
  - `int2`, `int3`, `int4`, `int6`, `int8` (symmetric).
- **Float weights and activations:**
  - **Zero-exponent formats** `float2e0m1` … `float8e0m7`: these represent symmetric INT exactly, and Humming converts eligible INT weights to them automatically;
  - **Standard small floats:** `float4e2m1` (FP4), `float6e2m3`, `float6e3m2`, `float8e3m4`, `float8e4m3`, `float8e5m2`, `float8e5m3`;
  - **Scale-only:** `float8e8m0` (UE8M0, used by MX formats).
- **Activations `a_dtype`:** fp16, bf16, fp8 e4m3, fp8 e5m2, fp4 e2m1, int8, int4.
- **Output `c_dtype`:** fp16 or bf16.
- **Scale dtypes `bs_dtype`/`as_dtype`:** bf16, fp16, fp32, E4M3, E5M2, UE8M0.

Which weight types each activation supports:

| Activation | Weights allowed |
|---|---|
| FP16 | INT1–8 (symmetric or zero point), any signed FP with ≤ 8 bits and ≤ 5 exponent bits |
| BF16 | INT1–8 (symmetric or zero point), any signed FP ≤ 8 bits |
| FP8 e4m3 | symmetric INT1–5, INT1–4 with zero point, FP with ≤ 4 exponent / ≤ 3 mantissa bits |
| FP8 e5m2 | symmetric INT1–4, INT1–3 with zero point, FP with ≤ 5 exponent / ≤ 2 mantissa bits |
| FP4 e2m1 | symmetric INT1–3, INT1–2 with zero point, FP with ≤ 2 exponent / ≤ 1 mantissa bits |
| INT8 | symmetric INT1–8, INT1–7 with zero point |
| INT4 | symmetric INT1–4, INT1–3 with zero point |

### `LayerConfig`: what the stored weights are

| Field | Options |
|---|---|
| `shape_n`, `shape_k`, `pad_shape_n`, `pad_shape_k` | Padded shape and the padding amounts |
| `num_experts` | `0` = dense, otherwise MoE |
| `b_dtype`, `a_dtype`, `c_dtype` | Weight, activation and output types (above) |
| `bs_dtype`, `as_dtype` | Weight and activation scale types |
| `weight_scale_type` | `group` (per K-group per channel), `block` (an N×K block shares a scale), `channel`, `tensor` |
| `weight_scale_group_size`, `weight_scale_group_size_n` | Group size along K, and along N for `block` scales |
| `weight_scale_2_type` | Optional second-level scale: `none`, `channel`, `tensor` (e.g. NVFP4's per-tensor FP32 scale) |
| `has_zero_point`, `is_fp_zero_point` | Integer zero points (same bit-width as the weights) or fp16/bf16 zero points |
| `input_quant_mode` | `none`, `static_tensor`, `dynamic_token`, `dynamic_group`, `static_tensor_dynamic_group`, `dynamic_group_token` |
| `input_scale_group_size` | Group size along K for activation scales |
| `use_int_weight_scale` | Integerizes group scales for INT8/INT4 activation paths (extracts a secondary scale) |
| `use_fused_e8m0_scale` | Folds E8M0 group scales into the FP4 → FP8/INT8 weight conversion |
| `use_packed_k_layout` | Packed-K layout for Hopper WGMMA with 8-bit activations |
| `has_bias` | Fused bias add in the epilogue |

### `ComputeConfig`: how to compute

| Field | Options |
|---|---|
| `gemm_type` | `dense`; `indexed` (MoE that reads tokens through routing indices); `grouped_contiguous` (each expert's tokens stored together); `grouped_masked` (fixed capacity per expert, plus a valid-row count) |
| `use_f16_accum` | fp16 accumulation (faster, less accurate) |
| `use_batch_invariant` | Deterministic results regardless of batch composition |
| `use_m_major_input_scale` | Layout of activation scales |

### `TuningConfig`: how the kernel runs

| Group | Options |
|---|---|
| Compute backend | `mma_type`: `mma` (SM75+), `wgmma` (Hopper), `umma` (Blackwell tcgen05), `mxmma` (Blackwell block-scaled) |
| Tiling | `block_shape` (M, N, K), `warp_shape` |
| Pipelining | `num_stages`, `producer_stage_unroll`, `consumer_stage_unroll`, `num_ctas_per_sm`, `use_warp_spec`, `producer_registers` |
| Data movement | `use_tma` plus per-tensor `use_tma_{a, b, c, as, as2, bs, bs2, bzp, bias}` |
| Output / shared memory | `output_chunk_rows`, `smem_reuse_mode` (`none`, `last_stage`, `all_stages`) |
| Scheduling and reuse | `use_stream_k`, `raster_group_m`, `evict_weights_first`, `multi_cast_size_a/b` (cluster multicast) |
| Backend-specific | `wgmma_use_late_as`, `wgmma_split_issue_wait`, `umma_num_dequant_warpgroups`, `umma_cta_group_size`, `umma_split_async_loads` |
| Between kernels | `use_pdl` (programmatic dependent launch) |

Per-device tuning heuristics exist for SM75, SM8x, SM90, SM90-H20, SM100, SM120, SM121, and T-Head PPU (SM80-compatible).

### Activation pre-pass options (`ProcessInputProblemConfig` / `ProcessInputTuningConfig`)

- **Problem config:**
  - `quant_mode`, `quant_dtype`, `quant_group_size`, `group_scale_dtype`;
  - `activation_type`: `none`, `unary`, `binary_split`, `binary_interleaved`, plus an `activation_impl` expression such as `silu_and_mul`;
  - `hadamard_block_size`;
  - `layout`: `normal`, `grouped_mask`, `scatter`, plus `scatter_width`;
  - `zero_invalid`.
- **Tuning config:** `threads_per_task`, `values_per_thread`, `tokens_per_block`, `use_tile_partition`, `separate_outputs`, `two_stage`, `use_pdl`.

### Checkpoint schemas (formats it can load)

GPTQ, AWQ, AutoRound, BitNet (ternary), compressed-tensors (weights and inputs), FP8, ModelOpt (NVFP4, MXFP8, input scales), MXFP4, gpt-oss MXFP4, AMD Quark, and Humming's native schema.

### Python API

- `HummingLayer`: `load_from_unquantized` or a schema loader, then `.transform()`, then `__call__`.
- Ops: `humming_gemm`, `process_input`, `quant_weight`, `dequant_weight`, `pack_weight`, `unpack_weight`, `repack_weight`, `convert_weight`, `process_mxfp4_w4a8_weight`, `tops_bench`.
- vLLM flags: `--linear-backend humming`, `--moe-backend humming`, `--quantization humming`.

---

## DeepGEMM (deepseek-ai/DeepGEMM)

A JIT-compiled CUDA library for DeepSeek's training and inference kernels, SM90/SM100.
Layout suffixes: `nt` means A is row-major and B is transposed (the usual "x @ W^T"). Each GEMM comes in `nt`, `nn`, `tn` and `tt` variants.

### GEMMs

| Function | What it does |
|---|---|
| `fp8_fp4_gemm_{nt,nn,tn,tt}` | Dense FP8 × FP8 or FP8 × FP4 GEMM with fine-grained block scales. `recipe` sets scale granularity, default 1×128 for activations and 128×128 for weights. Options: `recipe_a/b`, `alpha`, `compiled_dims`, `disable_ue8m0_cast`, optional C accumulator |
| `bf16_gemm_{nt,nn,tn,tt}` | Dense BF16 GEMM |
| `cublaslt_gemm_{nt,nn,tn,tt}`, `cublaslt_nvfp4_gemm_nt` | cuBLASLt wrappers, for comparison and fallback |
| `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous` | MoE grouped along M: experts' tokens concatenated, with fixed N and K (prefill and training forward) |
| `m_grouped_fp8_fp4_gemm_nt_masked` | MoE grouped along M with a fixed capacity per expert plus `masked_m` (decode with CUDA graphs; takes DeepEP low-latency output) |
| `m_grouped_bf16_gemm_{nt,nn}_contiguous`, `m_grouped_bf16_gemm_nt_masked` | BF16 versions |
| `k_grouped_fp8_gemm_{tn,nt}_contiguous`, `k_grouped_fp4_gemm_nt_contiguous`, `k_grouped_bf16_gemm_tn_contiguous` | Grouped along K: the MoE **weight-gradient** backward pass (default recipe 1×1×128, optional psum layout) |
| `einsum`, `fp8_einsum` | Fixed einsum patterns used by MLA: `bhd,hdr->bhr`, `bhr,hdr->bhd`, `bhd,bhr->hdr`, `bmk,bnk->mn` |
| `batched_syrk`, `batched_symm` | Symmetric products (X·Xᵀ, and symmetric × general): Muon optimizer's Newton–Schulz step |
| `tf32_hc_prenorm_gemm` | TF32 GEMM fused with a sum of squares: the pre-norm GEMM in mHC (manifold-constrained hyper-connections) |
| Legacy (Triton) | `m_grouped_bf16_gemm_{nt,nn}_contiguous_tl`, `a_fused_*`, `b_fused_*` k-/m-grouped BF16 GEMMs |

### Attention / indexer (DeepSeek V3.2 / V4 sparse attention)

| Function | What it does |
|---|---|
| `fp8_fp4_mqa_logits` | Lightning-indexer scores, `relu(q·k) * weight` summed over heads. Each query covers a ragged KV span; FP8 or FP4 (MXFP4/MXFP8 with UE8M0 scales) |
| `fp8_fp4_paged_mqa_logits` | The same over a paged KV cache (decode) |
| `fp8_fp4_sparse_mqa_logits`, `fp8_fp4_paged_sparse_mqa_logits` | Sparse-indexer versions that score only selected KV blocks |
| `get_*_mqa_logits_metadata` | Scheduling metadata for the above |

### Fused MoE and routing

| Function | What it does |
|---|---|
| `fp8_fp4_mega_moe`, `bf16_mega_moe` | One kernel for expert-parallel dispatch → L1 GEMM → SwiGLU → L2 GEMM → combine, over symmetric memory. Supports shared experts, `activation_clamp`, `fast_math`, recipe (1,1,32) |
| `transform_weights_for_mega_moe`, `get_symm_buffer_for_mega_moe`, `localize` | Weight layout, communication buffer, and locality-domain placement |
| `bf16_mega_gate` | Fused router: scoring (`sqrtsoftplus`), top-k, shared-as-routed experts, routed scaling factor, bias and mask |

### Utilities and options

- **Layout helpers:** UE8M0 scale packing (4 per int32), TMA alignment, MN-major transforms, `transform_sf_into_required_layout`, contiguous-layout alignment getters and setters.
- **Runtime knobs:** `set_num_sms`, `set_tc_util`, `set_pdl`, `use_deterministic_algorithms`, `set_ignore_compile_dims`, `set_block_size_multiple_of`.
- **Python cast helpers:** `per_token_cast_to_fp8`, `per_channel_cast_to_fp8`, `per_block_cast_to_fp8`, `per_token_cast_to_fp4`, and their inverses.
- **Environment variables:** `DG_JIT_*` (cache directory, NVCC choice, PTX/SASS dumps, spill checks, line info), `DG_PRINT_CONFIGS`, debug asserts, cuBLASLt workspace options.
- **Epilogues:** identity and alpha scaling.

---

## TPU status of every item (2026-11-15)

Sources are the current code of tokamax, tpu-inference, qwix and JAX. Only "measured" items have been benchmarked here; the rest come from reading the code.

| Item | TPU equivalent | Status |
|---|---|---|
| FP8 / INT8 W8A8 dense, group scales | tpu-inference `quantized_matmul` (tuned int8×int8 on v6; fp8×fp8, fp8×fp4 on v7) | Covered |
| Dense BF16 GEMM | XLA | Covered |
| MoE grouped GEMM, symmetric quantized (`m_grouped_*`, Humming `grouped_*`) | tokamax `gmm_v2`, `ragged_dot` | Covered |
| MoE weight-gradient GEMM (`k_grouped_*`) | tokamax `tgmm_v2` | Covered |
| Lightning indexer / MQA logits | tokamax `lightning_indexer`, tpu-inference `deepseek_v4/indexer` | Covered |
| mHC (`tf32_hc_prenorm_gemm`) | tokamax `mhc`, `mhc_lite`; tpu-inference `deepseek_v4/mhc` | Covered |
| Mega MoE (inference) | tokamax `fused_moe`, tpu-inference `fused_moe` | Covered (training is our mok-tpu project) |
| Fused router (`mega_gate`) | tokamax `router_topk`, tpu-inference `router_topk` | Covered |
| **Zero-point weights (AWQ/GPTQ), dense** | Every kernel raises `zero_point is not supported`; XLA fallback | **Open, measured at 0.38–0.79× bf16 on v6e** |
| **Zero-point weights, MoE** | `gmm_v2` / `ragged_dot` raise the same error | **Open** |
| **Packed 1/2/3/5/6/7-bit weights** | Qwix stores them in int4/int8 containers | **Open** |
| **W4A8** | No int4-weight tunings anywhere | **Open** |
| **W4A4 (int4 × int4)** | Not exposed, though v5e/v6e MXUs do int4×int4 natively (4× bf16) | **Open** |
| **FP4 weights on v5e/v6e** | fp4 tuned only for v7 | **Mostly open** |
| **Arbitrary small-FP weights** (fp5/fp6/fp7, zero-exponent formats) | None | **Open** |
| **Fine-grained FP8 training GEMMs** (1×128 / 128×128 scales, forward + dgrad + wgrad) | tokamax `fused_fp8_matmul` uses one scale per row | **Open** (v7x, training) |
| **Symmetric products for Muon** (`syrk`/`symm`) | None | Open, niche |
| Activation pre-pass with fused quant + Hadamard | Partly inside tpu-inference's blockwise kernel (dynamic activation quant); no Hadamard | Partly open |
