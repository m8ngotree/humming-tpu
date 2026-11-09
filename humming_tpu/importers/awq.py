"""AWQ (AutoAWQ GEMM format) <-> standard format.

AWQ checkpoint tensors for a linear with in_features K, out_features N, group g:
    qweight  int32 [K, N // 8]       8 uint4 along N per int32
    qzeros   int32 [K // g, N // 8]  same packing
    scales   fp16  [K // g, N]
Nibble j of each int32 holds column 8*c + AWQ_ORDER[j].
AWQ is already [K, N], so no transpose is needed (GPTQ is the one that packs along K).
"""

from __future__ import annotations

import numpy as np

from humming_tpu.config import WeightConfig

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def unpack_awq(packed: np.ndarray) -> np.ndarray:
    """int32 [R, C // 8] -> int32 [R, C] of values 0..15 in natural column order."""
    packed = packed.astype(np.uint32)
    rows, cols8 = packed.shape
    out = np.empty((rows, cols8, 8), dtype=np.int32)
    for nibble, col in enumerate(AWQ_ORDER):
        out[:, :, col] = (packed >> (4 * nibble)) & 0xF
    return out.reshape(rows, cols8 * 8)


def pack_awq(values: np.ndarray) -> np.ndarray:
    """Inverse of unpack_awq (used to fabricate AWQ checkpoints in tests)."""
    rows, cols = values.shape
    v = values.astype(np.uint32).reshape(rows, cols // 8, 8)
    packed = np.zeros((rows, cols // 8), dtype=np.uint32)
    for nibble, col in enumerate(AWQ_ORDER):
        packed |= (v[:, :, col] & 0xF) << (4 * nibble)
    return packed.view(np.int32)


def from_awq(qweight, qzeros, scales, group_size: int, scale_dtype: str = "bfloat16"):
    """Returns (cfg, q int32 [K,N], scales [G,N], zeros int32 [G,N])."""
    q = unpack_awq(np.asarray(qweight))
    zeros = unpack_awq(np.asarray(qzeros))
    k, n = q.shape
    cfg = WeightConfig(
        shape_k=k,
        shape_n=n,
        b_dtype="uint4",
        weight_scale_group_size=group_size,
        has_zero_point=True,
        bs_dtype=scale_dtype,
    )
    scales = np.asarray(scales).astype(np.float32)
    assert scales.shape == zeros.shape == (cfg.num_groups, n)
    return cfg, q, scales, zeros
