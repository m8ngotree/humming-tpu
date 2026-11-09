"""Plain-JAX reference: quantize, dequantize, matmul. Source of truth for tests.

Quantized values are held unpacked in int32 here; storage packing is a kernel
concern (see humming_tpu/packing.py).
"""

from __future__ import annotations

import jax.numpy as jnp

from humming_tpu.config import WeightConfig


def quantize(weight: jnp.ndarray, cfg: WeightConfig):
    """Round-to-nearest group quantization of a [K, N] float weight.

    Returns (q int32 [K, N], scales [G, N] bs_dtype, zeros int32 [G, N] or None).
    """
    k, n = weight.shape
    assert (k, n) == (cfg.shape_k, cfg.shape_n)
    w = weight.astype(jnp.float32).reshape(cfg.num_groups, cfg.group_size, n)

    if cfg.has_zero_point:
        w_min = jnp.minimum(w.min(axis=1), 0.0)
        w_max = jnp.maximum(w.max(axis=1), 0.0)
        scales = (w_max - w_min) / cfg.qmax
        scales = jnp.where(scales == 0, 1.0, scales)
        zeros = jnp.clip(jnp.round(-w_min / scales), cfg.qmin, cfg.qmax)
    else:
        scales = jnp.abs(w).max(axis=1) / cfg.qmax
        scales = jnp.where(scales == 0, 1.0, scales)
        zeros = jnp.zeros_like(scales)

    # Quantize against the scale as it will actually be stored.
    scales = scales.astype(cfg.scale_jnp_dtype)
    s = scales.astype(jnp.float32)[:, None, :]
    q = jnp.clip(jnp.round(w / s + zeros[:, None, :]), cfg.qmin, cfg.qmax)

    q = q.reshape(k, n).astype(jnp.int32)
    zeros = zeros.astype(jnp.int32) if cfg.has_zero_point else None
    return q, scales, zeros


def dequantize(q, scales, zeros, cfg: WeightConfig) -> jnp.ndarray:
    """[K, N] float32 weight from its quantized form."""
    k, n = q.shape
    qg = q.astype(jnp.float32).reshape(cfg.num_groups, cfg.group_size, n)
    if zeros is not None:
        qg = qg - zeros.astype(jnp.float32)[:, None, :]
    w = qg * scales.astype(jnp.float32)[:, None, :]
    return w.reshape(k, n)


def matmul(x, q, scales, zeros, cfg: WeightConfig) -> jnp.ndarray:
    """fp32-accurate y = x @ dequant(W), returned in c_dtype."""
    w = dequantize(q, scales, zeros, cfg)
    y = jnp.dot(x.astype(jnp.float32), w, precision="highest")
    return y.astype(cfg.c_dtype)
