"""Baselines we must beat, copied in behaviour from upstream so benchmarks need
no tpu-inference install.

xla_awq: tpu-inference AWQ linear (layers/vllm/quantization/awq.py, _apply_fused,
checked 2026-11-08 at main). Weights are stored as jnp.uint4 [G, g, N], zeros
uint4 [G, N], scales [G, N]; each call does
    weight = (qweight.astype(int8) - qzeros.astype(int8)) * scales
    einsum("bd,df->bf", x, weight.reshape(K, N))
Whether XLA fuses that dequant into the dot or materializes a bf16 [K, N] in HBM
is the Phase 1 question.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def prepare_xla_awq(q, scales, zeros, group_size: int):
    k, n = q.shape
    qweight = q.astype(jnp.uint4).reshape(k // group_size, group_size, n)
    return qweight, scales.astype(jnp.bfloat16), zeros.astype(jnp.uint4)


@jax.jit
def xla_awq(x, qweight, scales, qzeros):
    qzeros = jnp.expand_dims(qzeros, 1).astype(jnp.int8)
    scales = jnp.expand_dims(scales, 1)
    weight = (qweight.astype(jnp.int8) - qzeros) * scales
    weight = weight.reshape((-1, weight.shape[-1]))
    return jnp.einsum("bd,df->bf", x, weight)


@jax.jit
def bf16_matmul(x, w):
    return jnp.einsum("bd,df->bf", x, w)


@jax.jit
def xla_w8_channel(x, q_int8, channel_scales):
    """Symmetric int8 weight, per-output-channel scale, dequant left to XLA.

    Contrast case for xla_awq: the simplest dequant XLA could fuse into the dot.
    """
    weight = q_int8.astype(jnp.bfloat16) * channel_scales
    return jnp.einsum("bd,df->bf", x, weight)
