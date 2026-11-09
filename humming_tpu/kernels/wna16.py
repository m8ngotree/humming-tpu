"""W{4,8}A16 group-quantized matmul, v0: dequant tile in VMEM, then MXU dot.

    y[M, N] = x[M, K] @ ((q[K, N] - z[G, N]) * s[G, N])   (group-broadcast along K)

Storage ("kernel layout", produced by prepare()):
    q      jnp.uint4/int4/uint8/int8 [K, N]   XLA/Mosaic pack sub-byte types natively
    scales bf16 [G, N]
    zeros  bf16 [G, N] (exact for ints <= 256) or None
Zeros are bf16 rather than packed uint4 so a single group row can be sliced
dynamically in VMEM. Costs ~2.5% extra HBM bytes at g=128; revisit later.

Grid is (M tiles, N tiles, K tiles) with K innermost and an fp32 VMEM
accumulator. The scale/zero block covers every group for its N tile (a [G, tn]
slab), because a block of tk // g rows would break the sublane-multiple-of-8 rule.

This is the straightforward version. Known next steps (see README):
fold zero into a bias, move scaling to output space for small M, bf16 VPU math
on v6e, and custom bit packing for odd bit widths.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from humming_tpu.config import WeightConfig

_CONTAINER = {"uint4": jnp.uint4, "int4": jnp.int4, "uint8": jnp.uint8, "int8": jnp.int8}


def prepare(q, scales, zeros, cfg: WeightConfig):
    """Reference-format (int32 q/zeros) -> kernel layout."""
    if cfg.b_dtype not in _CONTAINER:
        raise NotImplementedError(f"v0 kernel stores {list(_CONTAINER)} only, got {cfg.b_dtype}")
    q = jnp.asarray(q).astype(_CONTAINER[cfg.b_dtype])
    scales = jnp.asarray(scales).astype(jnp.bfloat16)
    if zeros is not None:
        zeros = jnp.asarray(zeros).astype(jnp.bfloat16)
    return q, scales, zeros


def _kernel(x_ref, q_ref, s_ref, *rest, group: int, has_zero: bool):
    if has_zero:
        z_ref, o_ref, acc_ref = rest
    else:
        z_ref = None
        o_ref, acc_ref = rest

    k_idx = pl.program_id(2)

    @pl.when(k_idx == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    tk = q_ref.shape[0]
    groups_per_block = tk // group
    for i in range(groups_per_block):  # static unroll
        g_row = k_idx * groups_per_block + i
        w = q_ref[i * group:(i + 1) * group, :].astype(jnp.int32).astype(jnp.float32)
        if has_zero:
            w = w - z_ref[pl.ds(g_row, 1), :].astype(jnp.float32)
        w = w * s_ref[pl.ds(g_row, 1), :].astype(jnp.float32)
        x = x_ref[:, i * group:(i + 1) * group]
        acc_ref[...] += jnp.dot(x, w.astype(jnp.bfloat16), preferred_element_type=jnp.float32)

    @pl.when(k_idx == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@functools.partial(jax.jit, static_argnames=("group", "tm", "tn", "tk", "interpret"))
def wna16_matmul(x, q, scales, zeros, *, group: int, tm: int | None = None,
                 tn: int = 512, tk: int = 1024, interpret: bool = False):
    m, k = x.shape
    k2, n = q.shape
    assert k == k2, (x.shape, q.shape)
    tn = min(tn, n)
    tk = min(tk, k)
    if tm is None:
        tm = min(_round_up(m, 16), 256)
    if n % tn or k % tk or tk % group:
        raise ValueError(f"bad tiling: n={n} tn={tn} k={k} tk={tk} group={group}")

    m_pad = _round_up(m, tm)
    x = x.astype(jnp.bfloat16)
    if m_pad != m:
        x = jnp.pad(x, ((0, m_pad - m), (0, 0)))

    num_groups = k // group
    has_zero = zeros is not None
    in_specs = [
        pl.BlockSpec((tm, tk), lambda i, j, kk: (i, kk)),
        pl.BlockSpec((tk, tn), lambda i, j, kk: (kk, j)),
        pl.BlockSpec((num_groups, tn), lambda i, j, kk: (0, j)),
    ]
    operands = [x, q, scales]
    if has_zero:
        in_specs.append(pl.BlockSpec((num_groups, tn), lambda i, j, kk: (0, j)))
        operands.append(zeros)

    out = pl.pallas_call(
        functools.partial(_kernel, group=group, has_zero=has_zero),
        out_shape=jax.ShapeDtypeStruct((m_pad, n), jnp.bfloat16),
        grid=(m_pad // tm, n // tn, k // tk),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((tm, tn), lambda i, j, kk: (i, j)),
        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret,
    )(*operands)
    return out[:m]
