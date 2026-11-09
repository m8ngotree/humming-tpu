"""W{4,8}A16 group-quantized matmul, v0: dequantize a tile in VMEM, then MXU dot.

    y[M, N] = x[M, K] @ ((q[K, N] - z[G, N]) * s[G, N])   (s, z broadcast over each K group)

Kernel layout (produced by `prepare`):
    q       jnp.uint4 | int4 | uint8 | int8  [K, N]   sub-byte types are packed natively by XLA
    scales  bfloat16 [G, N]
    zeros   bfloat16 [G, N] or None                   exact for integers <= 256
Zeros are stored as bf16 rather than packed uint4 so a single group row can be
sliced dynamically in VMEM. That costs ~2.5% extra HBM bytes at group 128.

Grid: (M blocks, N blocks, K blocks), K innermost with an fp32 VMEM accumulator.
The scale/zero BlockSpec covers every group for its N block ([G, block_n]): a
block of block_k // group rows would usually violate the sublane-multiple-of-8 rule.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from humming_tpu.config import WeightConfig

_LANES = 128
_CONTAINERS = {"uint4": jnp.uint4, "int4": jnp.int4, "uint8": jnp.uint8, "int8": jnp.int8}
_CONTAINER_DTYPES = frozenset(jnp.dtype(d) for d in _CONTAINERS.values())


def prepare(q, scales, zeros, cfg: WeightConfig):
    """Convert reference-format tensors (int32 q / zeros) to the kernel layout."""
    if cfg.b_dtype not in _CONTAINERS:
        raise NotImplementedError(
            f"wna16 v0 supports b_dtype in {sorted(_CONTAINERS)}, got {cfg.b_dtype!r}"
        )
    q = jnp.asarray(q).astype(_CONTAINERS[cfg.b_dtype])
    scales = jnp.asarray(scales).astype(jnp.bfloat16)
    if zeros is not None:
        zeros = jnp.asarray(zeros).astype(jnp.bfloat16)
    return q, scales, zeros


def _wna16_kernel(x_ref, q_ref, s_ref, *rest, group: int, has_zero_point: bool):
    if has_zero_point:
        z_ref, o_ref, acc_ref = rest
    else:
        z_ref = None
        o_ref, acc_ref = rest

    k_idx = pl.program_id(2)

    @pl.when(k_idx == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    groups_per_block = q_ref.shape[0] // group
    for i in range(groups_per_block):  # static unroll; groups_per_block is small
        rows = slice(i * group, (i + 1) * group)
        group_row = pl.ds(k_idx * groups_per_block + i, 1)
        w = q_ref[rows, :].astype(jnp.int32).astype(jnp.float32)
        if has_zero_point:
            w = w - z_ref[group_row, :].astype(jnp.float32)
        w = w * s_ref[group_row, :].astype(jnp.float32)
        acc_ref[...] += jnp.dot(
            x_ref[:, rows], w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
        )

    @pl.when(k_idx == pl.num_programs(2) - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _pick_block(dim: int, preferred: int, align: int) -> int:
    """Largest multiple of `align` that is <= preferred and divides dim (or dim itself)."""
    if dim <= preferred:
        return dim
    for block in range(preferred - preferred % align, 0, -align):
        if dim % block == 0:
            return block
    return dim


def _validate(x, q, scales, zeros, group: int) -> None:
    k, n = q.shape
    if x.shape[-1] != k:
        raise ValueError(f"x last dim {x.shape[-1]} != weight K {k}")
    if jnp.dtype(q.dtype) not in _CONTAINER_DTYPES:
        raise TypeError(f"q dtype {q.dtype} not in {sorted(_CONTAINERS)}; use prepare()")
    if group <= 0 or k % group:
        raise ValueError(f"group {group} must be positive and divide K={k}")
    if scales.shape != (k // group, n):
        raise ValueError(f"scales shape {scales.shape} != {(k // group, n)}")
    if zeros is not None and zeros.shape != scales.shape:
        raise ValueError(f"zeros shape {zeros.shape} != scales shape {scales.shape}")
    if n % _LANES or k % _LANES:
        raise ValueError(f"K={k} and N={n} must be multiples of {_LANES}; pad the weight")


@functools.partial(jax.jit, static_argnames=("group", "block_m", "block_n", "block_k", "interpret"))
def wna16_matmul(
    x: jax.Array,
    q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array | None,
    *,
    group: int,
    block_m: int | None = None,
    block_n: int = 512,
    block_k: int = 1024,
    interpret: bool = False,
) -> jax.Array:
    """y = x @ dequant(q, scales, zeros) in bfloat16. x may have leading batch dims."""
    _validate(x, q, scales, zeros, group)
    k, n = q.shape
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, k).astype(jnp.bfloat16)
    m = x.shape[0]

    block_n = _pick_block(n, block_n, _LANES)
    block_k = _pick_block(k, max(block_k, group), group)
    if block_k % _LANES and block_k != k:
        raise ValueError(f"block_k={block_k} must be a multiple of {_LANES}")
    if block_m is None:
        block_m = min(_round_up(m, 16), 256)  # 16 = bf16 sublane packing
    m_padded = _round_up(m, block_m)
    if m_padded != m:
        x = jnp.pad(x, ((0, m_padded - m), (0, 0)))

    has_zero_point = zeros is not None
    num_groups = k // group
    scale_spec = pl.BlockSpec((num_groups, block_n), lambda i, j, kk: (0, j))
    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda i, j, kk: (i, kk)),
        pl.BlockSpec((block_k, block_n), lambda i, j, kk: (kk, j)),
        scale_spec,
    ]
    operands = [x, q, scales.astype(jnp.bfloat16)]
    if has_zero_point:
        in_specs.append(scale_spec)
        operands.append(zeros.astype(jnp.bfloat16))

    out = pl.pallas_call(
        functools.partial(_wna16_kernel, group=group, has_zero_point=has_zero_point),
        out_shape=jax.ShapeDtypeStruct((m_padded, n), jnp.bfloat16),
        grid=(m_padded // block_m, n // block_n, k // block_k),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, kk: (i, j)),
        scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(*operands)
    return out[:m].reshape(*batch_shape, n)
