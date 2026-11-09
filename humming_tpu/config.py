"""Quantized-weight format spec.

Field names follow Humming's LayerConfig (docs/config.md in vllm-project/humming)
so a Humming weight config maps onto this one without renaming. Only the subset
we implement is here; everything else is rejected loudly rather than ignored.

Logical layout convention (differs from Humming, matches tpu-inference / JAX):
    weight is [K, N]  (y = x @ W, x is [M, K])
    scales / zeros are [K // group, N]
Humming and PyTorch use [N, K]; importers transpose.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

_INT_BITS = {f"int{b}": b for b in range(1, 9)}
_UINT_BITS = {f"uint{b}": b for b in range(1, 9)}


@dataclasses.dataclass(frozen=True)
class WeightConfig:
    shape_k: int
    shape_n: int
    b_dtype: str = "uint4"  # weight type. uintN = asymmetric with zero point, intN = symmetric
    weight_scale_group_size: int = 128  # 0 -> per-channel
    has_zero_point: bool = True
    bs_dtype: str = "bfloat16"  # weight scale dtype
    a_dtype: str = "bfloat16"  # activation dtype fed to the MXU
    c_dtype: str = "bfloat16"  # output dtype

    def __post_init__(self):
        if self.b_dtype not in _INT_BITS and self.b_dtype not in _UINT_BITS:
            raise NotImplementedError(f"b_dtype={self.b_dtype}: only intN/uintN for now")
        if self.has_zero_point != self.b_dtype.startswith("uint"):
            raise ValueError("zero points pair with unsigned weights (uintN), symmetric with intN")
        group = self.weight_scale_group_size
        if group and self.shape_k % group:
            raise ValueError(f"shape_k={self.shape_k} not divisible by group {group}")
        if self.a_dtype != "bfloat16" or self.c_dtype != "bfloat16":
            raise NotImplementedError("only W*A16 bf16 for now")

    @property
    def bits(self) -> int:
        return _INT_BITS.get(self.b_dtype) or _UINT_BITS[self.b_dtype]

    @property
    def group_size(self) -> int:
        return self.weight_scale_group_size or self.shape_k

    @property
    def num_groups(self) -> int:
        return self.shape_k // self.group_size

    @property
    def qmin(self) -> int:
        return 0 if self.has_zero_point else -(1 << (self.bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1 if self.has_zero_point else (1 << (self.bits - 1)) - 1

    @property
    def scale_jnp_dtype(self):
        return jnp.dtype(self.bs_dtype)

    def weight_bytes(self) -> float:
        """HBM bytes for weight + scales + zeros with ideal bit packing."""
        k, n, g = self.shape_k, self.shape_n, self.num_groups
        total = k * n * self.bits / 8
        total += g * n * self.scale_jnp_dtype.itemsize
        if self.has_zero_point:
            total += g * n * self.bits / 8
        return total
