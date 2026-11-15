"""Roofline lower bounds for a quantized matmul on one JAX TPU device.

time >= max(bytes / HBM bandwidth, flops / MXU peak), using public specs for one
JAX device. On v5e/v6e that is a whole chip. On v7x it is one TensorCore (half a
chip), because JAX lists each core as a separate device. VPU dequant cost is
deliberately not modelled: at large M, the gap between this bound and measured
time is where VPU limits show up.
"""

from __future__ import annotations

import dataclasses

from humming_tpu.config import WeightConfig


@dataclasses.dataclass(frozen=True)
class Chip:
    name: str
    hbm_gbps: float
    bf16_tflops: float
    int8_tops: float  # fp8 on v7x


CHIPS = {
    "v5e": Chip("v5e", hbm_gbps=819, bf16_tflops=197, int8_tops=394),
    "v6e": Chip("v6e", hbm_gbps=1640, bf16_tflops=918, int8_tops=1836),
    # JAX lists each v7x TensorCore as its own device, so a single-device kernel
    # runs on half a chip. "v7x" is that per-device bound; "v7x-chip" is both cores.
    "v7x": Chip("v7x", hbm_gbps=3685, bf16_tflops=1154, int8_tops=2307),
    "v7x-chip": Chip("v7x-chip", hbm_gbps=7370, bf16_tflops=2307, int8_tops=4614),
}


@dataclasses.dataclass(frozen=True)
class Bound:
    bytes: float
    flops: float
    mem_us: float
    mxu_us: float

    @property
    def us(self) -> float:
        return max(self.mem_us, self.mxu_us)


def matmul_bound(m: int, k: int, n: int, cfg: WeightConfig | None, chip: Chip) -> Bound:
    """Lower bound for y[m, n] = x[m, k] @ W[k, n]; cfg=None means a bf16 weight."""
    activation_bytes = 2 * (m * k + m * n)
    weight_bytes = cfg.weight_bytes() if cfg is not None else 2 * k * n
    total_bytes = activation_bytes + weight_bytes
    flops = 2 * m * k * n
    return Bound(
        bytes=total_bytes,
        flops=flops,
        mem_us=total_bytes / (chip.hbm_gbps * 1e9) * 1e6,
        mxu_us=flops / (chip.bf16_tflops * 1e12) * 1e6,
    )
