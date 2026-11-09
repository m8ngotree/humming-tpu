"""Roofline lower bounds for a quantized matmul on one TPU chip.

time >= max(bytes / HBM_bw, flops / MXU_peak). Public per-chip specs; v7x numbers
are per chip (2 TensorCores). VPU dequant cost is deliberately NOT modelled: the
gap between this bound and measured time at large M is where VPU limits show up.
"""

from __future__ import annotations

import dataclasses

from humming_tpu.config import WeightConfig


@dataclasses.dataclass(frozen=True)
class Chip:
    name: str
    hbm_gbps: float
    bf16_tflops: float
    int8_tops: float


CHIPS = {
    "v5e": Chip("v5e", 819, 197, 394),
    "v6e": Chip("v6e", 1640, 918, 1836),
    "v7x": Chip("v7x", 7370, 2307, 4614),  # 8-bit is fp8 on v7x
}


def bound_us(m: int, cfg: WeightConfig | None, k: int, n: int, chip: Chip) -> dict:
    """Lower bound in microseconds. cfg=None means a plain bf16 weight."""
    act_bytes = m * k * 2 + m * n * 2
    w_bytes = cfg.weight_bytes() if cfg is not None else k * n * 2
    total_bytes = act_bytes + w_bytes
    flops = 2 * m * k * n
    t_mem = total_bytes / (chip.hbm_gbps * 1e9) * 1e6
    t_mxu = flops / (chip.bf16_tflops * 1e12) * 1e6
    return {"bytes": total_bytes, "t_mem_us": t_mem, "t_mxu_us": t_mxu, "bound_us": max(t_mem, t_mxu)}
