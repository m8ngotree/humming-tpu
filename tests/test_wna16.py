"""CPU tests: Pallas kernels run in interpret mode against the fp32 reference.

On a TPU VM, set HUMMING_TPU_COMPILE=1 to run the kernel through Mosaic instead.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from humming_tpu import baselines, reference, roofline
from humming_tpu.config import WeightConfig
from humming_tpu.importers import awq
from humming_tpu.kernels import wna16

INTERPRET = os.environ.get("HUMMING_TPU_COMPILE") != "1"


def _make(k, n, m, b_dtype, group, seed=0):
    cfg = WeightConfig(
        shape_k=k,
        shape_n=n,
        b_dtype=b_dtype,
        weight_scale_group_size=group,
        has_zero_point=b_dtype.startswith("uint"),
    )
    key_w, key_x = jax.random.split(jax.random.key(seed))
    w = jax.random.normal(key_w, (k, n), jnp.float32) * 0.02
    x = jax.random.normal(key_x, (m, k), jnp.float32).astype(jnp.bfloat16)
    q, s, z = reference.quantize(w, cfg)
    return cfg, w, x, q, s, z


def _rel_err(actual, expected):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("b_dtype", ["uint4", "int4", "uint8", "int8"])
def test_reference_quant_error_is_small(b_dtype):
    cfg, w, _, q, s, z = _make(512, 256, 1, b_dtype, 128)
    assert int(q.min()) >= cfg.qmin and int(q.max()) <= cfg.qmax
    tol = 0.12 if cfg.bits == 4 else 0.01
    assert _rel_err(reference.dequantize(q, s, z, cfg), w) < tol


@pytest.mark.parametrize("m", [1, 7, 40])
@pytest.mark.parametrize(
    "b_dtype,group", [("uint4", 128), ("int4", 128), ("uint8", 256), ("uint4", 0)]
)
def test_kernel_matches_reference(m, b_dtype, group):
    cfg, _, x, q, s, z = _make(1024, 512, m, b_dtype, group)
    y = wna16.wna16_matmul(
        x,
        *wna16.prepare(q, s, z, cfg),
        group=cfg.group_size,
        block_n=256,
        block_k=512,
        interpret=INTERPRET,
    )
    assert y.shape == (m, 512) and y.dtype == jnp.bfloat16
    assert _rel_err(y, reference.matmul(x, q, s, z, cfg)) < 1e-2


def test_kernel_leading_batch_dims():
    cfg, _, x, q, s, z = _make(512, 256, 6, "uint4", 128)
    y = wna16.wna16_matmul(
        x.reshape(2, 3, 512), *wna16.prepare(q, s, z, cfg), group=128, interpret=INTERPRET
    )
    assert y.shape == (2, 3, 256)
    assert _rel_err(y.reshape(6, 256), reference.matmul(x, q, s, z, cfg)) < 1e-2


def test_kernel_rejects_bad_inputs():
    cfg, _, x, q, s, z = _make(512, 256, 4, "uint4", 128)
    kq, ks, kz = wna16.prepare(q, s, z, cfg)
    with pytest.raises(ValueError, match="last dim"):
        wna16.wna16_matmul(x[:, :256], kq, ks, kz, group=128, interpret=True)
    with pytest.raises(ValueError, match="scales shape"):
        wna16.wna16_matmul(x, kq, ks[:2], kz, group=128, interpret=True)
    with pytest.raises(TypeError, match="prepare"):
        wna16.wna16_matmul(x, q, ks, kz, group=128, interpret=True)


@pytest.mark.parametrize("path", ["xla_awq", "xla_w8"])
def test_xla_baselines_match_reference(path):
    if path == "xla_awq":
        cfg, _, x, q, s, z = _make(1024, 512, 8, "uint4", 128)
        y = baselines.xla_awq(x, *baselines.prepare_xla_awq(q, s, z, 128))
    else:
        cfg, _, x, q, s, z = _make(1024, 512, 8, "int8", 0)
        y = baselines.xla_w8_channel(x, q.astype(jnp.int8), s.astype(jnp.bfloat16))
    assert _rel_err(y, reference.matmul(x, q, s, z, cfg)) < 1e-2


def test_awq_import_roundtrip():
    cfg, _, _, q, s, z = _make(512, 256, 4, "uint4", 128)
    scales_fp16 = np.asarray(s.astype(jnp.float16))
    qweight = awq.pack_awq(np.asarray(q))
    qzeros = awq.pack_awq(np.asarray(z))
    assert qweight.dtype == np.int32 and qweight.shape == (512, 32)

    cfg2, q2, s2, z2 = awq.from_awq(qweight, qzeros, scales_fp16, 128)
    assert cfg2 == cfg
    np.testing.assert_array_equal(q2, np.asarray(q))
    np.testing.assert_array_equal(z2, np.asarray(z))
    np.testing.assert_array_equal(s2, scales_fp16.astype(np.float32))


def test_roofline_decode_is_memory_bound():
    cfg = WeightConfig(shape_k=4096, shape_n=4096)
    chip = roofline.CHIPS["v6e"]
    quant = roofline.matmul_bound(1, 4096, 4096, cfg, chip)
    dense = roofline.matmul_bound(1, 4096, 4096, None, chip)
    assert quant.mem_us > quant.mxu_us
    assert 3.5 < dense.us / quant.us < 4.0  # int4 + g128 scales/zeros ~= 3.8x fewer bytes


@pytest.mark.parametrize(
    "k,group,preferred,expected",
    [
        (4096, 128, 1024, 1024),  # 8 groups per block: smallest legal at g128
        (4096, 128, 512, 1024),  # preferred too small -> smallest legal
        (4096, 128, 2048, 2048),
        (12288, 128, 4096, 4096),  # 32 groups divides 96
        (12288, 128, 3500, 3072),  # 24 groups divides 96
        (1024, 256, 512, 1024),  # only 4 groups total -> whole K
        (1024, 1024, 512, 1024),  # per-channel
    ],
)
def test_pick_block_k(k, group, preferred, expected):
    block_k = wna16._pick_block_k(k, group, preferred)
    assert block_k == expected
    assert k % block_k == 0
    assert block_k == k or (block_k // group) % 8 == 0


def test_roofline_v7x_device_is_half_a_chip():
    cfg = WeightConfig(shape_k=4096, shape_n=4096)
    core = roofline.matmul_bound(1, 4096, 4096, cfg, roofline.CHIPS["v7x"])
    chip = roofline.matmul_bound(1, 4096, 4096, cfg, roofline.CHIPS["v7x-chip"])
    assert core.us == pytest.approx(2 * chip.us, rel=0.01)
