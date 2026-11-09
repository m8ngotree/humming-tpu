"""CPU tests: Pallas kernels run in interpret mode against the fp32 reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from humming_tpu import baselines, reference
from humming_tpu.config import WeightConfig
from humming_tpu.importers import awq
from humming_tpu.kernels import wna16


def _make(k, n, m, b_dtype, group, seed=0):
    cfg = WeightConfig(shape_k=k, shape_n=n, b_dtype=b_dtype,
                       weight_scale_group_size=group, has_zero_point=b_dtype.startswith("uint"))
    kw, kx = jax.random.split(jax.random.key(seed))
    w = jax.random.normal(kw, (k, n), jnp.float32) * 0.02
    x = jax.random.normal(kx, (m, k), jnp.float32).astype(jnp.bfloat16)
    return cfg, w, x, *reference.quantize(w, cfg)


def _rel_err(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("b_dtype", ["uint4", "int4", "uint8", "int8"])
def test_reference_quant_error_is_small(b_dtype):
    cfg, w, _, q, s, z = _make(512, 256, 1, b_dtype, 128)
    assert int(q.min()) >= cfg.qmin and int(q.max()) <= cfg.qmax
    tol = 0.12 if cfg.bits == 4 else 0.01
    assert _rel_err(reference.dequantize(q, s, z, cfg), w) < tol


@pytest.mark.parametrize("m", [1, 7, 40])
@pytest.mark.parametrize("b_dtype,group", [("uint4", 128), ("int4", 128), ("uint8", 256), ("uint4", 0)])
def test_kernel_matches_reference(m, b_dtype, group):
    k, n = 1024, 512
    cfg, _, x, q, s, z = _make(k, n, m, b_dtype, group)
    kq, ks, kz = wna16.prepare(q, s, z, cfg)
    tk = max(512, cfg.group_size)  # group=0 means one group spanning all of K
    y = wna16.wna16_matmul(x, kq, ks, kz, group=cfg.group_size, tn=256, tk=tk, interpret=True)
    ref = reference.matmul(x, q, s, z, cfg)
    assert y.shape == (m, n) and y.dtype == jnp.bfloat16
    assert _rel_err(y, ref) < 1e-2


def test_xla_awq_baseline_matches_reference():
    cfg, _, x, q, s, z = _make(1024, 512, 8, "uint4", 128)
    y = baselines.xla_awq(x, *baselines.prepare_xla_awq(q, s, z, 128))
    assert _rel_err(y, reference.matmul(x, q, s, z, cfg)) < 1e-2


def test_awq_import_roundtrip():
    cfg, _, x, q, s, z = _make(512, 256, 4, "uint4", 128)
    qweight = awq.pack_awq(np.asarray(q))
    qzeros = awq.pack_awq(np.asarray(z))
    assert qweight.dtype == np.int32 and qweight.shape == (512, 32)
    cfg2, q2, s2, z2 = awq.from_awq(qweight, qzeros, np.asarray(s.astype(jnp.float16)), 128)
    np.testing.assert_array_equal(q2, np.asarray(q))
    np.testing.assert_array_equal(z2, np.asarray(z))
    assert cfg2 == cfg
