"""Measure achievable HBM read bandwidth (sum over a large bf16 array)."""

import time

import jax
import jax.numpy as jnp


def main() -> None:
    for size_mb in (64, 256, 1024):
        x = jnp.ones((size_mb * 2**20 // 2,), jnp.bfloat16)
        f = jax.jit(lambda a: jnp.sum(a, dtype=jnp.float32))
        jax.block_until_ready(f(x))
        iters = 50
        start = time.perf_counter()
        for _ in range(iters):
            out = f(x)
        jax.block_until_ready(out)
        dt = (time.perf_counter() - start) / iters
        print(
            f"{size_mb:5d} MB read: {dt * 1e6:8.1f} us  ->  {size_mb * 2**20 / dt / 1e9:7.1f} GB/s"
        )


if __name__ == "__main__":
    main()
