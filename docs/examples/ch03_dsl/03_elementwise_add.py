"""
第 3 章 §3.11 · 练习 3：只用 T.copy 写一个 elementwise-add kernel

- 来源：cookbook 03_tilelang_dsl.md 的练习 3
- 目的：不涉及 T.gemm，最小化演示 alloc_shared + T.copy + T.Parallel 的组合
- 跑法：`python 03_elementwise_add.py`
- 期望输出：
    ok! kernel_source:
    __global__ void ...
- 坑：如果 tilelang 版本较低不认识 `A: T.Tensor(...)` 注解 hint 语法，回落到 tvm.script.tir
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B, block_N: int = 128):
    N = T.const("N")
    dtype = T.float32
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
        A_sh = T.alloc_shared((block_N,), dtype)
        B_sh = T.alloc_shared((block_N,), dtype)
        T.copy(A[bx * block_N], A_sh)
        T.copy(B[bx * block_N], B_sh)
        for i in T.Parallel(block_N):
            A_sh[i] = A_sh[i] + B_sh[i]
        T.copy(A_sh, C[bx * block_N])
    return C


def main():
    kernel = add.compile(N=4096, block_N=128)

    a = torch.randn(4096, device="cuda")
    b = torch.randn(4096, device="cuda")
    c = kernel(a, b)

    torch.testing.assert_close(c, a + b)
    print("ok! kernel_source:")
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
