"""
第 3 章 §3.11 · 练习 2：shared vs shared.dyn 的 codegen 差异

- 来源：cookbook 03_tilelang_dsl.md 的练习 2
- 目的：对比 T.alloc_shared 默认（"shared.dyn"）和显式 "shared" 生成的 CUDA 差别
- 跑法：`python 02_shared_vs_shared_dyn.py > out.txt`
- 期望输出：两份 CUDA 源码，可肉眼 diff 看到 __shared__ 声明的不同
- 坑：不同 TileLang 版本生成的 CUDA 可能略有偏移
"""

import tilelang
import tilelang.language as T


def build(scope: str):
    @tilelang.jit
    def matmul(A, B, block_M: int, block_N: int, block_K: int):
        M, N, K = T.const("M, N, K")
        dtype = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope=scope)
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope=scope)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return matmul


def main():
    cfg = dict(M=512, N=512, K=512, block_M=128, block_N=128, block_K=32)
    for scope in ["shared.dyn", "shared"]:
        print("=" * 60)
        print(f"scope = {scope!r}")
        print("=" * 60)
        try:
            kernel = build(scope).compile(**cfg)
            print(kernel.get_kernel_source())
        except Exception as e:
            print(f"[SKIP] compile failed for scope={scope!r}: {e}")


if __name__ == "__main__":
    main()
