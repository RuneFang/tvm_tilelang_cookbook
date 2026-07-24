"""
第 3 章 §3.11 · 练习 1：观察 num_stages 变化对 pipeline 注解的影响

- 来源：cookbook 03_tilelang_dsl.md 的练习 1
- 目的：改 num_stages=1 vs num_stages=3，看 TIR 里 software_pipeline_stage 注解如何变化
- 跑法：`python 01_pipeline_annotations.py`
- 期望输出：两份 TIR script，可以肉眼 diff 出 software_pipeline_stage / software_pipeline_order 差异
- 坑：num_stages=1 时不会分配额外 buffer，但注解依然会写入
"""

import tilelang
import tilelang.language as T


def build(num_stages: int):
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
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return matmul


def main():
    cfg = dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)

    for ns in [1, 3]:
        print("=" * 60)
        print(f"num_stages = {ns}")
        print("=" * 60)
        pf = build(ns).get_tir(**cfg)
        print(pf.script())


if __name__ == "__main__":
    main()
