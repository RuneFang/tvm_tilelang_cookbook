"""
第 7 章 §7.7 · 观察 LayoutInference 前后 layout_map 的填充

- 来源：cookbook 07_layout_and_fragment.md §7.7
- 目的：dump 阶段一后的 TIR（无 layout）与 lower() 之后的 device_mod（有 layout_map）
- 跑法：`python 03_before_after_layout_inference.py > out.txt`
- 期望输出：AFTER 版本里能 grep 到 "layout_map" 或 "MakeSwizzledLayout" 字样
- 坑：某些版本 LayoutInference 结果不会直接以 layout_map dict 字面量存在 script 输出里，
     可能被展开成下标表达式（例如出现 XOR / FloorMod 混合）
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, block_N),
                  T.ceildiv(M, block_M),
                  threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def main():
    cfg = dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
    pf = matmul.get_tir(**cfg)
    print("=" * 60)
    print("BEFORE LayoutInference (get_tir)")
    print("=" * 60)
    print(pf.script())

    art = tilelang.lower(pf, target="cuda")
    print("=" * 60)
    print("AFTER lower() (device_mod)")
    print("=" * 60)
    print(art.device_mod.script())


if __name__ == "__main__":
    main()
