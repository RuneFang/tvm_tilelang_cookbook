"""
第 7 章 §7.5 · 手动 annotate_layout：强制 A_shared 走 swizzled layout

- 来源：cookbook 07_layout_and_fragment.md §7.5
- 目的：演示如何用 T.annotate_layout 手动挂 layout，并 diff 编译产物
- 跑法：`python 02_annotate_layout.py > out.txt`
- 期望输出：两份 CUDA 源码，一份用推断出来的 layout、一份用手动挂的 128B swizzle
- 坑：make_swizzled_layout 期望 buffer 有足够大小满足 swizzle 粒度；32B 太小可能落到 quarter 版
"""

import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout


def build(with_annotate: bool):
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

            if with_annotate:
                T.annotate_layout({
                    A_shared: make_swizzled_layout(A_shared),
                    B_shared: make_swizzled_layout(B_shared),
                })

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return matmul


def main():
    cfg = dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
    for annotate in [False, True]:
        print("=" * 60)
        print(f"with_annotate = {annotate}")
        print("=" * 60)
        try:
            k = build(annotate).compile(**cfg)
            print(k.get_kernel_source())
        except Exception as e:
            print(f"[SKIP] compile failed: {e}")


if __name__ == "__main__":
    main()
