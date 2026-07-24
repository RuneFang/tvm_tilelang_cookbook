"""
第 12 章 §12.4 · 动态形状 T.dynamic("m") 版 GEMM

- 来源：cookbook 12_control_flow_dynamic_reduce_atomic.md §12.4
- 目的：写一个能同时处理多种运行时形状的 GEMM，不用重编译
- 跑法：`python 01_dynamic_shape_matmul.py`
- 期望输出：三种不同 (M,N,K) 组合的数值 diff，全部 PASS
- 坑：
    - block_M/N/K 必须是编译期常量，只能对全局形状用 T.dynamic
    - T.symbolic 是旧名字，仓库仍兼容但已 deprecated；请用 T.dynamic
    - upstream 完整参考：examples/dynamic_shape/example_dynamic.py
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul_dyn(block_M: int = 128, block_N: int = 128, block_K: int = 32,
               num_stages: int = 3, threads: int = 128):
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")
    in_dtype = "float16"
    out_dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, K), in_dtype),
             B: T.Tensor((K, N), in_dtype),
             C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


def main():
    kernel = matmul_dyn()

    for M, N, K in [(256, 256, 128), (512, 384, 256), (1024, 1024, 512)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        torch.cuda.synchronize()
        ref = (a.float() @ b.float()).to(torch.float16)
        diff = (c - ref).abs().max().item()
        ok = diff < 5.0
        print(f"(M={M:4}, N={N:4}, K={K:4})  max abs diff = {diff:.4f}   "
              f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
