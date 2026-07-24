"""
第 1 章 §1.1 · Minimal TileLang matmul (fp16 → fp16 with fp32 accum + ReLU)

- 来源：cookbook 01_hello_tilelang.md 的第 1.1 节 quickstart 版本
- 目的：验证 TileLang 装好了；跑通"最经典"的 tile matmul + relu
- 跑法：`python 01_matmul_minimal.py`
- 期望输出：`[hello] max abs diff = <小数>   PASS`
- 坑：需要有 CUDA GPU；1024x1024 大概 <100us 就能跑完，如果卡住多半是 nvcc 找不到

upstream 参考：examples/quickstart.py（本仓库根目录）
"""

import torch
import tilelang
import tilelang.language as T


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
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)  # relu
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def main():
    M = N = K = 1024
    kernel = matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=32)

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = kernel(a, b)

    # 参考实现：fp32 累加 + relu
    ref = torch.relu((a.float() @ b.float())).to(torch.float16)

    diff = (c - ref).abs().max().item()
    ok = diff < 1.0  # fp16 累加的量级，宽松阈值
    print(f"[hello] max abs diff = {diff:.4f}   {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
