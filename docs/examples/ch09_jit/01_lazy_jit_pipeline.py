"""
第 9 章 §9.2 · @tilelang.jit lazy 模式完整调用链

- 来源：cookbook 09_runtime_jit_kernel_cache.md §9.2
- 目的：验证 Layer 1(JIT) → 2(Cache) → 3(JITKernel) → 4(Adapter) → 5(Runtime) 通路
- 跑法：`python 01_lazy_jit_pipeline.py`
- 期望输出：
    first  compile:   <大概 5~30 秒>
    second compile:   <毫秒级>（内存缓存命中）
    numerical check:  PASS
- 坑：第一次跑要下载 tilelang / nvcc，可能很慢；第二次跑非常快
"""

import time
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        block_M, block_N, block_K = 128, 128, 32
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def main():
    t0 = time.time()
    mm = matmul(1024, 1024, 1024)  # 第一次：会真正编译
    t1 = time.time()
    print(f"first  compile: {t1 - t0:.2f}s")

    t2 = time.time()
    mm2 = matmul(1024, 1024, 1024)  # 第二次：命中 _kernel_cache
    t3 = time.time()
    print(f"second compile: {(t3 - t2) * 1000:.3f}ms  (should be ~0)")

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    c = mm(a, b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.float16)
    diff = (c - ref).abs().max().item()
    ok = diff < 5.0  # fp16 累加宽松阈值
    print(f"numerical check: {'PASS' if ok else 'FAIL'}   (max abs diff = {diff:.4f})")


if __name__ == "__main__":
    main()
