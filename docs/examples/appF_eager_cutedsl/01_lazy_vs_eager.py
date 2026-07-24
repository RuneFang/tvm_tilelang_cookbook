"""
附录 F §F.1 · Lazy vs Eager JIT 对比

- 来源：cookbook F_eager_and_cutedsl.md §F.1
- 目的：把 lazy 版 matmul 和 eager 版 add 各写一个，一眼看出两种写法的差别
- 跑法：`python 01_lazy_vs_eager.py`
- 期望输出：
    [lazy]  compile mm(1024, 1024, 1024) ...
            call mm(a, b) diff = ...  PASS
    [eager] call add(a, b) diff = ... PASS
- 坑：eager 模式的具体 API 会随版本演进，如果 T.empty_like / OutTensor 报错说明版本不支持
"""

import torch
import tilelang
import tilelang.language as T


# ─────────── lazy 模式：返回 PrimFunc、外层是工厂 ───────────

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 128, bx * 128])
    return kernel  # ← 关键：显式 return PrimFunc


# ─────────── eager 模式：直接在函数体里操作 tensor，没有 return kernel ───────────

@tilelang.jit
def add_eager(A, B):
    # 注意：没有 @T.prim_func，也不 return kernel
    N = A.shape[0]
    C = T.empty_like(A)
    with T.Kernel(T.ceildiv(N, 128), threads=128) as (bx,):
        offset = bx * 128
        for i in T.Parallel(128):
            if offset + i < N:
                C[offset + i] = A[offset + i] + B[offset + i]
    return C  # ← 关键：return 的是输出 tensor


def main():
    # lazy
    mm = matmul(1024, 1024, 1024)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = mm(a, b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.float16)
    print(f"[lazy]  matmul(1024x1024)  diff = {(c - ref).abs().max():.4f}")

    # eager
    try:
        a1 = torch.randn(2048, device="cuda", dtype=torch.float16)
        b1 = torch.randn(2048, device="cuda", dtype=torch.float16)
        c1 = add_eager(a1, b1)
        torch.cuda.synchronize()
        print(f"[eager] add(2048)         diff = {(c1 - (a1 + b1)).abs().max():.4f}")
    except Exception as e:
        print(f"[eager] skipped: {e}")


if __name__ == "__main__":
    main()
