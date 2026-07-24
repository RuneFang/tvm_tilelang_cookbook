"""
附录 F §F.2 · CuTeDSL 后端切换 —— 一份 matmul 同时给两条后端跑

- 来源：cookbook F_eager_and_cutedsl.md §F.2
- 目的：同一个 PrimFunc，通过 execution_backend 参数分别走默认 (CUDA C via nvcc) 和 CuTeDSL 后端
- 跑法：
    直接跑（脚本内已显式指定 execution_backend）：`python 02_cutedsl_backend.py`
    如想改默认后端可设环境变量：`TILELANG_EXECUTION_BACKEND=cutedsl python 02_cutedsl_backend.py`
- 期望输出：默认后端始终能通；CuTeDSL 后端根据你 GPU/环境不同要么通要么给出明确报错
- 坑：
    - CuTeDSL 依赖 `import cutlass` / `import cutlass.cute`，需要额外 pip install
    - alloc_global 等 API 目前不支持，某些 kernel 会 fallback；本示例用最基础的 matmul 尽量避开
"""

import os
import torch
import tilelang
import tilelang.language as T


def build(backend: str):
    @tilelang.jit(out_idx=[-1], execution_backend=backend)
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
                for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                    T.copy(A[by * 128, k * 32], A_shared)
                    T.copy(B[k * 32, bx * 128], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * 128, bx * 128])
        return kernel

    return matmul


def try_backend(name: str):
    print("=" * 60)
    print(f"execution_backend = {name!r}")
    print("=" * 60)
    try:
        m = build(name)(512, 512, 512)
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        c = m(a, b)
        torch.cuda.synchronize()
        ref = (a.float() @ b.float()).to(torch.float16)
        diff = (c - ref).abs().max().item()
        print(f"  ok, diff = {diff:.4f}")
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")


def main():
    print("TILELANG_EXECUTION_BACKEND =", os.environ.get("TILELANG_EXECUTION_BACKEND", "<unset>"))

    try_backend("auto")     # 默认路径（cuda / cython / tvm_ffi 里第一个可用的）
    try_backend("cutedsl")  # CuTeDSL 分支


if __name__ == "__main__":
    main()
