"""
第 1 章 §1.3 · 观察编译流水线每个阶段的中间产物

- 来源：cookbook 01_hello_tilelang.md 的 "想自己看一眼？" 代码块
- 目的：一次性 dump 出阶段一（TIR）、阶段二（lowered device_mod）、阶段三/五（CUDA + host 源码）
- 跑法：`python 02_look_around_pipeline.py > out.txt`（输出很长，建议重定向）
- 期望输出：
    ===== TIR (after parse, before any pass) =====
    ...TIR script...
    ===== device_mod (after lowering pipeline) =====
    ...lowered TIR...
    ===== kernel_source (CUDA C++) =====
    __global__ void matmul_kernel_0(...)
    ...
- 坑：get_tir / lower / compile 三条路径独立执行——每次都会重新走一遍前置步骤
"""

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
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def main():
    cfg = dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)

    # 阶段一：解析出来的 TIR（未 lower）
    prim_func = matmul.get_tir(**cfg)
    print("===== TIR (after parse, before any pass) =====")
    print(prim_func.script())

    # 阶段二：lower 后拿到 CompiledArtifact
    artifact = tilelang.lower(prim_func, target="cuda")
    print("===== device_mod (after lowering pipeline) =====")
    print(artifact.device_mod.script())

    # 阶段三：codegen 输出的 CUDA 源码
    print("===== kernel_source (CUDA C++) =====")
    print(artifact.kernel_source)

    # 完整走完 6 个阶段，拿到可调用 kernel
    kernel = matmul.compile(**cfg)
    print("===== kernel.get_kernel_source() =====")
    print(kernel.get_kernel_source())
    print("===== kernel.get_host_source() =====")
    print(kernel.get_host_source())


if __name__ == "__main__":
    main()
