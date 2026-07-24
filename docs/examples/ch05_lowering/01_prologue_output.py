"""
第 5 章 §5.2.4 · 打印段 A（Prologue）出口的 IR

- 来源：cookbook 05_lowering_pipeline.md §5.2.4 末尾的代码块
- 目的：直接调 CUDAPassPipelineBodyPrologue，看段 A 结束后 IR 长什么样
- 跑法：`python 01_prologue_output.py > out.txt`
- 期望输出：一整份没有 T.copy/T.gemm/T.Pipelined 的 TIR script
- 坑：CUDAPassPipelineBodyPrologue 是一个函数，直接 (mod, target) 调用即可
"""

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
from tvm.target import Target


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
    pf = matmul.get_tir(**cfg)
    if "global_symbol" not in pf.attrs:
        pf = pf.with_attr("global_symbol", "main")

    mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})
    tgt = Target("cuda")

    out = CUDAPassPipelineBodyPrologue(mod, tgt)
    print("===== IRModule after CUDAPassPipelineBodyPrologue =====")
    print(out.script())


if __name__ == "__main__":
    main()
