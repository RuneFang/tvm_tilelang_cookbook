"""
第 5 章 §5.6 · 手动串 pass 看 InjectSoftwarePipeline 前后

- 来源：cookbook 05_lowering_pipeline.md §5.6 的模板
- 目的：手动逐 pass 跑到 InjectSoftwarePipeline 前后各打印一次 IR，做肉眼 diff
- 跑法：`python 02_step_by_step_before_after.py > out.txt`
- 期望输出：
    ===== BEFORE InjectSoftwarePipeline =====
    ...IR with single loop + software_pipeline annotations...
    ===== AFTER  InjectSoftwarePipeline =====
    ...IR with prologue/steady/epilogue three-part loop...
- 坑：不同 tilelang / tvm 版本 pass 命名可能变，捕获 AttributeError 会跳过；此脚本捕获所有异常并 print 出来方便排查
"""

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tirx
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

    # 段 A 前半：把 IR 推进到 PipelinePlanning 之后、InjectSoftwarePipeline 之前
    steps = [
        ("BindTarget", lambda m: tirx.transform.BindTarget(tgt)(m)),
        ("MaterializeKernelLaunch",
            lambda m: tilelang.transform.MaterializeKernelLaunch()(m)),
        ("AddWrapperForSingleBufStore",
            lambda m: tilelang.transform.AddWrapperForSingleBufStore()(m)),
        ("LegalizeNegativeIndex",
            lambda m: tilelang.transform.LegalizeNegativeIndex()(m)),
        ("InjectAssumes",
            lambda m: tilelang.transform.InjectAssumes()(m)),
        ("Simplify", lambda m: tilelang.transform.Simplify()(m)),
        ("LayoutReducer",
            lambda m: tilelang.transform.LayoutReducer()(m)),
        ("ProducerConsumerWarpSpecialized",
            lambda m: tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(m)),
        ("LowerBlackwell2SM",
            lambda m: tilelang.cuda.transform.LowerBlackwell2SM()(m)),
        ("IfStmtBinding",
            lambda m: tilelang.transform.IfStmtBinding()(m)),
        ("PipelinePlanning",
            lambda m: tilelang.transform.PipelinePlanning()(m)),
    ]
    for name, step in steps:
        try:
            mod = step(mod)
        except Exception as e:
            print(f"[SKIP] {name} raised: {e}")

    print("===== BEFORE InjectSoftwarePipeline =====")
    print(mod.script())

    try:
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    except Exception as e:
        print(f"[ERROR] InjectSoftwarePipeline failed: {e}")
        return

    print("===== AFTER  InjectSoftwarePipeline =====")
    print(mod.script())


if __name__ == "__main__":
    main()
