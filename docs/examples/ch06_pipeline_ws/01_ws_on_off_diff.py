"""
第 6 章 §6.8 · 亲手看 pipeline / WS 前后的 IR 差异

- 来源：cookbook 06_pipeline_and_warp_specialize.md §6.8
- 目的：在 WS on / off 两种 pass config 下 dump device_mod，直接肉眼 diff
- 跑法：`python 01_ws_on_off_diff.py > out.txt`
- 期望输出：两大段 lowered device_mod script；WS_ON 版会多出 mbarrier_wait_parity 等
- 坑：WS 需要 SM90+ 才会真的启用；在旧卡上 WS_ON 可能会 fallback，两版本看起来会近似
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int, num_stages: int):
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
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def main():
    cfg = dict(M=1024, N=1024, K=1024,
               block_M=128, block_N=128, block_K=32, num_stages=3)

    pf = matmul.get_tir(**cfg)

    # 1) WS OFF：只开软件流水
    art_off = tilelang.lower(
        pf,
        target="cuda",
        pass_configs={"tl.disable_warp_specialized": True},
    )
    print("=" * 60)
    print("[WS OFF] device_mod")
    print("=" * 60)
    print(art_off.device_mod.script())

    # 2) WS ON：完整 pipeline
    art_on = tilelang.lower(pf, target="cuda")
    print("=" * 60)
    print("[WS ON]  device_mod")
    print("=" * 60)
    print(art_on.device_mod.script())


if __name__ == "__main__":
    main()
