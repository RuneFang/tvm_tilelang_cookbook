"""
第 6 章 §6.9.5 · WS-off vs WS-on 回归测试（K-trip 不对齐场景）

- 来源：cookbook 06_pipeline_and_warp_specialize.md §6.9.5 回归测试模板
- 目的：验证 K-trip 不是 num_stages 倍数的 misaligned 场景下，WS 版和 WS-off 版结果 bit-exact
- 跑法：`python 02_wsoff_vs_ws_regression.py`
- 期望输出：`PASS`
- 坑：
    - 需要 Hopper (SM90+) 才能真启用 WS；旧卡上此测试没多大意义
    - "bit-exact" 需要 rtol=0, atol=0；若不同 tile 调度序不同、fp16 累加顺序变化会失败
"""

import torch
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
    # 关键：K-trip = K/block_K = 30，不是 num_stages=4 的倍数
    cfg = dict(M=256, N=256, K=30 * 32,
               block_M=128, block_N=128, block_K=32,
               num_stages=4)

    k_ws_off = matmul.compile(**cfg,
                              pass_configs={"tl.disable_warp_specialized": True})
    k_ws_on = matmul.compile(**cfg)

    # 1) 硬签名断言：确认 WS 真的启用了、且没走到 buggy 分支
    src = k_ws_on.get_kernel_source()
    if "mbarrier_wait_parity" not in src:
        print("[warn] WS 未启用（可能 GPU arch<SM90 或版本较旧）；跳过硬签名断言")
    else:
        assert "producer_phase_cnt[0] %" not in src, \
            "Regression: buggy provenance-lost path 复活了！"

    # 2) 数值 bit-exact 对比
    a = torch.randn(cfg["M"], cfg["K"], device="cuda", dtype=torch.float16)
    b = torch.randn(cfg["K"], cfg["N"], device="cuda", dtype=torch.float16)
    c_off = k_ws_off(a, b)
    c_on = k_ws_on(a, b)

    torch.testing.assert_close(c_on, c_off, rtol=0, atol=0)
    print("PASS")


if __name__ == "__main__":
    main()
