"""
第 13 章 §13.1 · Cluster + mbarrier — Rank 之间用 shared 传数据的最小 demo

- 来源：cookbook 13_cluster_tma_hopper.md §13.1 §13.4
- 目的：cluster_dims=(2,1,1) 里 rank 0 把 shared 数据推给 rank 1，rank 1 拿到后写回 global
- 跑法：`python 01_cluster_sm_to_sm.py`
- 期望输出：`PASS`
- 硬件门槛：**需要 SM90+（H100 / Hopper 系列）**，否则 T.ClusterKernel 会 assert
- 坑：
    - 只有支持 cluster 的架构才能编译 T.ClusterKernel
    - remote_barrier 的 arrive_count 编译器会按 lowering path 自动重写
"""

import sys
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def push_pull(N: int = 128):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
        with T.ClusterKernel(1, threads=N, cluster_dims=(2, 1, 1)) as (bx,):
            rank = T.block_rank_in_cluster()
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")
            s_bar = T.alloc_cluster_barrier([1])

            # rank 0：从 global 载 A → 推给 rank 1 的 s_dst
            if rank == 0:
                T.copy(A, s_src)
                T.cluster_sync()  # 保证 rank 1 已把 barrier 初始化好
                T.copy_cluster(s_src, s_dst, dst_block=1, remote_barrier=s_bar[0])
            else:
                T.cluster_sync()
                T.mbarrier_wait_parity(s_bar[0], 0)
                T.copy(s_dst, B)
    return main


def main():
    dev_props = torch.cuda.get_device_properties(0)
    if dev_props.major < 9:
        print(f"[SKIP] 当前 GPU compute capability {dev_props.major}.{dev_props.minor} < 9.0")
        sys.exit(0)

    N = 128
    k = push_pull(N=N)

    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.zeros(N, device="cuda", dtype=torch.float32)
    k(a, b)
    torch.cuda.synchronize()

    diff = (b - a).abs().max().item()
    ok = diff < 1e-6
    print(f"{'PASS' if ok else 'FAIL'}  max diff = {diff:.6e}")


if __name__ == "__main__":
    main()
