"""
第 12 章 §12.5 · T.reduce_sum — 向量求和最小 kernel

- 来源：cookbook 12_control_flow_dynamic_reduce_atomic.md §12.5
- 目的：单 block、无 pipeline 的 reduce_sum；把一个 4096 长的向量求成一个标量
- 跑法：`python 02_reduce_sum_vector.py`
- 期望输出：`PASS  ref=<..> got=<..> diff=<..>`
- 坑：
    - 单 block 只能处理 N <= block（block=1024 时 N<=1024）
    - 想跨 block 归并需要 atomic_add，另见第 12 章 §12.7
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def vec_sum(N: int, block: int):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), Out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=block) as (bx,):
            A_frag = T.alloc_fragment((N,), "float32")
            OutFrag = T.alloc_fragment((1,), "float32")
            T.copy(A, A_frag)
            T.reduce_sum(A_frag, OutFrag, dim=0, clear=True)
            T.copy(OutFrag, Out)
    return main


def main():
    N = 1024
    kernel = vec_sum(N=N, block=128)
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    out = kernel(a)
    torch.cuda.synchronize()

    ref = a.sum().item()
    got = out.item()
    diff = abs(ref - got)
    ok = diff < 1e-2
    print(f"{'PASS' if ok else 'FAIL'}  ref={ref:.6f} got={got:.6f} diff={diff:.6e}")


if __name__ == "__main__":
    main()
