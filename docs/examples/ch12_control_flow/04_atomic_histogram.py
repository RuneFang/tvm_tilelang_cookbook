"""
第 12 章 §12.7 · T.atomic_add — 直方图 histogram

- 来源：cookbook 12_control_flow_dynamic_reduce_atomic.md §12.7
- 目的：最经典的 atomic 场景：多个 CTA 同时更新同一份 histogram bin
- 跑法：`python 04_atomic_histogram.py`
- 期望输出：`PASS  first few bins: [...]`
- 坑：
    - Out 必须先清零（此处用 torch.zeros 初始化）
    - 用 memory_order 默认（None 走硬件默认，通常 relaxed，快但对跨 CTA 顺序无保证）
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def histogram(N: int, num_bins: int, block: int = 256):
    @T.prim_func
    def main(X: T.Tensor((N,), "int32"), Out: T.Tensor((num_bins,), "int32")):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            tx = T.get_thread_binding()
            i = bx * block + tx
            if i < N:
                b = X[i]
                T.atomic_add(Out[b], 1)
    return main


def main():
    N = 1 << 16
    num_bins = 32
    kernel = histogram(N=N, num_bins=num_bins)

    torch.manual_seed(0)
    x = torch.randint(0, num_bins, (N,), device="cuda", dtype=torch.int32)
    out = torch.zeros(num_bins, device="cuda", dtype=torch.int32)
    kernel(x, out)
    torch.cuda.synchronize()

    ref = torch.bincount(x.long(), minlength=num_bins).to(torch.int32)
    diff = (out - ref).abs().max().item()
    ok = diff == 0
    print(f"{'PASS' if ok else 'FAIL'}  first few bins: {out[:5].tolist()}  ref: {ref[:5].tolist()}")


if __name__ == "__main__":
    main()
