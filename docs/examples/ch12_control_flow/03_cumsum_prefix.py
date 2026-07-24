"""
第 12 章 §12.6 · T.cumsum — 前缀和（in-place）

- 来源：cookbook 12_control_flow_dynamic_reduce_atomic.md §12.6
- 目的：对一个 shared 里的 128 元素向量做 in-place 前缀和
- 跑法：`python 03_cumsum_prefix.py`
- 期望输出：`PASS  head=[...prefix sum head 5 elems...]`
- 坑：
    - dst.shape 必须和 src.shape 严格一致（不像 reduce 会降一维）
    - fragment 版会走 shared 中转，别惊讶
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def cumsum128():
    N = 128
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=N):
            A_s = T.alloc_shared((N,), "float32")
            T.copy(A, A_s)
            T.cumsum(A_s, A_s, dim=0)
            T.copy(A_s, B)
    return main


def main():
    N = 128
    kernel = cumsum128()
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = kernel(a)
    torch.cuda.synchronize()

    ref = a.cumsum(dim=0)
    diff = (b - ref).abs().max().item()
    ok = diff < 1e-3
    print(f"{'PASS' if ok else 'FAIL'}  head={b[:5].tolist()}  diff={diff:.6e}")


if __name__ == "__main__":
    main()
