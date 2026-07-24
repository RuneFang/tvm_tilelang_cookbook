"""
第 4 章 §4.3 · 试跑 NestedLoopChecker（只读 pass 示例）

- 来源：cookbook 04_pass_system.md §4.3 的 try_visitor.py
- 目的：把一个合法的 matmul kernel 送进 NestedLoopChecker，验证 pass 能顺利跑完
- 跑法：`python 01_try_visitor.py`
- 期望输出：`passed`
- 坑：如果版本较低找不到 tilelang.analysis.nested_loop_checker，会直接 ImportError
"""

import tilelang
import tilelang.language as T
from tilelang.analysis.nested_loop_checker import NestedLoopChecker


@tilelang.jit
def good(A, B):
    M = T.const("M")
    A: T.Tensor((M, M), T.float16)
    B: T.Tensor((M, M), T.float16)
    C = T.empty((M, M), T.float16)
    with T.Kernel(1, threads=32) as bx:
        for i, j in T.Parallel(M, M):
            C[i, j] = A[i, j] + B[i, j]
    return C


def main():
    pf = good.get_tir(M=128)
    # 打上 global_symbol，pass 才能定位到函数
    if "global_symbol" not in pf.attrs:
        pf = pf.with_attr("global_symbol", "good")
    NestedLoopChecker()(pf)
    print("passed")


if __name__ == "__main__":
    main()
