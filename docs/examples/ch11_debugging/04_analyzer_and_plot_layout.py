"""
第 11 章 §11.5 + §11.6 · Analyzer 静态 roofline + plot_layout 布局可视化

- 来源：cookbook 11_debugging_and_visualization.md §11.5 §11.6
- 目的：一个脚本演示两个"看性能/看布局"的工具
- 跑法：`python 04_analyzer_and_plot_layout.py`
- 期望输出：
    Analyzer:
      total_flops           = ...
      total_global_bytes    = ...
      estimated_time (ms)   = ...
      expected_tflops       = ...   (SM<9.0 才有）
    plot_layout → ./tmp/transpose_4x4.png
- 坑：
    - Analyzer 只识别 T.gemm / T.copy；手写 elementwise 不算 FLOPs
    - Hopper (SM90) 及以上的 expected_tflops 为 None
    - plot_layout 需要 `pip install "tilelang[vis]"` 装 matplotlib
"""

import os
import tilelang
import tilelang.language as T
from tilelang.carver.arch import CUDA
from tilelang.tools import Analyzer


@tilelang.jit
def matmul(M: int, N: int, K: int, block_M: int, block_N: int, block_K: int):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


def demo_analyzer():
    tir = matmul.get_tir(M=1024, N=1024, K=1024,
                         block_M=128, block_N=128, block_K=32)
    device = CUDA("cuda")
    result = Analyzer.analysis(tir, device)
    print("Analyzer:")
    print(f"  total_flops           = {result.total_flops}")
    print(f"  total_global_bytes    = {result.total_global_bytes}")
    print(f"  estimated_time (ms)   = {result.estimated_time}")
    print(f"  expected_tflops       = {result.expected_tflops}")
    print(f"  expected_bandwidth_GBps = {result.expected_bandwidth_GBps}")


def demo_plot_layout():
    try:
        from tilelang.tools import plot_layout
        transpose = T.Layout([4, 4], lambda i, j: (j, i))
        os.makedirs("./tmp", exist_ok=True)
        plot_layout(
            transpose,
            save_directory="./tmp",
            name="transpose_4x4",
            formats="png",
            view="input",
        )
        print("plot_layout → ./tmp/transpose_4x4.png")
    except ImportError:
        print("plot_layout: 需要 `pip install \"tilelang[vis]\"` 装 matplotlib")


def main():
    demo_analyzer()
    demo_plot_layout()


if __name__ == "__main__":
    main()
