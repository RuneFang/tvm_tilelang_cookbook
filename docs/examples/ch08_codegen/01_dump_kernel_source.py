"""
第 8 章 §8.7 · dump kernel source 观察 codegen 产物

- 来源：cookbook 08_codegen_tir_to_cuda.md §8.7
- 目的：编译一个 matmul，直接打印生成的 CUDA C++ 源码字符串
- 跑法：`python 01_dump_kernel_source.py > kernel.cu`
- 期望输出：一段可肉眼审的 CUDA C++ 源码；能看到 __shared__ 声明、mbarrier、
             __launch_bounds__、producer/consumer 分支等
- 坑：
    - out_idx 决定"哪个位置是 output tensor"；这里第 2 个参数（C）是输出
    - 如果 kernel 里 T.empty(...) 会自动 append 到 out_idx
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
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


def main():
    kernel = matmul(1024, 1024, 1024)
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
