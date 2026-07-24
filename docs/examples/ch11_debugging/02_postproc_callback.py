"""
第 11 章 §11.2 · register_cuda_postproc_callback — 拦截 codegen 出来的 CUDA 源码

- 来源：cookbook 11_debugging_and_visualization.md §11.2
- 目的：注册一个 postproc callback，把生成的 CUDA 源码打印+改写（加一行注释）
- 跑法：`python 02_postproc_callback.py > out.txt`
- 期望输出：
    ======== GENERATED CUDA ========
    ...一大段 CUDA C++...
    然后编译后 kernel.get_kernel_source() 会打印出改写后的版本
- 坑：
    - callback 必须在 @tilelang.jit 之前注册
    - callback 返回什么字符串，nvcc 就编什么；改坏了会导致 nvcc 报错
"""

import tilelang
import tilelang.language as T
from tilelang.engine.callback import register_cuda_postproc_callback


@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, target):
    print("=" * 40, "GENERATED CUDA", "=" * 40)
    print(code)
    return "// modified by callback\n" + code


@tilelang.jit
def matmul(M: int, N: int, K: int):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 128, bx * 128])
    return main


def main():
    k = matmul(512, 512, 512)
    print("=" * 40, "AFTER callback", "=" * 40)
    print(k.get_kernel_source())


if __name__ == "__main__":
    main()
