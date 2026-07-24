"""
第 13 章 §13.5 · Warp Specialization：T.ws(0) / T.ws(1) 手写 producer-consumer

- 来源：cookbook 13_cluster_tma_hopper.md §13.5
- 目的：不用 T.Pipelined 自动 WS，而是手写 producer 分支与 consumer 分支，
       让读者看到 mbarrier 手动同步是什么感觉
- 跑法：`python 02_warp_specialize_manual.py`
- 期望输出：编译成功；打印 kernel source 里能看到 mbarrier 相关代码
- 硬件门槛：不做数值验证；仅编译并 dump 源码。若目标 arch < SM90 会自动 fallback
- 坑：
    - T.ws(0) = tid<128, T.ws(1) = 128<=tid<256；threads 至少 256 才有意义
    - 本示例是"骨架"级别的，真实生产 kernel 用 T.Pipelined 自动生成的 WS 更稳
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def ws_demo(M: int = 128, K: int = 512, block_K: int = 32):
    dtype = "float16"

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), Out: T.Tensor((M,), "float32")):
        with T.Kernel(1, threads=256) as (bx,):
            A_shared = T.alloc_shared((M, block_K), dtype)
            partial = T.alloc_fragment((M,), "float32")
            T.clear(partial)

            for k in T.serial(K // block_K):
                # producer：仅让 tid<128 的 warpgroup 负责 copy
                with T.ws(0):
                    T.copy(A[0, k * block_K], A_shared)

                # consumer：仅让 128<=tid<256 的 warpgroup 累加
                with T.ws(1):
                    for i, j in T.Parallel(M, block_K):
                        partial[i] = partial[i] + T.cast(A_shared[i, j], "float32")

            T.copy(partial, Out)
    return main


def main():
    kernel = ws_demo()
    src = kernel.get_kernel_source()
    print(src)
    # 一个"WS 是否真的落地"的简单硬签名：
    if "mbarrier" in src or "wg_wait" in src or "wg_arrive" in src.lower():
        print("[hint] 生成的 CUDA 里可以看到 mbarrier / warpgroup 同步原语")
    else:
        print("[hint] 未看到 WS 硬件同步原语 —— 你的 GPU 可能不是 SM90+，编译器 fallback 了")


if __name__ == "__main__":
    main()
