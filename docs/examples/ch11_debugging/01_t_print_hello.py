"""
第 11 章 §11.1 · T.print — GPU 内运行时打印

- 来源：cookbook 11_debugging_and_visualization.md §11.1
- 目的：最小演示 T.print 的正确用法（用 if tid==0 避免 128 份重复输出）
- 跑法：`python 01_t_print_hello.py`
- 期望输出（stdout 会看到一行）：
    msg='hello world' BlockIdx=(0,0,0), ThreadIdx=(0,0,0): 0
- 坑：
    - T.print 需要 GPU 才能真正执行
    - 不加 if tid==0 你会看到 8 份重复输出
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=8) as (bx,):
            tid = T.get_thread_binding()
            if tid == 0:
                T.print(tid, msg="hello world")
    return main


def main():
    k = kernel()
    k()  # 无参 launch


if __name__ == "__main__":
    main()
