"""
第 11 章 §11.3 · pass_diff Python API — 每个 pass 前后 IR diff

- 来源：cookbook 11_debugging_and_visualization.md §11.3
- 目的：Python 侧直接调 pass_diff，选一小段 pass 做 A/B 分析，生成 HTML 报告
- 跑法：`python 03_pass_diff_api.py`
- 期望输出：
    - 每一步 pass 的 "insertions"/"deletions" 行数
    - tmp/selected_passes.html 文件（含彩色 diff）
- 坑：
    - 环境变量 TILELANG_PASS_DIFF=html 必须在 import tilelang 之前设，本示例用 Python API 绕开
    - html_path 目录会自动创建
"""

import os
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.pass_diff import pass_diff


@tilelang.jit
def kernel(M: int, K: int):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), B: T.Tensor((M, K), "float32")):
        with T.Kernel(T.ceildiv(M, 128), threads=128) as bx:
            A_s = T.alloc_shared((128, K), "float32")
            T.copy(A[bx * 128, 0], A_s)
            for i, j in T.Parallel(128, K):
                A_s[i, j] = A_s[i, j] + 1.0
            T.copy(A_s, B[bx * 128, 0])
    return main


def main():
    pf = kernel.get_tir(M=1024, K=32)
    if "global_symbol" not in pf.attrs:
        pf = pf.with_attr("global_symbol", "main")

    os.makedirs("tmp", exist_ok=True)

    steps = pass_diff(
        pf,
        [
            ("BindTarget", tvm.tirx.transform.BindTarget(tvm.target.Target("cuda"))),
            ("Simplify", tilelang.transform.Simplify()),
            ("ThreadSync-shared", tilelang.transform.ThreadSync("shared")),
        ],
        mode="both",
        context=5,
        html_path="tmp/selected_passes.html",
    )

    for step in steps:
        print(f"{step['name']:40s}  +{step['insertions']:>4}  -{step['deletions']:>4}  "
              f"changed={step['changed']}")

    print("\nHTML report → tmp/selected_passes.html")


if __name__ == "__main__":
    main()
