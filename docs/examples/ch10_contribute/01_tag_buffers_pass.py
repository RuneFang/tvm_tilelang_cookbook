"""
第 10 章 §10.3 · 玩具 Python pass：在 PrimFunc body 外包一层 AttrStmt 标记

- 来源：cookbook 10_contribute.md §10.3.1
- 目的：不写 C++，纯 Python 实现一个改写型 pass，跑一个最小 PrimFunc 验证
- 跑法：`python 01_tag_buffers_pass.py`
- 期望输出：after 的 body 顶部多出一个 "tl.tagged" 的 attr 标记
- 坑：
    - 真正在 TileLang 主干里 pass 大都是 C++ 实现，这里只演示怎么落到 Python
    - 必须用 @functor.mutator 装饰 + PyStmtExprMutator（tilelang 的 vendored TVM 是 tirx 分支）
    - visit_xxx_ 里要调 super() 兜底递归，否则 body 里其他节点不会被访问到
"""

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.tirx import AttrStmt, PyStmtExprMutator, functor
from tvm.tirx.transform import prim_func_pass


@functor.mutator
class _TagBodyMutator(PyStmtExprMutator):
    def visit_evaluate_(self, op):
        # 这里只做"演示改写"：原样返回。真实 pass 会在这里判断节点并构造新节点。
        return op


def TagBody():
    """在 PrimFunc body 外包一层 AttrStmt 标记（玩具示例）。"""

    def pass_fn(func, mod, ctx):
        mutator = _TagBodyMutator()
        new_body = mutator.visit_stmt(func.body)
        new_body = AttrStmt(0, "tl.tagged", 1, new_body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)


@tilelang.jit
def add(A, B, block_N: int = 128):
    N = T.const("N")
    dtype = T.float32
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
        for i in T.Parallel(block_N):
            C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]
    return C


def main():
    pf = add.get_tir(N=4096, block_N=128)
    mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})

    print("===== before =====")
    print(mod.script())

    mod2 = TagBody()(mod)

    print("===== after =====")
    print(mod2.script())


if __name__ == "__main__":
    main()
