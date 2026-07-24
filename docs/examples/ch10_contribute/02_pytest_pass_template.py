"""
第 10 章 §10.6 · 一个可以直接跑的 pytest 单元测试模板

- 来源：cookbook 10_contribute.md §10.6
- 目的：演示怎么给自己的 pass 写测试；testing/python/transform/ 里的真实测试也是这个骨架
- 跑法：
    直接跑（打印 pass 输出）：`python 02_pytest_pass_template.py`
    以 pytest 跑：`pytest -xvs 02_pytest_pass_template.py`
- 期望输出：`pytest` 版会显示 PASSED；直接跑会打印 before/after 两版 script
- 坑：
    - pass 用函数式 prim_func_pass(pass_fn, opt_level=0) 包装（tilelang 代码库的惯用写法）
    - 需要 IRModule 里的 PrimFunc 有 global_symbol 属性才能被断言正常拿到
"""

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm.tirx.transform import prim_func_pass


def _identity_pass():
    """一个什么都不做的 pass，作为最小示例。"""

    def pass_fn(func, mod, ctx):
        return func

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


def _make_input():
    pf = add.get_tir(N=4096, block_N=128)
    return tvm.IRModule({pf.attrs["global_symbol"]: pf})


def test_identity_pass_preserves_body():
    mod = _identity_pass()(_make_input())
    name = list(mod.functions.keys())[0]
    assert mod[name].body is not None


if __name__ == "__main__":
    print("===== before =====")
    print(_make_input().script())
    mod = _identity_pass()(_make_input())
    print("===== after =====")
    print(mod.script())

    test_identity_pass_preserves_body()
    print("PASS (test_identity_pass_preserves_body)")
