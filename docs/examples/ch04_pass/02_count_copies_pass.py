"""
第 4 章 §4.7 · 自己写一个 pass：数一数 kernel 里有多少 T.copy

- 来源：cookbook 04_pass_system.md §4.7 的 count_copies.py
- 目的：练习 PyStmtExprVisitor + prim_func_pass 组合
- 跑法：`python 02_count_copies_pass.py`
- 期望输出：
    [CountCopies] main: 3 T.copy calls
- 坑：pass 必须跑在解析后（阶段一），lower 之后 tile-op 会被展开，就数不到了
"""

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tirx
from tvm.tirx import Call, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass


@tirx.functor.visitor
class _CopyCounter(PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.n = 0

    def visit_call_(self, op: Call):
        # T.copy 在解析后是 Call("tl.tileop.copy", ...) intrinsic
        if str(op.op) == "tl.tileop.copy":
            self.n += 1
        super().visit_call_(op)  # 别忘了继续下降


def CountCopies():
    counter = _CopyCounter()

    def pass_fn(func, mod, ctx):
        counter.visit_stmt(func.body)
        name = func.attrs.get("global_symbol", "<anon>")
        print(f"[CountCopies] {name}: {counter.n} T.copy calls")
        return func

    return prim_func_pass(pass_fn, opt_level=0), counter


@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    with T.Kernel(T.ceildiv(N, block_N),
                  T.ceildiv(M, block_M),
                  threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)  # +1
            T.copy(B[ko * block_K, bx * block_N], B_shared)  # +1
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])       # +1
    return C


def main():
    pf = matmul.get_tir(M=1024, N=1024, K=1024,
                        block_M=128, block_N=128, block_K=32)
    if "global_symbol" not in pf.attrs:
        pf = pf.with_attr("global_symbol", "main")
    mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})

    pass_obj, counter = CountCopies()
    pass_obj(mod)
    # 期望：3
    assert counter.n == 3, f"expected 3 T.copy calls, got {counter.n}"
    print("PASS")


if __name__ == "__main__":
    main()
