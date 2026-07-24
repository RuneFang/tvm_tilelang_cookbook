"""
第 2 章 §2.2 · TIR 树把玩：手写一个 add_one PrimFunc

- 来源：cookbook 02_tvm_tir_basics.md 的 §2.2
- 目的：不写 GPU kernel，只演示"如何用 TVMScript 直接写一个 PrimFunc、打印 TIR 树"
- 跑法：`python 01_primfunc_add_one.py`
- 期望输出：
    ===== add_one.script() =====
    @T.prim_func
    def add_one(A: T.Buffer((16,), "float32"), ...):
        for i in range(16):
            B[i] = A[i] + T.float32(1)
    ===== IR tree walker output =====
    For var=i extent=16
      BufferStore B[[i]] = <Add>
        Add
          BufferLoad A[i]
          FloatImm 1.0
- 坑：这个例子不涉及 GPU，纯 CPU-mode 也能跑；用来快速理解 IR 是"一棵树"
"""

from tilelang import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def add_one(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
    for i in range(16):
        B[i] = A[i] + 1.0


def walk(node, indent=0):
    """一个玩具遍历器：打印 IR 树的每一层节点类型 + 关键字段。"""
    pad = "  " * indent

    if isinstance(node, tir.For):
        print(f"{pad}For var={node.loop_var.name_hint} extent={node.extent}")
        walk(node.body, indent + 1)
    elif isinstance(node, tir.SeqStmt):
        print(f"{pad}SeqStmt (n={len(node.seq)})")
        for s in node.seq:
            walk(s, indent + 1)
    elif isinstance(node, tir.BufferStore):
        print(f"{pad}BufferStore {node.buffer.name}[{list(node.indices)}] = <{type(node.value).__name__}>")
        walk(node.value, indent + 1)
    elif isinstance(node, tir.Add):
        print(f"{pad}Add")
        walk(node.a, indent + 1)
        walk(node.b, indent + 1)
    elif isinstance(node, tir.BufferLoad):
        print(f"{pad}BufferLoad {node.buffer.name}[{list(node.indices)}]")
    elif isinstance(node, tir.FloatImm):
        print(f"{pad}FloatImm {node.value}")
    elif isinstance(node, tir.IntImm):
        print(f"{pad}IntImm {node.value}")
    else:
        print(f"{pad}<{type(node).__name__}>")


def main():
    print("===== add_one.script() =====")
    print(add_one.script())

    print("===== IR tree walker output =====")
    walk(add_one.body)


if __name__ == "__main__":
    main()
