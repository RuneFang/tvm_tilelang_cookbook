"""
第 7 章 §7.2 · 手写一个 Layout（转置 / row-major / swizzled）

- 来源：cookbook 07_layout_and_fragment.md §7.2
- 目的：不涉及 kernel，纯玩 Layout 对象；理解"逻辑坐标 → 物理位置"就是一个函数
- 跑法：`python 01_layout_basics.py`
- 期望输出：
    transpose(3, 5) = [5, 3]
    row_major(3, 5) = [3 * N + 5 展开]
    xor_swizzle(3, 5) = [3, 5 XOR (3 & 0x7)]
- 坑：Layout.map_forward_index 接收 list；返回值是 PrimExpr list
"""

from tilelang.layout import Layout


def main():
    # 1) 转置 layout
    transpose = Layout((16, 16), lambda i, j: [j, i])
    print("transpose(3, 5) =", transpose.map_forward_index([3, 5]))

    # 2) row-major linear layout（M, N 是常量 32）
    N = 32
    row_major = Layout((16, N), lambda i, j: [i * N + j])
    print("row_major(3, 5) =", row_major.map_forward_index([3, 5]))

    # 3) XOR-3-bit swizzle
    xor_swz = Layout((16, 16), lambda i, j: [i, j ^ (i & 0x7)])
    print("xor_swizzle(3, 5) =", xor_swz.map_forward_index([3, 5]))


if __name__ == "__main__":
    main()
