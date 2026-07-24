# 第 7 章 · Layout 系统与 Fragment

> **TL;DR**：TileLang 里 `Layout` 和 `Fragment` **不是抽象概念**——它们是**一段函数**，
> 输入"逻辑坐标 `(i, j)`"，输出"物理坐标 `(phys_offset)`"（Layout）或"物理坐标 + 落在哪个线程"（Fragment）。
>
> 本章目标：让你看完之后能自己写出一个 `Layout(shape=[16, 16], forward_fn=lambda i, j: [j, i])`
> 的转置 layout，能看懂 `swizzled_layout` 里那些位运算，能明白 `alloc_fragment` 为什么必须要有 layout inference。
>
> **本章会读到的真实源码**：
> - [`tilelang/layout/layout.py`](../../tilelang/layout/layout.py) —— Python 侧 `Layout` 类
> - [`tilelang/layout/fragment.py`](../../tilelang/layout/fragment.py) —— Python 侧 `Fragment` 类
> - [`tilelang/layout/swizzle.py`](../../tilelang/layout/swizzle.py) —— 各种预制 swizzled layout 工厂
> - [`tilelang/language/annotations.py`](../../tilelang/language/annotations.py) —— `T.annotate_layout`
> - [`src/layout/layout.h`](../../src/layout/layout.h) —— C++ 底层 `LayoutNode` / `FragmentNode`
> - [`src/transform/layout_inference.cc`](../../src/transform/layout_inference.cc) —— LayoutInference pass
>
> **前置**：读完 [第 3 章](./03_tilelang_dsl.md)（知道 `alloc_shared` / `alloc_fragment` 是什么）和
> [第 5 章](./05_lowering_pipeline.md)（知道 `LayoutInference` 在 pipeline 里的位置）。

---

## 7.1 为什么需要 Layout？——先看一个不用 layout 会出啥事

假设你在写 128×128 tile 的 GEMM，`A_shared` 是 shared memory 上的 128×32 buffer。

```python
A_shared = T.alloc_shared((128, 32), "float16")
T.copy(A[by*128 : by*128+128, ko*32 : ko*32+32], A_shared)  # global → shared
T.gemm(A_shared, B_shared, C_local)                          # tensor core
```

**问题一（shared memory bank conflict）**：CUDA shared memory 有 32 个 bank，
每个 bank 4 字节宽。如果 128 个线程同时读 `A_shared` 的**同一列**（stride=32 * 2 字节 = 64 字节 = 16 banks），
会有多个线程落到**同一个 bank**——这叫 **bank conflict**，访问要串行化，慢好几倍。

**问题二（ldmatrix 对齐）**：Ampere 及以后的 Tensor Core 用 `ldmatrix.x4` 一次读 4 个 8×8 tile，
它要求 8 行的起始地址落在特定 "swizzled" 排布上。你直接 row-major 存进 shared，`ldmatrix` 会取到错的数。

**问题三（fragment 分给谁）**：`C_local = T.alloc_fragment((128, 128), "float32")` —— 这个 tile
到底"哪个元素归哪个线程"？一个 warp 有 32 线程、一个 CTA 有 128 线程（4 warp），
`C_local[i, j]` 具体存在哪个线程的哪个寄存器里？必须有明确规定，否则 `T.gemm` 和后续 `T.copy(C_local, C[...])`
会对不上。

**Layout 就是回答这三个问题的机制**——它是一个数学函数，写下"逻辑坐标 → 物理位置"的具体规则。

## 7.2 Layout：`(逻辑索引) → (物理索引)` 的一段函数

真实签名（[`tilelang/layout/layout.py:10`](../../tilelang/layout/layout.py)）：

```python
@tvm_ffi.register_object("tl.Layout")
class Layout(Node):
    def __init__(self, shape, forward_fn):
        """
        shape        : list[int]        # 逻辑形状
        forward_fn   : callable         # 逻辑坐标 -> 物理坐标（PrimExpr or list[PrimExpr]）
        """
```

> **看不懂头两行不要紧**，它们只是"Python 类接到 C++ 对象"的固定写法：
> - `class Layout(Node)` —— 继承 `Node`，表示这个 Python 类其实是一个 **C++ 侧对象的壳**（真正的实现是 C++ 的 `LayoutNode`，Python 这层只是转发）。
> - `@tvm_ffi.register_object("tl.Layout")` —— 把这个 Python 类和 C++ 侧名为 `"tl.Layout"` 的对象类型**登记绑定**起来，于是从 C++ 返回给 Python 的 Layout 对象，会自动"变成"这个 Python 类的实例。
>
> 这套跨 C++/Python 的桥接叫 **FFI**（Foreign Function Interface），第 8 章 8.2 会专门讲。**这里你只需知道：`Layout` 的真身在 C++，Python 端是它的可调用壳**——所以下面这些 `map_forward_index` 之类的方法，最终都落到 C++ 实现上。

### 最小例子：一个 2D 转置 layout

```python
from tilelang.layout import Layout

# 逻辑 shape (16, 16)，物理坐标 = (j, i)——把二维数组转置
layout = Layout((16, 16), lambda i, j: [j, i])

print(layout)                            # DebugOutput: 显示 shape 和 forward index
print(layout.map_forward_index([3, 5]))  # → [5, 3]
```

或者一个 **row-major linear layout**（就是普通的 C 数组下标算法）：

```python
# 逻辑 (M, N) → 物理一维偏移 i*N + j
layout = Layout((M, N), lambda i, j: [i * N + j])
```

或者一个 **swizzled layout**（打乱行的顺序，消 bank conflict）：

```python
# XOR 3 bit：i 的低 3 位和 j 的高 3 位做异或
layout = Layout((16, 16), lambda i, j: [i, j ^ (i & 0x7)])
```

### Layout 在 IR 里做什么？

TileLang 里每个 `alloc_shared / alloc_fragment` 出来的 buffer，**要么** 有个显式的 Layout 挂在它头上
（你通过 `T.annotate_layout({buf: layout})` 挂上去），**要么** 由 `LayoutInference` pass 自动推断出来。

一旦 buffer 有了 layout，之后**所有对这个 buffer 的下标访问**都会被 `LowerTileOp` /
`LowerSharedBarrier` 之类的 pass 用 layout 的 `map_forward_index` 重写成"物理偏移"。

也就是说：**Layout 是一段被 pass 拿来重写下标的规则**。它不是运行时对象，而是一段编译期函数。

```
   T.copy(A_shared[i, j], ...)                            # 你写的
                │
   ─────────────┼──── LayoutInference 决定 A_shared 的 Layout
                │
   layout.map_forward_index([i, j])                       # pass 用 layout 展开
                │
   ─────────────┼──── LowerTileOp / codegen 消费
                │
   A_shared_data[i * 32 + (j ^ ((i & 7) * 8))]            # 生成的物理访问
```

## 7.3 Fragment：Layout + "分给哪个线程"

`alloc_shared` 的 buffer 是**整个 threadblock 共享**的，所以 Layout 只需要说"物理偏移在哪"。
但 `alloc_fragment` 的 buffer 每个线程持有一份**私有**（放在寄存器里），
所以还得说"这个位置**归哪个线程**"。这就是 `Fragment`。

真实签名（[`tilelang/layout/fragment.py:12`](../../tilelang/layout/fragment.py)）：

```python
@tvm_ffi.register_object("tl.Fragment")
class Fragment(Layout):
    def __init__(self,
                 shape,
                 forward_fn=None,        # (*idx, [rep]) -> (thread, index)  一步给完
                 forward_thread_fn=None, # (*idx, [rep]) -> thread           两步给：先线程
                 replicate=1,            # 一份数据被复制到多少个线程
                 forward_index_fn=None): # (*idx) -> index                   再给物理位置
```

> 💡 **两个 fn 参数怎么选**？
> - `forward_fn`：一次给出"这个逻辑元素落在哪个线程的哪个位置"
> - `forward_thread_fn` + `forward_index_fn`：分开写，各管一头
> 底层是同一件事——**一个 Fragment 需要"线程映射 forward_thread + 位置映射 forward_index"两条信息**。

### 最小例子：**每 4 个线程共享一个元素**（replicate=4）

```python
from tilelang.layout import Fragment

# 一个 shape=(32,) 的 fragment，把 32 个元素分给 128 个线程，
# 每个元素被 4 个线程持有（都拿一份 copy）
frag = Fragment(
    shape=[32],
    forward_thread_fn=lambda i, rep: i * 4 + rep,   # thread id = i*4 + rep
    forward_index_fn=lambda i: [0],                 # 每个线程只存 1 个元素
    replicate=4,
)

print(frag.get_thread_size())    # 128
```

### 常见 Fragment：`MakeGemmFragmentC(block_m, block_n, warp_m, warp_n, element_size)`

这是 Ampere / SM80 Tensor Core `mma.m16n8k16` 的输出 fragment 布局，直接调 C++ 里的工厂函数
生成（[`src/layout/gemm_layouts.cc`](../../src/layout/gemm_layouts.cc)）。

在 quickstart 的 matmul 里，你**不用手写**它——`LayoutInference` pass 看到
`T.gemm(A_shared, B_shared, C_local)` 会自动帮 `C_local` 挑一个合适的 GEMM Fragment。
这就是下一节要讲的推断机制。

## 7.4 LayoutInference：让 pass 帮你把 layout 填齐

真实入口（[`tilelang/transform/__init__.py:41`](../../tilelang/transform/__init__.py)）：

```python
def LayoutInference():
    """LayoutInference pass. Assign layouts to fragment/shared buffers
    based on gemm/copy operations that constrain them."""
    return _ffi_api.LayoutInference()   # 底层 C++ pass
```

对应源码在 [`src/transform/layout_inference.cc`](../../src/transform/layout_inference.cc)。

### 它做什么？

它在 pipeline 里的位置（回顾 [第 5 章](./05_lowering_pipeline.md)）：

```
  ...
  LowerAndLegalize
    ├─ ...
    ├─ LayoutInference        ← 就是它
    ├─ LowerTileOp
    ├─ LowerL2Persistent
    └─ ...
```

任务：**给每个 fragment / swizzled shared buffer 决定一个 Layout / Fragment**。
它的输入是**约束**：

```
   T.gemm(A_shared, B_shared, C_local)
              │         │         │
              └─────────┴─────────┴───────► "输入 A、B、C 三个 buffer 必须满足 mma 指令的 layout 要求"

   T.copy(A_shared, A_fragment)
              │         │
              └─────────┴───────────────► "src 和 dst 的 layout 必须能对上"

   for i, j in T.Parallel(block_M, block_N):
        C_local[i, j] = ...              ► "T.Parallel + C_local 意味着 C_local 需要一个能均匀分给线程的 fragment layout"
```

Pass 内部做的事：
1. 遍历所有 `tl.tileop.gemm / tl.tileop.copy / T.Parallel` 节点，收集约束
2. 用约束求解器（`InferFragmentLayout`、`InferAtomicAddLayout` 等——见 [`src/op/`](../../src/op/) 各个 op 的 `InferLayout` 方法）
   决定每个 buffer 的 Layout
3. 把结果写进 `block.annotations["layout_map"]`（`LayoutNode` 里 `attr::kLayoutMap` 常量，见 [`src/layout/layout.h:288`](../../src/layout/layout.h)）

后面的 pass 就从 `layout_map` 里读 layout 去重写下标。

### 什么时候要手动指定？

**大多数情况下**——比如 quickstart 的 matmul——都**不用**手写 layout。
LayoutInference 从约束里推断出来的 layout **已经是最优的**（Tensor Core 指令要什么、shared memory bank
偏移用什么 swizzle，全都算好）。

**要手动指定的情况**：
1. 你自己写了一个不常见的访存模式，LayoutInference 推不出来
2. 你想强制某个 buffer 用某个特定 swizzle 来测性能
3. 你在写一个新的 tile op，需要给它挂一个 fully-replicated fragment（例如 mask buffer）

## 7.5 `T.annotate_layout` —— DSL 层手动挂 layout

真实签名（[`tilelang/language/annotations.py:29`](../../tilelang/language/annotations.py)）：

```python
def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer.

    layout_map : dict[Buffer, Layout | Fragment | Callable]
        key   = buffer 对象（alloc_shared/alloc_fragment 返回的那个）
        value = 一个 Layout / Fragment 对象，或者一个 lambda（会自动包成 Layout）
    """
```

### 用法（改 quickstart，强制 A_shared 走 128B swizzled layout）

```python
import tilelang, tilelang.language as T
from tilelang.layout import make_swizzled_layout

@tilelang.jit
def matmul(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 32):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local  = T.alloc_fragment((block_M, block_N), T.float32)

        # ⬇️ 手动指定 layout ⬇️
        T.annotate_layout({
            A_shared: make_swizzled_layout(A_shared),      # 128B swizzle
            B_shared: make_swizzled_layout(B_shared),
        })

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by*block_M, ko*block_K], A_shared)
            T.copy(B[ko*block_K, bx*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*block_M, bx*block_N])
    return C
```

### 生成的 IR 里的痕迹

`T.annotate_layout({A_shared: layout_a, ...})` 展开成一个 block attribute：

```
Block {
    annotations = {
        "layout_map": { A_shared.data: layout_a, ... }
    }
    body = ...
}
```

`LayoutInference` 遇到这个 attr 时**优先用你指定的**，只对没指定的 buffer 才推断。

## 7.6 Swizzle：消 bank conflict 的位运算魔法

先说**为什么需要 swizzle**。看下面这段"最朴素"的 shared memory 存取：

```
A_shared[i, j]  →  physical offset = i * 32 + j        # row-major

  bank[0-31] 分布如下（每 bank 4 字节 = 2 个 fp16）：
  行 0: [b0 b0 b1 b1 b2 b2 ... b15 b15]                  ← 一行 32 个 fp16 = 32*2 = 64 byte = 16 banks
  行 1: [b16 b16 b17 b17 ... b31 b31]                    ← 行 1 落在 bank 16..31
  行 2: [b0 b0 b1 b1 ... b15 b15]                        ← 行 2 又回到 bank 0..15
  行 3: [b16 b16 ...]
  ...
```

如果 32 个线程同时读 `A_shared[t, 0]`（一列）：`t=0` 读 bank 0、`t=1` 读 bank 16、`t=2` 读 bank 0、`t=3` 读 bank 16……
**同一 bank 冲突！** 变成 16-way conflict。

**Swizzle 做的事**：把行的物理位置**打乱**，让"一列" 访问也能均匀分到 32 个 bank。
典型做法是**用 i 的某几位 XOR 到 j 上**：

```
   物理列 = j XOR ((i >> shift) & mask)

  行 0: XOR 0     [b0 b1 b2 ... b15]
  行 1: XOR 1     [b1 b0 b3 b2 ...]                      ← 一列取完 32 行覆盖到 32 个不同 bank
  行 2: XOR 2     [b2 b3 b0 b1 ...]
  ...
```

预制工厂函数（均来自 [`tilelang/layout/swizzle.py`](../../tilelang/layout/swizzle.py)）：

| 函数 | 作用 |
|---|---|
| `make_swizzled_layout(buf, k_major=True, allow_pad=True)` | 通用 swizzle，自动挑合适粒度 |
| `make_full_bank_swizzled_layout(buf)` | 128B swizzle |
| `make_half_bank_swizzled_layout(buf)` | 64B swizzle |
| `make_quarter_bank_swizzled_layout(buf)` | 32B swizzle |
| `make_wgmma_swizzled_layout(buf, continuity, k_major)` | Hopper WGMMA 专用 |
| `make_volta_swizzled_layout(buf, is_a, k_inner)` | SM70 (Volta) Tensor Core 专用 |
| `make_tcgen05mma_swizzled_layout(buf, continuity, k_major)` | Blackwell TCGEN05MMA 专用 |
| `make_linear_layout(buf)` | 不 swizzle 的 row-major |
| `make_fully_replicated_layout_fragment(buf, threads)` | 每个线程都持有全量数据的 fragment |

### 一句话记忆

**"128B / 64B / 32B swizzle" = 每 128 / 64 / 32 字节做一次 XOR shuffle**。
数越大 swizzle 力度越强、能消更严重的 conflict，但对 buffer 起始地址对齐要求也越高。

## 7.7 全景：quickstart 里每个 buffer 的 layout 最终是什么

对 quickstart 的 matmul 跑一次 `tilelang.lower(pf, target="cuda").device_mod.script()`，
你会在 device_mod 的 block annotations 里看到类似：

```
block.annotations = {
    "layout_map": {
        A_shared_data:  MakeSwizzledLayout(shape=(128, 32), ...),   # 自动挑的 swizzle
        B_shared_data:  MakeSwizzledLayout(shape=(32, 128), ...),
        C_local_data:   MakeGemmFragmentC(block_m=128, block_n=128,
                                          warp_m=64, warp_n=64,
                                          element_size=32),
    }
}
```

**没有人显式写过这些**——都是 `LayoutInference` 从"你调了 `T.gemm`、用了 shared / fragment 存储域"
这两条约束里推出来的。

**验证方式**（对着 quickstart 的 matmul 跑一遍即可复现）：

```python
pf = matmul.get_tir(**cfg)
# 阶段一结束时看不到 layout_map（这时 LayoutInference 还没跑）
print("=== BEFORE LayoutInference ===")
print(pf.script())

art = tilelang.lower(pf, target="cuda")
# 阶段二 lowering 已经跑过，layout_map 已经就位（并且已经被下游 pass 用来重写下标了）
print("=== AFTER lower() ===")
print(art.device_mod.script())
```

搜 `"layout_map"` 或者 `"tl.MakeSwizzledLayout"` 这个字样定位。

## 7.8 亲手做一遍

**练习 1（读）**：改 quickstart，把 3.5 那段 GEMM 加一行 `T.annotate_layout({A_shared: make_linear_layout(A_shared)})`
（强制不 swizzle）；对比修改前后 `kernel.get_kernel_source()` 里对 `A_shared` 的下标计算表达式的变化。

**练习 2（读）**：把 `alloc_fragment((128, 128), "float32")` 改成 `alloc_fragment((100, 100), "float32")`。
再跑 `matmul.get_tir(...).script()`——看看会不会有报错，报错提示说明了什么？（提示：LayoutInference 需要**能整除**warp shape）

**练习 3（写）**：自己写一个 32×32 的 layout，逻辑坐标 `(i, j)` → 物理坐标 `[i, (j + i) % 32]`
（相当于每一行左移 i 位），观察生成的 CUDA 里 `A_shared[i, j]` 访问变成什么样子：

```python
from tilelang.layout import Layout

shift_layout = Layout((32, 32), lambda i, j: [i, (j + i) % 32])

# 在 kernel 里
T.annotate_layout({A_shared: shift_layout})
```

> ⚠️ **常见误解**
>
> - **"Layout 是一张存起来的映射表"** —— 不是。`Layout` / `Fragment` 本质是**一段编译期的函数**（给逻辑坐标 `(i, j)`，算出物理偏移 / 归哪个线程），它在编译期就被求值并烘焙进索引表达式，**运行时不存在这张表、也没有查表开销**。
> - **"`Fragment` 和 `Layout` 是一回事"** —— `Layout` 只回答"物理偏移在哪"；`Fragment` 在此之上**多回答一个"这个值归哪个线程/lane 持有"**。寄存器里的数据（如 `T.gemm` 的累加器）用 `Fragment`；shared memory 里的数据用 `Layout`。
> - **"必须自己写 layout 才能用上 swizzle"** —— 多数情况下**不用**。`LayoutInference` pass 会从 `T.gemm` / `T.copy` / `T.Parallel` 的约束里自动推出最优 swizzle / fragment。**先别手写**——先让它推，用第 11 章的 layout 可视化看推出来的是什么，只有当你确实要偏离默认策略时才 `T.annotate_layout(...)`。

## 7.9 本章要带走的三件事

1. **Layout / Fragment 是"逻辑坐标 → 物理位置"的一段编译期函数**——
   `Layout` 只给物理偏移；`Fragment` 还多给一个"归哪个线程"。
2. **绝大多数情况下你不用写 layout**——`LayoutInference` pass 从 `T.gemm` / `T.copy` /
   `T.Parallel` 的约束里推断，得到的**是最优 swizzle / 最优 fragment**。
3. **要手写 layout 时用 `T.annotate_layout({buf: layout})`**，layout 可以是：
   - `Layout(shape, fn)` / `Fragment(shape, ...)` 手写
   - `make_swizzled_layout(buf)` / `make_wgmma_swizzled_layout(buf)` 等预制工厂
   - 直接一个 `lambda i, j: [...]`（会自动包成 Layout）

---

下一章 [第 8 章 · Codegen：从 TIR 到 CUDA 源码到 cubin](./08_codegen_tir_to_cuda.md)：
终于到最后一个阶段——一个已经 lower 完的 IRModule，是怎么被"逐节点打印"成一段 CUDA C++ 字符串、
再被 NVRTC 编译成 cubin 的？我们会打开 `CodeGenTileLangCUDA` 看它对每种 TIR 节点分别产出什么。
