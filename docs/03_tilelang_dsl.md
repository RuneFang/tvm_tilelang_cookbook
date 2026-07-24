# 第 3 章 · TileLang DSL 层次

> **TL;DR**：本章逐一打开 `T.Kernel` / `T.alloc_shared` / `T.alloc_fragment` /
> `T.Pipelined` / `T.Persistent` / `T.Parallel` / `T.copy` / `T.gemm` / `T.clear` 每一个
> DSL 关键字**在源码里的真实签名**，说清楚：
> 
> 1. **它是什么**（普通函数？上下文管理器？占位 intrinsic？）
> 2. **它的参数具体是什么意思**
> 3. **它在生成的 TIR 里对应哪种节点**
> 4. **它属于第 1 章那张图的哪个阶段**
> 
> 每一节的所有代码片段**都能真的跑**，签名全部来自源码；如果你在别处看到不一样的签名，
> 
> 
> **本章会读到的真实源码**：
> 
> - [`tilelang/language/kernel.py`](../../tilelang/language/kernel.py)（`T.Kernel`）
> - [`tilelang/language/allocate.py`](../../tilelang/language/allocate.py)（`T.alloc_*`）
> - [`tilelang/language/loop.py`](../../tilelang/language/loop.py)（`T.Pipelined` / `T.Persistent` / `T.Parallel` / `T.serial` / `T.unroll`）
> - [`tilelang/language/copy_op.py`](../../tilelang/language/copy_op.py)（`T.copy`）
> - [`tilelang/language/gemm_op.py`](../../tilelang/language/gemm_op.py)（`T.gemm` / `T.wgmma_gemm` / `T.tcgen05_gemm`）
> - [`tilelang/language/fill_op.py`](../../tilelang/language/fill_op.py)（`T.clear` / `T.fill`）
> 
> **前置**：读完 [第 2 章](./02_tvm_tir_basics.md)（知道 Stmt / Expr / For / AttrStmt / Call 是什么）。

---

## 3.1 DSL 的两个角色：**语法糖 vs intrinsic**

TileLang DSL 里的关键字乍看像"函数"，但**实际上分两类**，行为截然不同。理解这个分类，
后面每个关键字都能秒懂：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TileLang DSL 关键字                              │
├────────────────────────────┬────────────────────────────────────────────┤
│      A. "语法糖" (frame)     │      B. "占位 intrinsic" (call)             │
│                            │                                            │
│  它在解析阶段就展开成"标准    │  它在解析阶段只是造一个 Call 节点，          │
│  的 TIR 节点组合"，之后就没  │  节点名 = "tl.tileop.xxx"，                 │
│  它啥事了。                  │  之后靠 pass 把这个 Call 展开成具体指令。    │
│                            │                                            │
│  例子：                      │  例子：                                     │
│  - T.Kernel   → For + attrs │  - T.copy   → Call("tl.tileop.copy")       │
│  - T.Parallel → parallel-For│  - T.gemm   → Call("tl.tileop.gemm")       │
│  - T.Pipelined→ For + attrs │  - T.clear  → Call("tl.tileop.fill")       │
│  - T.alloc_*  → Allocate    │                                            │
│                            │                                            │
│  归口于 [1.2 图] 的 **阶段一** │  归口于 [1.2 图] 的 **阶段二前期**          │
│  已经产出完整节点            │  产出符号；阶段二的 lower_tile_op pass 展开   │
└────────────────────────────┴────────────────────────────────────────────┘
```

> 💡 **一句话记忆**：`T.Kernel / T.Parallel / T.Pipelined / T.alloc_*` **在解析后**就"落地"了；
> `T.copy / T.gemm / T.clear` **在解析后**还是"半成品"符号，要等 pass 把它变成真代码。

## 3.2 `T.Kernel`：一次 kernel launch 的 grid + block 声明

真实签名（来自 [`tilelang/language/kernel.py:277`](../../tilelang/language/kernel.py)）：

```python
def Kernel(
    *blocks: int | tirx.PrimExpr,           # gridDim.(x[,y[,z]])
    threads: int | list[int] | tuple | None = None,  # blockDim.(x[,y[,z]])
    prelude: str | None = None,             # 一段会被 include 进生成 CUDA 的 C 代码
) -> KernelLaunchFrame
```

### 参数详解

- `*blocks`：位置参数，1~3 个，依次对应 `gridDim.x`、`gridDim.y`、`gridDim.z`
- `threads`：可以是整数（只设 `blockDim.x`），也可以是 `(x, y, z)` 元组；`-1` 表示"跳过这一维绑定"
- `prelude`：一段 C 代码字符串，会被塞到生成的 CUDA 源码顶部（一般不用）

### 用法：作为上下文管理器

```python
# 1-D grid
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    # bx 就是 blockIdx.x
    ...

# 2-D grid + 2-D block
with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
    tx, ty = T.get_thread_bindings()   # 拿到 threadIdx.x / threadIdx.y
    ...
```

### 它在 TIR 里长什么样

一次 `T.Kernel(gx, gy, threads=(tx, ty))` 会展开成一坨 `thread_binding` For：

```
For(blockIdx.x, extent=gx, kind=kThreadBinding)
  For(blockIdx.y, extent=gy, kind=kThreadBinding)
    For(blockIdx.z, extent=1,  kind=kThreadBinding)
      For(threadIdx.x, extent=tx, kind=kThreadBinding)
        For(threadIdx.y, extent=ty, kind=kThreadBinding)
          For(threadIdx.z, extent=1,  kind=kThreadBinding)
            <你 with 里写的那些语句>
```

> 💡 **概念卡：`thread_binding`**
> 第 2 章讲过 `For.kind` 可以是 Serial / Parallel / Vectorized / Unrolled / **ThreadBinding**。
> ThreadBinding 就是"这个循环变量不是普通迭代变量，它直接**是** `blockIdx.x` 或 `threadIdx.x`"。
> Codegen 看到这种 For 就不会打印 `for (int i = ...)`，而是**吞掉**它、把循环变量当成 CUDA 内建变量用。

### 亲手看一眼

```python
prim_func = matmul.get_tir(**cfg)     # cfg 见第 1 章
print(prim_func.script())
```

你会在输出里看到类似 `T.launch_thread("blockIdx.x", ...)` 或直接的 `blockIdx.x` 变量——那就是它。

### 相关同门

- `T.ClusterKernel(*blocks, cluster_dims=..., threads=...)`：Hopper (SM90) 的 thread block cluster 版
- `T.get_thread_binding(dim)` / `T.get_thread_bindings()` / `T.get_block_bindings()`：kernel 内部取绑定变量
- `T.get_thread_extent(dim)` / `T.get_block_extent(dim)`：取 threads / blocks 的 extent

## 3.3 `T.alloc_shared` / `T.alloc_fragment` / `T.alloc_local` / `T.alloc_var`

真实签名（来自 [`tilelang/language/allocate.py`](../../tilelang/language/allocate.py)）：

```python
def alloc_shared(shape, dtype, scope="shared.dyn") -> Buffer
def alloc_fragment(shape, dtype, scope="local.fragment") -> Buffer
def alloc_local(shape, dtype, scope="local") -> Buffer
def alloc_var(dtype, *args, scope="local.var", init=None) -> Buffer
def alloc_global(shape, dtype) -> Buffer      # 全局 workspace
```

### 差别只有一个：`scope`

它们内部**都**调用同一个函数 `T.sblock_alloc_buffer(shape, dtype, scope=...)`——
真正区分它们的是 `scope` 参数。

### 概念卡：**storage scope**（存储域）

CUDA 内存分层，从"远且慢、大但共享"到"近且快、小但独占"：

```
                   ┌──────────────┐
                   │   global     │  ← "shared.dyn / shared"    "global"
      ↑ 快、少     │   memory     │
      │  ▲        ├──────────────┤
      │  │        │  L2 cache    │
      │  │        ├──────────────┤
      │  │  一个  │  shared      │  ← T.alloc_shared → scope="shared.dyn"
      │  │  block │  memory      │       (整个 threadblock 共享)
      │  │  可见  │              │
      │  ▼        ├──────────────┤
      │           │  register    │  ← T.alloc_local     → scope="local"
      │  单线程   │  file /      │  ← T.alloc_fragment  → scope="local.fragment"
      ↓ 大、多    │  local mem   │       (每个线程私有；fragment 表示分给 warp
                   └──────────────┘        的 tile 里"本线程的那一份")
```

`local.fragment` 是 TileLang 独有的一个 scope：从**语义**上看它就是"local"（thread private），
但打上这个 tag 后 layout inference pass 会给它推断出一个 `Fragment` layout
（一个 warp 32 线程如何瓜分这个 tile）。第 7 章会详讲 fragment。

### 生成的 TIR 节点

每次 `alloc_shared(shape, dtype)` 都生成一个 `Allocate` Stmt：

```
Allocate(A_shared, shape=(128, 32), dtype="float16", scope="shared.dyn")
  body = <后续所有对 A_shared 的读写>
```

### 陷阱：`shared.dyn` vs `shared`

`shared.dyn` = **动态**共享内存（对应 CUDA 里 `extern __shared__`，大小在 launch 时算出来）。
`shared` = 静态共享内存（编译期定尺寸）。TileLang 默认给你 `shared.dyn`
是为了后续 `merge_shared_memory_allocations` pass 能把多个 buffer 塞进同一段 shared 里省空间。

## 3.4 `T.Parallel`：把这层循环分派到线程

真实签名（来自 [`tilelang/language/loop.py:13`](../../tilelang/language/loop.py)）：

```python
def Parallel(
    *extents: int | tirx.PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,        # T.Fragment 类型，给这个 nest 挂个 fragment layout
    prefer_async: bool | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame
```

### 用法

```python
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)      # relu
```

**语义**：把 `block_M * block_N` 个元素**按某个 layout** 摊到当前 kernel 的 threads 上并行做。
"某个 layout" 一开始你不用指定，`LayoutInference` pass 会给你推断一个（第 7 章会讲）。
你也可以手工给 `loop_layout=some_fragment` 强指定。

### 常见困惑：`T.Parallel` vs 普通 Python `for`

TileLang 里有 **4 种 for**，各有各的用途：

| 写法                                              | For.kind    | 含义                         |
| ----------------------------------------------- | ----------- | -------------------------- |
| `for i in range(n):` （非 T.xxx）                  | Serial      | 每个线程都串行跑 n 次               |
| `for i in T.serial(n):` / `T.Serial(n)`         | Serial      | 同上，显式                      |
| `for i in T.Parallel(n):`                       | Parallel    | 把 n 摊到线程上                  |
| `for i in T.Pipelined(n, num_stages=k):`        | Serial + 注解 | 串行但打上"帮我做 k-stage 软件流水"的注解 |
| `for i in T.vectorized(n):` / `T.Vectorized(n)` | Vectorized  | 期望被向量化                     |
| `for i in T.unroll(n):` / `T.Unroll(n)`         | Unrolled    | 期望被展开                      |

> 💡 **看到 `for ko in T.Pipelined(...)` 里面套 `for i, j in T.Parallel(...)` 别慌**：
> 外层 `T.Pipelined` 依然是"这个 tile 里第 k 次外层迭代"（跨时间的串行），
> 内层 `T.Parallel` 是"这一次外层迭代里，块内所有线程一起做的一批工作"（跨线程的并行）。
> 两者维度正交，一个在**时间**上、一个在**空间**上。

## 3.5 `T.Pipelined`：把这层循环变成软件流水

真实签名（来自 [`tilelang/language/loop.py:112`](../../tilelang/language/loop.py)）：

```python
def Pipelined(
    start: tirx.PrimExpr,
    stop: tirx.PrimExpr | None = None,
    num_stages: int = 0,
    order: list[int] | None = None,   # 手动排 pipeline 时用
    stage: list[int] | None = None,   # 手动排 pipeline 时用
    sync: list[list[int]] | None = None,
    group: list[list[int]] | None = None,
) -> frame.ForFrame
```

### 用法

**推荐用法**（让编译器自己排 pipeline）：

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[..., ko*block_K], A_shared)
    T.copy(B[ko*block_K, ...], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

**手动排 pipeline**（很少用，见第 5 章）：

```python
for i in T.Pipelined(n, order=[1, 0], stage=[0, 1]):
    base: T.int32 = i * block         # 可 replay 的绑定，不算 pipeline 步骤
    T.copy(A[base], shared)           # order 里的第 0 项 → stage 0
    T.copy(shared, B[base])           # order 里的第 1 项 → stage 1
```

### 概念卡：**软件流水（software pipeline）**

想象你在食堂打饭：**做菜（读 global）慢，吃饭（算 gemm）快**。

- **无 pipeline**：读一批 → 算一批 → 读一批 → 算一批（读的时候干等）
- **k-stage pipeline**：`k` 个"托盘"（多版本 buffer），读的时候把 K+1 批数据放到下一个托盘上，
  同时"吃"当前托盘。**读和算重叠**，隐藏访存延迟。

```
时间轴 → → → →

无 pipeline:
  [load 0][gemm 0]        [load 1][gemm 1]        [load 2][gemm 2]

3-stage pipeline (num_stages=3):
  [load 0][load 1][load 2][gemm 0]
                          [load 3][gemm 1]
                                  [load 4][gemm 2]
                                          [load 5][gemm 3]
                                          ...
```

`num_stages=3` 意思是"允许最多 3 批数据同时在飞行"，因此 A_shared / B_shared
在生成的代码里会被**开成 3 倍大小**（3 个 slot 循环用）。这就是第 6 章要讲的 "multi-version buffer"。

### 生成的 TIR 节点

`T.Pipelined` **本身**不做展开，只是产出一个带 `software_pipeline_stage` 等注解的 For：

```
For(ko, extent=..., kind=Serial, annotations={
      "software_pipeline_stage":  [0, 0, 1],
      "software_pipeline_order":  [0, 1, 2],
})
  body = ...
```

> **这两个注解数组是软件流水的核心**——它告诉后面的 pass "循环体里每条语句分到哪个阶段、按什么顺序排"：
>
> - **`software_pipeline_stage`**：长度 = 循环体里可被调度的语句数（每条语句一个值）。值是**流水线阶段编号**——**0 最早、越大越晚**。同一个 stage 里的语句可以重叠/并行执行；不同 stage 之间有依赖关系（后一阶段必须等前一阶段的数据到位）。`[0, 0, 1]` 表示：第 0、1 条语句（通常是 global→shared 的 `T.copy` 读数据）同属 **stage 0**（producer 阶段）；第 2 条语句（`T.gemm` 计算）属于 **stage 1**（consumer 阶段）——即"先搬数据、再算"。
> - **`software_pipeline_order`**：同样长度，值是**语句在循环体里的顺序位置**。`[0, 1, 2]` 表示"按自然书写顺序执行"——这对应最常见的情形（先 copy A、再 copy B、再 gemm）。如果你的循环体写得更复杂（比如 gemm 前面还有 bias 加法），order 数组也可能不是 `[0,1,2]`。
> - **两个数组长度必须相等**（源码 `inject_pipeline.cc:3444` 有 `ICHECK_EQ` 断言）——因为它们**按同一索引描述同一条语句**：`stage[i]` 和 `order[i]` 对应循环体里第 `i` 条可调度语句。
>
> ⚠️ **不要混淆 `stage` 数组和 `num_stages`（缓冲深度）**：`num_stages` 是"要开几份 buffer 做双缓冲/三缓冲"，决定显存分配；`stage` 数组是"每条语句分到流水线第几档"，决定执行编排。二者是不同维度的东西。

真正把它展开成"3-stage 双缓冲循环"是 `PipelinePlanning` + `InjectPipeline` 两个 pass 干的（第 6 章）。

## 3.6 `T.Persistent`：外层 tile 调度（"一个 threadblock 处理多个 tile"）

真实签名（来自 [`tilelang/language/loop.py:90`](../../tilelang/language/loop.py)）：

```python
def Persistent(
    domain: list[tirx.PrimExpr],      # 整个 tile 域，例如 [ceildiv(M, block_M), ceildiv(N, block_N)]
    wave_size: tirx.PrimExpr,         # 一个 wave 内有多少个 threadblock（通常 = SM 数）
    index: tirx.PrimExpr,             # 当前 threadblock 在 wave 里的编号（一般 = blockIdx.x）
    group_size: tirx.PrimExpr | int = 8,   # 光栅化（rasterization）分组大小，用于 L2 命中率
) -> frame.ForFrame
```

### 用法

```python
with T.Kernel(wave_size, threads=128) as bid:   # 只 launch wave_size 个 block
    for i, j in T.Persistent(
            domain=[T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
            wave_size=wave_size,
            index=bid):
        # 每次迭代处理一个 tile (i, j)
        ...
```

### 概念卡：**Persistent Kernel**

**普通 kernel**：一个 tile launch 一个 threadblock，threadblock 用完就消失，
launch 开销 * tile 数。tile 数很多时（几千个），launch 开销累加会变可观。

**Persistent kernel**：只 launch **wave_size** 个 threadblock（wave_size ≈ SM 数），
让它们**持续存活**、内部**循环**处理多个 tile。这样：

- launch 只发生一次
- 相邻 tile 可以共享一些寄存器 / 状态
- 配合 L2 光栅化（`group_size`），相邻 tile 落在相邻 SM 上，L2 命中率↑

```
普通 kernel (M/block_M * N/block_N = 128 个 tile):
    ┌───┐┌───┐┌───┐  ...  ┌───┐    ← 128 次 launch，各 threadblock 各干一个 tile
    │TB0││TB1││TB2│       │TB127│
    └───┘└───┘└───┘       └────┘

Persistent kernel (wave_size = 8):
    Wave 0: TB0-tile0  TB1-tile1  ... TB7-tile7      ← 只 launch 8 个 TB
    Wave 1: TB0-tile8  TB1-tile9  ... TB7-tile15     ← 各 TB 自己循环处理下一批
    Wave 2: TB0-tile16 ...
    ...
```

> ⚠️ **陷阱**：Persistent 模式下，pipeline 循环最内层的 K 循环处理完之后**不能带着未清零的
> mbarrier phase 进入下一个 tile**，否则跨 tile 之间的同步会错位。第 6 章会把这个陷阱的因果链讲透。

## 3.7 `T.copy`：跨内存域拷贝的高层 intrinsic

真实签名（来自 [`tilelang/language/copy_op.py:54`](../../tilelang/language/copy_op.py)，只列常用参数）：

```python
def copy(
    src, dst, *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,   # "tma" / "cp_async" / "sync"
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt
```

### 用法

```python
T.copy(A[by*block_M, ko*block_K], A_shared)     # global → shared
T.copy(A_shared, A_local)                        # shared → fragment / register
T.copy(C_local, C[by*block_M, bx*block_N])      # fragment → global
```

`src` 和 `dst` 可以是 `Buffer`（整块）、`BufferRegion`（切片）或 `BufferLoad`（点访问）。

### 它在 TIR 里长什么样

它**不是**一个立刻可执行的语句，而是一个 `Call` intrinsic：

```
Evaluate(Call("tl.tileop.copy", src_region, dst_region, annotations={...}))
```

`Call` 本身没做任何拷贝，它只是**留下一个符号**："这里请帮我做一次 copy"。

### 阶段二里发生了什么

`LowerTileOp` pass（`src/transform/lower_tile_op.cc`）碰到这个 `Call("tl.tileop.copy", ...)`
时，会**看情况**把它展开成不同的东西：

```
      T.copy(global A, shared A_shared)
                ↓  LowerTileOp pass
      ┌─────────────────────────────────────────────────────┐
      │  1. 如果 disable_tma=False 且 满足对齐/形状约束      │
      │     → 展开成 TMA (Tensor Memory Accelerator) 指令   │
      │     → 生成 mbarrier arrive/wait                     │
      │  2. 否则 → 展开成一个 T.Parallel + cp.async         │
      │  3. 小尺寸 / scalar → 直接 BufferStore              │
      └─────────────────────────────────────────────────────┘
```

> 上图两个"异步拷贝"名词，先知道它们都是**"让搬数据和计算重叠起来"的硬件手段**（第 6/13 章详解）：
> 
> - **TMA**（Tensor Memory Accelerator）：Hopper（SM90）新增的**硬件搬运引擎**，一条指令搬一整块 tile，最省事、最快；
> - **cp.async**：Ampere（SM80）就有的**异步拷贝指令**，是没有 TMA 时的退路。
> 
> 简言之：同一句 `T.copy`，编译器会**看目标硬件和形状**自动选 TMA / cp.async / 普通 store——你不用手动区分。

这就是为什么 `T.copy` 看起来简单，pass 内部却要考虑 layout / 对齐 / 目标硬件。

### 相关同门

- `T.copy_cluster(...)` —— 跨 CTA 的 cluster-level 拷贝（TMA multicast / SM-to-SM）
- `T.tma_load(...)` / `T.tma_store(...)` —— 更底层、直接指定 TMA 描述符

> 上面出现了两个缩写，先记一下（第 13 章详解）：**CTA**（Cooperative Thread Array）就是 **CUDA 里的一个 thread block**（一个 `T.Kernel` 的 block）——二者是同义词，读源码时经常混用；**TMA** 是 Hopper 的硬件异步搬运引擎。

## 3.8 `T.gemm`：tile-level 矩阵乘

真实签名（来自 [`tilelang/language/gemm_op.py:149`](../../tilelang/language/gemm_op.py)）：

```python
def gemm(
    A, B, C,                                   # Buffer / BufferLoad / BufferRegion
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,                           # ROCm 才用
    mbar: BarrierType | None = None,           # Blackwell TCGEN5MMA 需要
) -> tirx.PrimExpr
```

### 用法

```python
T.clear(C_local)                                # 累加器清 0
for ko in T.Pipelined(...):
    T.copy(A[...], A_shared)
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)         # C_local += A_shared @ B_shared
```

**语义**：`C += op(A) @ op(B)`（`clear_accum=True` 则 `C = op(A) @ op(B)`），
`op` 由 `transpose_A/B` 决定。这个操作是**tile 级**的——对应 `block_M × block_N × block_K` 那个 tile。

### 它在 TIR 里长什么样

同样是一个 `Call` intrinsic：

```
Evaluate(Call("tl.tileop.gemm", A, B, C, ...))
```

### 阶段二里被展开成什么

`LowerTileOp` pass 会根据 target 挑一个具体后端：

```
      T.gemm(A_shared, B_shared, C_local)
                ↓
      ┌──────────────────────────────────────────────┐
      │  SM70/75/80  → mma m16n8k16 (Tensor Core)    │
      │  SM90 (H100) → wgmma (warp-group MMA)        │
      │  SM100 (B100)→ tcgen05_mma                    │
      │  AMD MI      → mfma                          │
      │  CPU         → 三重循环标量 GEMM             │
      └──────────────────────────────────────────────┘
```

对应的具体 lowering 代码在 [`tilelang/tileop/gemm/`](../../tilelang/tileop/gemm) 里。

### 概念卡：**warp policy**

`policy` 决定"把 tile 的 (M, N) 切给多少个 warp、按什么形状切"：

- `GemmWarpPolicy.Square` —— 尽量正方形分（比如 4 个 warp → 2×2）
- `GemmWarpPolicy.FullRow` —— 全部沿 M 切
- `GemmWarpPolicy.FullCol` —— 全部沿 N 切

### 相关同门

- `T.wgmma_gemm(...)` —— 强制走 Hopper WGMMA、**不**自动插 wait（用于手动调度 async）
- `T.tcgen05_gemm(...)` —— 强制走 Blackwell TCGEN5MMA、**不**自动插 wait
- `T.gemm_sp(...)` —— 稀疏 gemm

## 3.9 `T.clear` / `T.fill`：把 buffer 填 0 / 常量

真实签名（来自 [`tilelang/language/fill_op.py:40`](../../tilelang/language/fill_op.py)）：

```python
def clear(buffer) -> tirx.PrimExpr
def fill(buffer, value) -> tirx.PrimExpr
```

`T.clear(x)` 就是 `T.fill(x, 0)`。同样是一个 `Call("tl.tileop.fill", ...)` intrinsic，
由 `LowerTileOp` pass 展开成一个 `T.Parallel` 内嵌 `BufferStore` 的小循环。

## 3.10 全景：quickstart 里每一行的"归属"

回头看 quickstart 那段 matmul，把 3.1 的分类打上标签：

```python
with T.Kernel(gx, gy, threads=128) as (bx, by):          # A. 语法糖：阶段一
    A_shared = T.alloc_shared((block_M, block_K), dtype) # A. 语法糖：阶段一
    B_shared = T.alloc_shared((block_K, block_N), dtype) # A. 语法糖：阶段一
    C_local  = T.alloc_fragment((block_M, block_N), acc) # A. 语法糖：阶段一

    T.clear(C_local)                                     # B. intrinsic：阶段二 lower_tile_op

    for ko in T.Pipelined(nk, num_stages=3):             # A. 语法糖 (+ 注解)：阶段一
        T.copy(A[...], A_shared)                         # B. intrinsic：阶段二 lower_tile_op
        T.copy(B[...], B_shared)                         # B. intrinsic：阶段二 lower_tile_op
        T.gemm(A_shared, B_shared, C_local)              # B. intrinsic：阶段二 lower_tile_op

    for i, j in T.Parallel(block_M, block_N):            # A. 语法糖：阶段一
        C_local[i, j] = T.max(C_local[i, j], 0)          # 普通表达式：BufferStore

    T.copy(C_local, C[...])                              # B. intrinsic：阶段二 lower_tile_op
```

**"A" 的东西 `matmul.get_tir(...)` 里就已经是最终形态**；
**"B" 的东西 `matmul.get_tir(...)` 里还是 `Call("tl.tileop.xxx", ...)`**，`tilelang.lower(...)` 之后才展开。

想验证这段话？运行下面这个脚本对比两次输出：

```python
# 阶段一结束时（frame 展开完，intrinsic 还是符号）：
pf = matmul.get_tir(**cfg)
print(pf.script())            # 你会在里面看到 "tl.tileop.copy" / "tl.tileop.gemm" 等字样

# 阶段二结束时（LowerTileOp 已经跑过，intrinsic 展开成具体指令）：
art = tilelang.lower(pf, target="cuda")
print(art.device_mod.script())  # 里面的 "tl.tileop.copy" 已经不见了，
                                # 换成了 T.Parallel + cp.async / TMA / mma 之类
```

## 3.11 亲手做一遍

**练习 1**：改 quickstart，把 `num_stages=3` 改成 `num_stages=1`，用 `get_tir` 打印 TIR，
看看 `software_pipeline_stage` 注解的值变化。

**练习 2**：把 `T.alloc_shared` 换成 `T.alloc_shared(..., scope="shared")`（去掉 `.dyn`），
再对比 `get_kernel_source()`，观察生成的 CUDA 里 `__shared__` 声明的差别。

**练习 3**：手写一个只用 `T.copy` 的 elementwise-add kernel，跑通并对比 PyTorch：

```python
import tilelang, tilelang.language as T
import torch

@tilelang.jit
def add(A, B, block_N: int = 128):
    N = T.const("N")
    dtype = T.float32
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
        A_sh = T.alloc_shared((block_N,), dtype)
        B_sh = T.alloc_shared((block_N,), dtype)
        T.copy(A[bx*block_N], A_sh)
        T.copy(B[bx*block_N], B_sh)
        for i in T.Parallel(block_N):
            A_sh[i] = A_sh[i] + B_sh[i]
        T.copy(A_sh, C[bx*block_N])
    return C

kernel = add.compile(N=4096, block_N=128)
a = torch.randn(4096, device="cuda")
b = torch.randn(4096, device="cuda")
c = kernel(a, b)
torch.testing.assert_close(c, a + b)
print("ok! kernel_source:")
print(kernel.get_kernel_source())
```

## 3.12 本章要带走的三件事

1. **DSL 分两类**：**A. 语法糖**（`T.Kernel / T.alloc_* / T.Pipelined / T.Parallel`）
   在解析后就落到标准 TIR 节点；**B. tile-level intrinsic**（`T.copy / T.gemm / T.clear`）
   在解析后还是 `Call` 符号，靠 pass 展开。
2. **每个 DSL 关键字都能对应到第 1 章的"6 个阶段"中的某个阶段**——A 类在阶段一，B 类在阶段二前期。
3. **看 IR 是理解 DSL 的最好方式**：一个 `get_tir()` 一个 `tilelang.lower(...).device_mod`，
   两次打印一比对，DSL 关键字的"生老病死"就都在里面。

---

下一章 [第 4 章 · Pass 系统与 Pass Pipeline](./04_pass_system.md)：
既然"lower" 这个词一直挂在嘴边，我们终于打开 pass 这个盒子，
看 TVM 的 pass 抽象、`PrimFuncPass`、`PassContext`，然后自己写一个能挂进 pipeline 的 pass。
