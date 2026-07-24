# 第 12 章 · 控制流 / 动态形状 / Reduce · Scan · Atomic

> **TL;DR**：GEMM 只用到 TileLang 语言表面的一小块；真正写业务 kernel（attention、norm、MoE、量化）时，你还需要**控制流**、**动态形状**、**规约 / 前缀和 / 原子**这三类原语。本章把它们一次性铺清楚，并标注每个原语在 `tilelang/language/` 里的真实出处。
>
> **本章目标**
> 补齐 TileLang **语言表面**（`tilelang.language`）里，正文 10 章没系统讲、但你写任何非纯 GEMM kernel 时都会用到的三类原语：
> 1. **控制流** — `if` / `while` / `break` / `T.serial` / `T.unroll` / `T.Parallel` / `T.Pipelined` / `T.Persistent` / `T.any_of` / `T.all_of`
> 2. **动态形状** — `T.dynamic("m")`（旧名 `T.symbolic`，已弃用）
> 3. **规约 / 前缀 / 原子** — `T.reduce_*` / `T.warp_reduce_*` / `T.cumsum` / `T.cummax` / `T.atomic_*`

> 所有 API 都对应 `tilelang/language/` 下真实文件，本章每一节末尾都注明"事实出处"。

---

## 12.0 一张定位图

```
                    tilelang.language 提供的原语
   ┌──────────────────────────┬──────────────────────────────────┐
   │      控制流层            │      数据变换层                  │
   │      （本章 §12.1-3）    │      （本章 §12.5-7）           │
   │                          │                                  │
   │  if / while / break      │  T.reduce_sum/max/min/...       │
   │  T.serial / T.unroll     │  T.warp_reduce_*                 │
   │  T.Parallel              │  T.cumsum / T.cummax             │
   │  T.Pipelined             │  T.atomic_add / _max / _min ...  │
   │  T.Persistent            │                                  │
   │  T.any_of / T.all_of     │                                  │
   └──────────────────────────┴──────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  跨界原语（§12.4）：动态形状 T.dynamic("m") —— 让编译期形状 │
   │  变成运行期形状，其它所有原语都会自动支持它。               │
   └──────────────────────────────────────────────────────────────┘
```

前面 10 章你用的都是 `T.copy` / `T.gemm` / `T.Pipelined` 这几个"顶层"块，把很多细节隐藏了。本章你要下沉一层：**当默认路径不满足需求时，怎么手写这些细节**。

---

## 12.1 条件与短路：`if` / `T.any_of` / `T.all_of`

### Python `if` 直接工作

```python
for i in T.serial(N):
    if i < N:                     # TIR 表达式即可
        C[i] = A[i] + B[i]

# 三元
x = A[i] if i < N else 0
```

**易踩坑**：`if <python bool>` 会在 lower 时被折成常量分支，不会进 TIR。所以你想在 kernel 里检查"某个 config 是否开启"，可以直接写 Python `bool`，编译器自动展开成 dead-code elimination。

### 多个标量条件：直接用 Python 的 `and` / `or`

```python
if (i < M) and (j < N):           # 语义 = (i<M) && (j<N)，编译器翻成 TIR 短路
    C[i, j] = A[i, j] + B[i, j]
```

**注意别把它和 `T.any_of` / `T.all_of` 搞混**。看真实 API（`tilelang/language/logical.py`）：

```python
def any_of(buffer: BufferLikeType) -> tirx.PrimExpr: ...
def all_of(buffer: BufferLikeType) -> tirx.PrimExpr: ...
```

它们**只接受一个 Buffer / BufferRegion**（不是一串布尔表达式），语义是"检查一整块 buffer 里
**是否有 true / 是否全为 true**"——典型用途是判断一整块 mask：

```python
mask = T.alloc_fragment((block_M, block_N), "bool")
# ... 填充 mask ...
if T.all_of(mask):                # 整块 mask 全为 true 才进分支
    ...
```

所以：**组合几个标量条件用 Python `and` / `or`；对一整块 buffer 做归并判断才用 `T.all_of` / `T.any_of`**。

### 自动越界保护

TileLang 有一个叫 **LegalizeSafeMemoryAccess** 的 pass（在第 4 章的 pass pipeline 里出现过）。它会在**它能证明可能越界**的访问外面自动加 `if`，能证明安全的就把 `if` 去掉。

后果就是：**很多简单的 residual tile 边界你根本不用手写 `if`**，直接写 `C[gi, gj] = A[gi, gj] + B[gi, gj]` 就行，pass 会自动补 guard。只有当你需要**为越界给出自定义值**（不是 skip，而是填 0 / 填 -inf）时才手写 `if`。

> **事实出处**：`tilelang/language/logical.py`；pass 位于 `src/transform/legalize_safe_memory_access.cc`。

---

## 12.2 循环家族：`T.serial` / `T.unroll` / `T.Parallel` / `T.Pipelined` / `T.Persistent`

**5 种循环、5 种语义**。理解它们最直接的方式是问"每次迭代之间还有没有并行性 / 复用性"：

| 循环 | 并行度 | 用途 | 典型场景 |
|---|---|---|---|
| `T.serial(N)` | 无 | 老老实实 for 循环 | K 维累加、外层 batch |
| `T.unroll(N)` | 无（但展开） | 编译期展开 | 小的常量 loop（比如 4x8 tile 里的 inner） |
| `T.Parallel(M, N)` | thread 级并行 | 逐元素、复制、点乘 | `T.copy` 底下就是它 |
| `T.Pipelined(N, num_stages=k)` | 阶段级流水 | 让 copy 和 compute 重叠 | 所有 GEMM/attention 的 K 循环 |
| `T.Persistent(...)` | 线程块级持久化 | 持久 kernel（persistent CTA） | flash-decoding、大 M/N 稀疏调度 |

### `T.Parallel` 的两个隐藏 knob

```python
for i, j in T.Parallel(M, N, coalesced_width=8, loop_layout=my_fragment):
    C[i, j] = A[i, j] + B[i, j]
```

- `coalesced_width=`：告诉 vectorization pass **这条 loop 沿最内维应该一次访问多少元素**。默认让 pass 自己推。
- `loop_layout=`：**只加在最外层循环**，把整套嵌套并行循环打上"应该按这个 fragment 的 thread 映射来切"的注解。要求 `fragment.InputDim == 嵌套 parallel 的维数`。

这两个 knob 只有在**默认布局推断结果不理想**时才手工用；99% 场景不用管。

### `T.Pipelined` 的隐式契约

第 5 章已经讲过软件流水，这里只补 3 条**你在正文没见到的细节**：

1. `num_stages=k` 分配 `k` 份 shared-memory 副本。副本越多，能 overlap 越多阶段，但 shared memory 也吃得越多。
2. `T.Pipelined` 里的**scalar `Bind` 语句**（比如 `s = something`）不算 pipeline slot——见 `docs/programming_guides/software_pipeline.md`。所以你数 stage 时要跳过赋值。
3. 如果 body 里出现 `if`，pipeline pass 不会跨过它拆 stage。想避免的话把 `if` 提到 body 外。

### `T.Persistent` —— 提示存在

`T.Persistent(domain, wave_size, index, group_size=...)` 用来写**persistent CTA 模式**（一个 CTA 干很多 tile，避免每次 launch）。这条路径**依赖 `MultiVersionBufferRewriter` 与 warp-spec 的正确协作**（第 6 章 6.9 讲的 phase 对齐陷阱正出在这里）。日常不会直接手写，绝大多数场景用官方的 persistent template 就够。

> **事实出处**：`tilelang/language/loop.py`，`docs/programming_guides/control_flow.md`，`docs/programming_guides/software_pipeline.md`。

---

## 12.3 `while` / `break` / `continue`

```python
i = 0
while i < N:
    if done[i]:
        break
    if skip[i]:
        i += 1
        continue
    ...
    i += 1
```

3 条实用规则：

1. `while True:` **会被编译器直接拒绝**（TileLang 显式检查 constant-true 条件）。
2. `break` / `continue` 在 `T.serial` / `T.unroll` / `T.Parallel` / `while` 里都合法。
3. 但**在 `T.Pipelined` 里用 `break` 是危险的**——软件流水已经预取了未来 K 步的数据，`break` 之后要么读到了不属于当前迭代的值，要么触发 undefined behavior。写 pipeline 时用 `if guard: ... else: ...` 而不是 `break`。

---

## 12.4 动态形状：`T.dynamic("m")`

这一节是**很多用户第一次撞墙**的地方。TileLang 默认 `M, N, K` 是编译期常量——每换一个形状就要重编译。

要让 kernel 处理**运行期才知道**的形状，用 `T.dynamic(name)` 声明符号：

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul_dyn(A, B, block_M, block_N, block_K,
               in_dtype, out_dtype, accum_dtype,
               num_stages, threads):
    # 三个运行期尺寸
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")

    A: T.Tensor((M, K), in_dtype)
    B: T.Tensor((K, N), in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N),
                  T.ceildiv(M, block_M),
                  threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_K, block_N), in_dtype)
        C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

（这段代码是 `examples/dynamic_shape/example_dynamic.py` 的最小化版本，直接可跑。）

**核心规则**：

1. **`block_*` 依然是编译期常量**——不能对 tile 尺寸做 `T.dynamic`。只有全局形状 `M/N/K` 是运行期的。
2. `T.dynamic("m,n,k")` 支持一次拿多个（用逗号或空格分隔），返回 tuple。
3. **`T.symbolic` 是 `T.dynamic` 的老名字**（`v0.1.9` 起弃用，但仍可用）。看老代码时别惊讶。
4. 每次调用 kernel，实际形状会**从 torch tensor 的 shape 中推断**并当作运行时参数传下去；grid 大小也是运行时算的。

**为什么会有性能代价**：编译器无法在编译期做 shape-specific 的 unroll 或 vectorization 选择，所以某些子路径可能会走保守版本。**批量推理跑多种形状**时值得开动态形状；**单一形状追极致性能**时最好保持静态。

> **事实出处**：`tilelang/language/symbolics.py`（真实定义），`examples/dynamic_shape/example_dynamic.py`（真实可运行的示例）。

---

## 12.5 规约：`T.reduce_*` 家族

规约就是"把 buffer 沿某个维度合成一维"。在 GEMM 里 K 维就是最典型的规约维，但 TileLang 把它作为通用原语暴露出来：

```python
def reduce_sum(buffer, out, dim=-1, clear=True, batch=1) -> None
def reduce_max(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False) -> None
def reduce_min(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False) -> None
def reduce_absmax(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False) -> None
def reduce_abssum(buffer, out, dim=-1, batch=1) -> None
def reduce_bitand(buffer, out, dim=-1, clear=True, batch=1) -> None
def reduce_bitor (buffer, out, dim=-1, clear=True, batch=1) -> None
def reduce_bitxor(buffer, out, dim=-1, clear=True, batch=1) -> None
```

**关键语义**：

1. **形状约束**：如果 `buffer.shape == [X, d, Y]`，那么 `out.shape` 必须是 `[X, Y]` 或 `[X, 1, Y]` 二选一。编译期会直接 assert 报错，不合就抛 `ValueError`。
2. **`clear=True`（默认）** ：out 先被 init（sum → 0，max → -inf，min → +inf）再累加。**`reduce_sum` 特殊**：即使 `clear=True`，它内部也会走"临时 buffer 再累加"的路径（源码注释里明说了，防止 warp reduce 里同一份被累加 warpSize 遍）。
3. **`batch=k`（默认 1）** ：把 N 个输出元素分成 `ceil(N/k)` 组，每组只用**一对 barrier**做 AllReduce，从而把总 barrier 数减少 `k×`。`batch` 必须整除每线程的输出数。想优化 Flash-Attention 里的 online-softmax 时特别有用。
4. **`nan_propagate`（fp16/bf16 max/min 专用）** ：True 时用 CUDA `__hmax_nan/__hmin_nan`（NaN 会传播），False 时用 `__hmax/__hmin`（NaN 被忽略）。默认 False。

buffer 的 **scope 组合**决定 lowering 路径（源码里已经处理 4 种组合）：

| src scope | dst scope | 编译器做的事 |
|---|---|---|
| shared | shared | 各分配 fragment，copy 进去做，做完 copy 回来 |
| shared | fragment | 只给 src 分配 fragment，dst 直接是 fragment |
| fragment | shared | 只给 dst 分配 fragment，做完 copy 回 shared |
| fragment | fragment | 直接原地做，没有中间 copy |
| 其它 | 其它 | 报错 |

**warp-level 快捷版本**：`T.warp_reduce_sum / _max / _min / _bitand / _bitor` 接受**寄存器标量**，用 warp shuffle 一步做完，返回同一 warp 里所有 thread 都拿到的规约值。适合手写 warp 内快速 reduction。

```python
val = A_frag[i]                     # 每 thread 一个标量
warp_sum = T.warp_reduce_sum(val)   # 一个 warp 32 个 thread 里的和
```

### 一个真实迷你示例：向量求和

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def vec_sum(N: int, block: int):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), Out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=block) as (bx,):
            A_frag  = T.alloc_fragment((N,), "float32")
            OutFrag = T.alloc_fragment((1,), "float32")
            T.copy(A, A_frag)
            T.reduce_sum(A_frag, OutFrag, dim=0, clear=True)
            T.copy(OutFrag, Out)
    return main
```

> **事实出处**：`tilelang/language/reduce_op.py`（16KB），`src/op/reduce.cc`（C++ pass 侧）。

---

## 12.6 前缀操作：`T.cumsum` / `T.cummax`

前缀（scan）比 reduce 复杂：它保留每个位置的**累计中间量**。

```python
def cumsum(src, dst=None, dim=0, reverse=False) -> None
def cummax(src, dst=None, dim=0, reverse=False) -> None
```

用法：

```python
@T.prim_func
def kernel(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
    with T.Kernel(1, threads=128):
        A_s = T.alloc_shared((128,), "float32")
        T.copy(A, A_s)
        T.cumsum(A_s, A_s, dim=0)      # in-place
        T.copy(A_s, B)
```

**关键规则**：

1. `dst=None` 时是**原地**版本。
2. `reverse=True` 表示从右向左累加。
3. `dst.shape` 必须和 `src.shape` **形状严格相等**（不像 reduce 会降一维）。
4. **fragment 版本走一次 shared memory 中转**：当 `src.scope() == "local.fragment"` 时，编译器会 alloc 一份 shared，先 copy 进去、在 shared 上做 scan，再 copy 回 fragment。这是 GPU scan 算法的固有代价，不是 bug。

**支持的输入**：`Buffer` / `BufferRegion` / `BufferLoad` 三种都可以，所以下面这种 slice 也合法：

```python
T.cumsum(InputG_fragment[i * chunk_size : (i + 1) * chunk_size], dim=0)
```

> **事实出处**：`tilelang/language/scan_op.py`，`src/op/scan.cc`。

---

## 12.7 原子操作：`T.atomic_*` 家族

Atomic 用来在**多线程同时写同一位置**的场景（histogram、GEMM 的 split-K 累加、稀疏 gather）保证正确性。

真实 API（`tilelang/language/atomic.py`）：

```python
def atomic_max  (dst, value, memory_order=None, return_prev=False) -> PrimExpr
def atomic_min  (dst, value, memory_order=None, return_prev=False) -> PrimExpr
def atomic_add  (dst, value, memory_order=None, return_prev=False, use_tma=False) -> PrimExpr
def atomic_addx2(dst, value, return_prev=False) -> PrimExpr    # 一次原子加 2 个元素
def atomic_addx4(dst, value, return_prev=False) -> PrimExpr    # 一次原子加 4 个元素
def atomic_load (src, memory_order="seq_cst") -> PrimExpr
def atomic_store(dst, src, memory_order="seq_cst") -> PrimExpr
def atomic_or   (dst, value, memory_order=None) -> PrimExpr
```

**要点**：

1. **`memory_order`** 支持 CUDA / C++11 那套（`"relaxed"`, `"acquire"`, `"release"`, `"acq_rel"`, `"seq_cst"`）。`None` 会用后端默认。默认对性能友好，需要跨 block 严格顺序时才升级。
2. **`return_prev=True`** 会返回原子操作**之前**的值——等价于 CAS 循环里的 old value，用来实现 lock-free 数据结构。
3. **`atomic_addx2` / `atomic_addx4`** 是硬件级 packed 原子（sm_60+ 上的 `atomicAdd(half2*)` 之类的 packed 指令），比等价的 2/4 次单元素原子加带宽高很多。要求 dst 对齐。
4. **`use_tma=True`（仅 `atomic_add`）**：Hopper 上用 TMA 硬件做全局归约，跳过传统的 L2 原子路径。适合大批量、稀疏访问。要求 Hopper (SM90) 或以上。

### 典型场景：Split-K GEMM 用 atomic_add 合并部分和

```python
# 概念性伪代码
with T.Kernel(grid_x, grid_y, grid_k, threads=128) as (bx, by, bk):
    C_local = T.alloc_fragment((block_M, block_N), "float32")
    T.clear(C_local)
    # ... 在 K 上跑自己的一段 ...
    T.gemm(A_s, B_s, C_local)
    # 每个 K 段把自己那块加到 global C 上
    T.atomic_add(C[by * block_M, bx * block_N], C_local)
```

在真实 example 里，split-K / stream-K GEMM 都是这个模式。

> **事实出处**：`tilelang/language/atomic.py`（21.5KB），`src/op/atomic_add.cc` / `src/op/atomic_reduce.cc`。

---

## 12.8 组合小抄

把这一章的原语按"你其实想干什么"整理成对照表：

| 你想 | 用什么 |
|---|---|
| 逐元素 | `T.Parallel` |
| 一个大 K 维累加 | `T.Pipelined` + `T.gemm` |
| 求和 / max / min 到一个向量 | `T.reduce_*` |
| 前缀和 / 前缀最大 | `T.cumsum` / `T.cummax` |
| 多线程更新同一位置 | `T.atomic_*` |
| 运行期才知道形状 | `T.dynamic("m")` |
| 边界检查（简单） | 不写，让 `LegalizeSafeMemoryAccess` 处理 |
| 边界检查（想自定义填充值） | 手写 `if (i < M) and (j < N): ... else: ...` |
| 判断一整块 mask 是否全 true / 有 true | `T.all_of(mask)` / `T.any_of(mask)` |
| 提前退出内层循环 | `break`（**不要**在 `T.Pipelined` 里用） |
| warp 内 32 个 thread 求和 | `T.warp_reduce_sum` |

---

## 12.9 小结

- 控制流原语的核心信号是**并行性丢失多少**：`serial → unroll → Parallel → Pipelined → Persistent`，越靠右越"并行、复杂、依赖 pass 的正确性"。
- 动态形状：只对**全局 tensor 尺寸**用 `T.dynamic`，tile 尺寸永远保持编译期常量。
- reduce 有 `batch=` 和 `nan_propagate=` 两个 flag 大多数教程不会讲，需要 flash-attention 级别优化时会用上。
- atomic 有 packed（`addx2`/`addx4`）和 TMA（`use_tma=True`）两条快速路径，能上就上。
- **调试这一章的 API 就用第 11 章的工具**：形状对不上时先 Pass Diff 看 IR，数值对不上时先 `T.print` 前 8 个元素，layout 疑似错了就打开 `TL_LAYOUT_VISUALIZATION_ENABLE`。

下一章 [第 13 章](./13_cluster_tma_hopper.md) 进入 Hopper / Blackwell 时代专属的 cluster、TMA、warp specialization、tcgen05 等高级特性。
