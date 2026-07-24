# 第 13 章 · Cluster / TMA / Hopper 深挖

> **TL;DR**：Hopper（SM90）的三样硬件能力——**Thread Block Cluster**（跨 CTA 共享 shared memory）、**TMA**（硬件 async bulk copy）、**Warp Specialization**（producer/consumer 分工）——是现代高性能 GEMM / Attention kernel 快的根本原因。本章把它们在 TileLang 里对应的原语和落地方式讲清楚。
>
> **本章目标**
> 讲清楚 NVIDIA Hopper（SM90）**独有的三样"新玩具"**——它们是过去两年 TileLang 里最重要的能力提升：
> 1. **Thread Block Cluster**：把多个 CTA 绑成一组，共享地址空间
> 2. **TMA（Tensor Memory Accelerator）**：硬件级 async bulk copy
> 3. **Warp Specialization**：把一个 CTA 内的 warp 拆成 producer / consumer 组
>
> 学完这章你能：读懂 Flash-Attention 3、Hopper GEMM 那种「一堆 mbarrier 满天飞」的 kernel，也能自己动手改写。

> 前置：本章大量依赖第 5-6 章（软件流水 & warp specialization）打的底子。如果你没读过，遇到「WS」「stage」这些字眼卡住时回去补一下。

---

## 13.0 为什么 Hopper 值得单开一章

Ampere（A100，SM80）时代的 GEMM 秘诀基本就是：
- shared memory + async copy (`cp.async`)
- MMA fragment（tensor core）
- 软件流水把 copy 和 mma 交错开

Hopper（H100，SM90）多了三件事，**每一件都能带来 30%+ 加速**：

| 新特性 | 一句话 |
|---|---|
| Cluster | 让 4 个 CTA 共享 shared memory 地址空间 |
| TMA | 一个 PTX 指令搬一整块 tile，硬件做 mask/coalesce |
| Warp Specialization | 一部分 warp 专职 copy，其它 warp 专职 mma |

它们**互相配合**：TMA 让 async copy 极致，cluster 让 4 个 CTA 共享数据不用重复 DRAM 读，WS 让 copy 与 compute 真正在硬件上并行。所以本章按「先讲原语，最后合起来看一个 Split-K」的顺序展开。

要求：**CUDA CC ≥ 9.0**（H100 / Hopper / RTX 5090 / Blackwell 都行）。

---

## 13.1 Thread Block Cluster：从"一个 CTA"到"一组 CTA"

### 概念

普通 CUDA 里，每个 CTA（thread block）都是独立的沙盒。Hopper 允许你把 **N 个 CTA 打包成 1 个 cluster**，这些 CTA：

- **同时启动**（硬件保证同时驻留）
- **能互看 shared memory**（一个新的 `shared::cluster` 地址空间）
- **能相互发同步信号**（cluster barrier）

在 TileLang 里，写 cluster kernel 用 `T.ClusterKernel`，而不是普通的 `T.Kernel`：

```python
with T.ClusterKernel(
        grid_x, grid_y,
        threads=128,
        cluster_dims=(4, 1, 1)) as (bx, by):
    rank = T.block_rank_in_cluster()   # 0..3
    ...
    T.cluster_sync()                    # 全 cluster barrier
```

`cluster_dims=(4,1,1)` 意味着**每 4 个 CTA 一组**。硬件目前最大支持 8。

### 真实 API 表（`tilelang/language/cluster.py`）

| Builtin | 返回 | 干啥 |
|---|---|---|
| `T.block_rank_in_cluster()` | `int32` | 当前 CTA 在 cluster 里的 rank（0-index） |
| `T.cluster_sync()` | — | arrive + wait，全 cluster barrier |
| `T.cluster_arrive()` / `T.cluster_arrive_relaxed()` | — | 只 arrive |
| `T.cluster_wait()` | — | 只 wait |
| `T.alloc_cluster_barrier([count])` | Buffer | 分配 mbarrier，需要 `count` 次 arrive 才会翻转 |
| `T.mbarrier_arrive(bar)` | — | 显式 arrive 一次 |
| `T.mbarrier_wait_parity(bar, parity)` | — | 等 barrier 翻到指定 parity |

还有一组 CLC（cluster launch control）：`T.clc_try_cancel`、`T.clc_is_canceled` 等——这些是**动态 CTA 调度**的高级用法，先跳过。

### Cluster 的两个"杀手锏"

有 cluster 之后，`T.copy` 可以做两件 A100 时代做不到的事：

1. **多播（multicast）**：一次 DRAM 读，同一份 tile 广播给多个 CTA
2. **CTA-to-CTA 直接拷贝（SM-to-SM copy）**：不走 global memory，直接把一块 shared memory 推到邻居 CTA 的 shared memory

后面 §13.3、§13.4 分别展开。

---

## 13.2 TMA：Hopper 的异步搬运工

### TMA 是什么

**TMA = Tensor Memory Accelerator**，硬件级的异步块拷贝引擎。你把一个「global 张量的多维切片」描述交给它，它自己算出 addressing、mask 边界、coalesce 访存，最后写完 signal 一个 mbarrier。

对比：

| 方式 | 谁算地址 | 谁做 mask | 完成信号 |
|---|---|---|---|
| 普通 `ld.global` | 每个 thread | 每个 thread `if` | `__syncthreads()` |
| A100 `cp.async` | 每个 thread | 每个 thread `if` | `cp.async.wait_group` |
| Hopper **TMA** | 硬件 descriptor | 硬件自动 | **mbarrier** |

TMA 的三大好处：

1. **单指令搬一整块**。整个 128×32 的 tile 就 1 条 `cp.async.bulk.tensor` 指令。
2. **不占用 warp 的算力**。发起 TMA 后 warp 可以立刻去干别的活。
3. **原生支持 mask**：边界 tile 自动 zero-fill，你不用写一堆 `if`。

### 在 TileLang 里你根本不需要显式调 TMA

关键：**你还是写 `T.copy`**。当满足下面几个条件时，编译器**自动**把它 lower 到 TMA：

- 目标是 Hopper (SM90) 或更高
- src/dst 之一是 global，另一是 shared
- 形状对齐（16-byte 对齐；具体见 `src/cuda/op/copy.cc` 里的 alignment 检查）
- （对 `copy_cluster`，多加一个 cluster 相关的条件——见 §13.3）

**验证 TMA 是否真的被 lower 出来**：用第 11 章的 `register_cuda_postproc_callback` dump 一下 CUDA 源码，grep `cp.async.bulk.tensor` 或 `tma_load` / `tma_store`。看不到就说明 TileLang 走了 fallback 路径，通常是形状 / 对齐没满足要求。

### 关键 API：mbarrier

TMA 完成时 signal 的是 **mbarrier**（memory barrier），Hopper 引入的新硬件同步原语。TileLang 里的封装：

```python
bar = T.alloc_cluster_barrier([1])       # 初始化，需 1 次 arrive
# ... 发起一次 TMA，硬件会自动 arrive ...
T.mbarrier_wait_parity(bar[0], parity=0) # 等它翻转
```

**parity** 是 mbarrier 的独有概念：barrier 翻转一次 parity 从 0 变 1，再翻转从 1 变回 0。所以循环里第 k 次等待时的 parity 是 `k % 2`。这是它**比传统 `__syncthreads()` 更省**的地方——不用重新分配 barrier。

---

## 13.3 TMA Multicast：一次 DRAM 读，给多个 CTA

### 场景

Split-K GEMM：多个 CTA 都要读**同一个 A 的 K-panel**。传统方式每个 CTA 各读一份，DRAM 带宽被浪费 N 倍。

Multicast 让**一个 CTA 发起 TMA**，硬件把结果**同时**塞进多个 CTA 的 shared memory：

```
Global memory ──TMA multicast──▶ smem (rank 0)
                              └─▶ smem (rank 1)     ← 同一份 tile，DRAM 不再多读
```

### API：`T.copy_cluster(src_global, dst_shared, cluster_mask=<int>)`

```python
with T.ClusterKernel(
        T.ceildiv(N, block_N),
        T.ceildiv(M, block_M),
        threads=128,
        cluster_dims=(4, 1, 1)) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_N), "float16")

    # 参与多播的 CTA 集合，用 bitmask 表示：ranks 0 和 1 共享一份
    T.copy_cluster(A[by * block_M, bx * block_N],
                   A_shared,
                   cluster_mask=0b0011)
```

**mask 语义**：

- `cluster_mask` 里每一 bit 对应一个 rank；set 表示该 rank 参与多播。
- **rank 等于 mask 里最低 set bit 的那个 CTA 负责真的发指令**（`cp.async.bulk.tensor ... multicast::cluster`）。
- 其它参与 CTA 只是被动接收，**不发指令**。
- mask 外的 CTA 各自走普通 TMA load（独立的 tile）。

上面例子里 `cluster_mask = 0b0011`：

| Rank | 行为 |
|---|---|
| 0 | 发起 multicast load |
| 1 | 被动接收，拿到跟 rank 0 一样的 tile |
| 2 | 独立 TMA load |
| 3 | 独立 TMA load |

**限制**：`cluster_mask` **必须是编译期常量**，不支持动态 mask。

---

## 13.4 SM-to-SM Copy：绕开 global memory 交换数据

### 场景

- Split-K 的 partial sum 归并：每个 CTA 算出自己那份，直接推给"归并 CTA"
- Producer-Consumer 里 producer CTA 直接填 consumer 的 buffer
- Cluster 内 all-to-all

### API：`T.copy_cluster(src_shared, dst_shared, dst_block=<rank>, remote_barrier=<mbar>)`

```python
if pid == 0:
    for i in T.Parallel(N):
        s_src[i] = A[i]
    # 把 s_src 推到 rank 1 的 s_dst，完事 signal rank 1 的 barrier
    T.copy_cluster(s_src, s_dst, dst_block=1, remote_barrier=s_barrier[0])

if pid == 1:
    T.mbarrier_wait_parity(s_barrier[0], 0)  # 等 rank 0 写完
    for i in T.Parallel(N):
        B[i] = s_dst[i]
```

### 三条 lowering 路径

编译器根据 `remote_barrier` 有没有给、region 是否 contiguous，**自动**在三条路径里选：

| 路径 | 触发条件 | 硬件指令 | arrive 次数 |
|---|---|---|---|
| **TMA fast path** | `remote_barrier` 给 + region 连续 | 1 条 `tl::tma_store_cluster` | 1 |
| **Multi-TMA path** | `remote_barrier` 给 + ND 但非连续 | 每 contiguous 行 1 条 TMA | 行数 |
| **SIMT fallback** | 没给 `remote_barrier`，或形状无法分解 | 每 thread 一个 scalar store 走 `map_shared_rank` | 若给了 `remote_barrier` 会自动注入 arrive |

**"连续"的定义**：最内维正好覆盖整个 buffer 宽度。如果你切片 `[..., 0:N_tile]` 而 `N_tile < buffer.shape[-1]`，就是不连续，走 Multi-TMA。

**关键**：**API 完全一样**，只是内部路径不同。你不用管选路径，但你要理解：**Multi-TMA 时 `arrive_count` 会被编译器自动改成"行数"**。所以你 `T.alloc_cluster_barrier([1])` 声明的 1 只是初值，编译期会重写。

### 同步契约

| | TMA fast | Multi-TMA | SIMT fallback |
|---|---|---|---|
| 源 CTA | 不用等 | 不用等 | 循环结束即同步 |
| 目标 CTA | `T.mbarrier_wait_parity(bar, parity)` | 同上 | `T.cluster_sync()`，或如果有 auto-arrived 也可 wait_parity |

### 前置要求（**很容易踩**）

1. src 和 dst 的 scope 必须都是 `shared` 或 `shared.dyn`。
2. mbarrier 要用 `T.alloc_cluster_barrier([arrive_count])` 分配，**不能**用普通 barrier 代替。
3. **分配 barrier 后、发送 copy 前**必须有一次 `T.cluster_sync()`——保证所有 CTA 都到 barrier init 点之后再有 CTA 开始 push 数据。
4. `cluster_mask` 和 `dst_block` **互斥**：一次 `T.copy_cluster` 只能干一件事。

---

## 13.5 Warp Specialization：一个 CTA 内的分工

### 概念回顾（详见第 6 章）

一个 CTA 里通常有多组 warp（比如 8 组 = 256 threads）。传统写法所有 warp 干同一件事。Hopper 的最佳实践是：

- 让 warpgroup 0 专职**发 TMA**（producer）
- 让 warpgroup 1 / 2 专职**做 mma**（consumer）

因为**TMA 发指令的过程本身很快，只需要一个 warp** 就够，剩下的算力全都用来做 tensor core。

### TileLang 里的 `T.ws` / `T.WarpSpecialize`

```python
import tilelang.language as T

# 让 tx < 128 的 warpgroup 干"分支 A"
with T.ws(0):
    T.copy(A_g, A_s)
    T.copy(B_g, B_s)

# tx >= 128 的 warpgroup 干"分支 B"
with T.ws(1):
    T.gemm(A_s, B_s, C_f)
```

真实实现（`tilelang/language/warpgroup.py`）：

- `T.ws(0)` = `if tx < 128:`
- `T.ws(1)` = `if tx >= 128 and tx < 256:`
- `T.ws(0, 1)` = `if tx < 128 or (tx>=128 and tx<256):`
- warp group size 固定 128（NVIDIA 硬件规定）
- 支持多维 threadIdx：内部会自动 flatten `tid = z*ey*ex + y*ex + x`

### 编译器怎么调度它

第 6 章已经讲过，这里补一个**关键实现细节**：

- **producer / consumer 之间需要 mbarrier 同步**——producer 发完 TMA arrive，consumer wait_parity 后开始 mma。
- **`MultiVersionBufferRewriter`** 生成 stage/version 索引；**`ProducerConsumerWarpSpecialized`** 消费它们。这两 pass 之间的**语义分层必须通过 provenance（compiler-internal intrinsic）通信**，不能靠语法模式匹配——这是 pass 组合里最容易踩的坑（详见第 6 章 6.7）。
- 如果你自己在源码里写 `if (k+1) % num_stages == 0` 这种表达式，**必须**保证它跟编译器生成的 stage 索引在 IR 上有区别，否则会被误替换。**建议避免写这种模式**。

### `no_set_max_nreg`

有时候你想跟编译器说"这个 kernel 别做 register throttle"，用：

```python
T.no_set_max_nreg()
```

这会在生成的 CUDA 里省掉 `.reg` 数量的显式设置。属于 fine-tuning，不推荐日常用。

---

## 13.6 综合案例：Cluster + Multicast + SM-to-SM 的 Split-K GEMM

把前面全揉在一起——这是 `docs/programming_guides/cluster_tma.md` 里给出的 sketch，我逐行注释：

```python
@T.prim_func
def split_k_gemm(A, B, C):
    with T.ClusterKernel(
            grid_x, grid_y, threads=256,
            cluster_dims=(4, 1, 1)) as (bx, by):

        rank    = T.block_rank_in_cluster()               # 0..3

        # ─── shared / fragment 分配 ───
        A_s     = T.alloc_shared((BM, BK), "float16")
        B_s     = T.alloc_shared((BK, BN), "float16")
        C_f     = T.alloc_fragment((BM, BN), "float32")
        C_parts = T.alloc_shared((4, BM, BN), "float32")   # 4 个 rank 各占一份
        barrier = T.alloc_cluster_barrier([3])             # 需要 3 次 arrive（rank 1,2,3）
        T.clear(C_f)

        # ─── Phase 1: 每个 rank 算自己一段 K ───
        for ko in T.Pipelined(T.ceildiv(K, BK * 4), num_stages=3):
            k_off = (rank + ko * 4) * BK

            # ★ multicast: rank 0/1 共享同一份 A tile，省 DRAM 带宽
            T.copy_cluster(A[by * BM, k_off], A_s,
                           cluster_mask=0b0011)
            T.copy(B[k_off, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)

        # 把 rank 自己的部分和搬进 C_parts[rank]
        T.copy(C_f, C_parts[rank])
        T.cluster_sync()

        # ─── Phase 2: rank 1/2/3 各自推自己那格到 rank 0 ───
        if rank != 0:
            # dst 是 rank 0 的 shared memory 里 index 相同的 slot
            # 各 rank 写不同 slot → 无写冲突
            T.copy_cluster(C_parts[rank], C_parts[rank],
                           dst_block=0,
                           remote_barrier=barrier[0])

        # rank 0 等 3 次 arrive → 归并 4 份，写回 global
        if rank == 0:
            T.mbarrier_wait_parity(barrier[0], 0)
            # C_parts[0..3] 现在在 rank 0 的 shared memory 里都有值
            T.copy(C_parts[0], C[by * BM, bx * BN])
```

**关键设计决策**（面试题级别）：

1. **为什么用 `C_parts[rank]` 而不是复用 `C_f`？** 因为 SM-to-SM 需要 src 在 shared，而 fragment 在寄存器里。
2. **为什么 `arrive_count = 3` 而不是 4？** 因为 rank 0 自己不 push（它是接收方），只有 rank 1/2/3 push。
3. **为什么每个 rank 写自己那个 slot？** 如果所有 rank 都 push 到同一个 `C_parts[0]`，就是**多个源写同一个目标**，会数据竞争。分槽 = 无锁。

---

## 13.7 Blackwell 一瞥：`tcgen05`、`lower_blackwell_2sm`

Hopper 之上还有 Blackwell（sm_100）。TileLang 已经开了几条实验性通道，但主要面向内部 template 用户：

| 特性 | 真实源码 |
|---|---|
| **`tcgen05` 布局**：Blackwell 新版 tensor core 的 fragment 布局 | `src/layout/tcgen05_layout.cc`，`src/cuda/transform/inject_tcgen05_fence.cc` |
| **2-SM MMA**：两个 SM 协同做一个 MMA | `src/cuda/transform/lower_blackwell_2sm.cc` |
| **Shared TMEM**：Blackwell 独有的 tensor memory scope | `src/cuda/transform/lower_shared_tmem.cc` |
| **示例**：blockscaled_gemm_sm100、flash_attention_sm100 | `examples/blockscaled_gemm_sm100/`, `examples/flash_attention_sm100/` |

日常使用**你几乎不会直接碰这些**——写 kernel 时它们都是通过 `T.gemm` + pass config 自动选路径。但你在 pass diff 里看到 `tcgen5_*`、`tmem`、`2sm` 这些字眼时，就知道是在 Blackwell 路径。

---

## 13.8 什么时候用什么：决策清单

```
你要写一个 kernel：

  ├─ 目标是 SM < 90 吗？
  │    └─ 是 → 别用本章任何东西，只用 T.Kernel + T.copy + T.Pipelined 就好
  │
  ├─ 单个 tile 是否要被多个 CTA 复用？
  │    └─ 是 → cluster + T.copy_cluster(..., cluster_mask=...)
  │
  ├─ 多个 CTA 是否要交换 partial sum / 中间结果？
  │    └─ 是 → cluster + T.copy_cluster(..., dst_block=..., remote_barrier=...)
  │
  ├─ 有大量 TMA 发起 + 大量 mma 计算，想并行做？
  │    └─ 是 → T.ws(0)/T.ws(1) 分工，用 mbarrier 同步
  │
  └─ 只想让 TileLang 尽力优化？
       └─ 什么都不做 —— 目标 = sm_90 时编译器会自动尝试用 TMA 加速普通 T.copy
```

**性能预算的经验数字**（不承诺准确，只是量级）：

- 加 TMA 单指令（vs. `cp.async`）：**5-15%**
- 加 warp specialization：**10-30%**
- 加 cluster multicast（Split-K 类）：**10-20%**
- 上 tcgen05（Blackwell）：**20-40%**

---

## 13.9 小结

- **Cluster** = 一组 CTA 共享 shared memory 地址空间；用 `T.ClusterKernel` + `cluster_dims=` 打开
- **TMA** = Hopper 硬件级 async bulk copy；你写 `T.copy` 编译器自动选路径，用 `register_cuda_postproc_callback` 验证是否落地
- **`T.copy_cluster(..., cluster_mask=...)`** = 多播；同一 tile 广播给多个 CTA
- **`T.copy_cluster(..., dst_block=..., remote_barrier=...)`** = SM-to-SM 拷贝；不走 global 交换数据
- **`T.ws(i)`** = 让某组 warp 干某分支；配合 mbarrier 做 producer-consumer
- **Blackwell** 特性目前主要在 template 层用；关注 `tcgen05_*`、`tmem`、`2sm` 关键字

下一章 [第 14 章](./14_quantization_fp8_mxfp_int4.md) 讲另一个 TileLang 里被大量优化但正文没系统讲的话题：**量化**——FP8 / MXFP / INT4 dequant 是怎么做进 kernel 的。
