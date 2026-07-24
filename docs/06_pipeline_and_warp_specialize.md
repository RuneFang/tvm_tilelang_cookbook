# 第 6 章 · 软件流水 + Warp Specialization 深挖

> **TL;DR**：本章解剖 TileLang 里最"魔法"的两个 pass ——
> **`InjectSoftwarePipeline`**（把 `T.Pipelined` 变成真正的多缓冲循环）和
> **`ProducerConsumerWarpSpecialized`**（把一个 threadblock 里的 warp 分成"搬砖工"和"算力工"）。
>
> 你会理解：
> 1. **software pipeline** 为什么要给 shared 复制"N 份 slot"，`num_stages` 到底是什么维度
> 2. **multi-version buffer** 怎么把 `A_shared[i][j]` 改写成 `A_shared[stage][i][j]`
> 3. **mbarrier phase counter** 是什么、为什么它出错就会导致数值静默错乱
> 4. **Warp Specialization** 为什么要把一个 threadblock 拆成两拨 warp，producer / consumer 各干嘛
> 5. **provenance vs syntax**：多个 pass 之间传递"这是编译器生成的"语义为什么只能靠专属 intrinsic 标签
>
> **本章会读到的真实源码**：
> - [`src/transform/inject_pipeline.cc`](../../src/transform/inject_pipeline.cc)（146 KB，最大的 pass 之一）
> - [`src/transform/pipeline_planning.cc`](../../src/transform/pipeline_planning.cc)
> - [`src/cuda/transform/multi_version_buffer_rewriter.cc`](../../src/cuda/transform/multi_version_buffer_rewriter.cc)
> - [`src/cuda/transform/producer_consumer_ws.cc`](../../src/cuda/transform/producer_consumer_ws.cc)（102 KB）
> - [`src/cuda/transform/fuse_mbarrier_arrive_expect_tx.cc`](../../src/cuda/transform/fuse_mbarrier_arrive_expect_tx.cc)
> - [`src/cuda/transform/lower_shared_barrier.cc`](../../src/cuda/transform/lower_shared_barrier.cc)
> - [`src/op/builtin.h`](../../src/op/builtin.h)（`mvb_stage_index` / `mbarrier_wait_parity` 等 intrinsic 定义）
>
> **前置**：读完第 3 章（DSL 里 `T.Pipelined` 是啥）、第 4 章（怎么写 pass）、第 5 章（pipeline 里 pass 的顺序）。

---

## 6.1 从 `T.Pipelined` 到 GPU 上"读算重叠"的完整链路

先看一眼**没有** pipeline / warp specialize 时的 matmul 核心循环：

```
for ko in range(K // block_K):
    load A[ko] → shared               # 慢，等 memory
    load B[ko] → shared               # 慢，等 memory
    __syncthreads()                    # 大家一起等
    C_local += A_shared @ B_shared     # 快，跑 Tensor Core
    __syncthreads()
```

**GPU 是个"读数据比算得慢很多"的机器**。上面这个写法 100% 的时间里，Tensor Core 只在其中一小段
（`gemm` 那步）忙碌，剩下都在等 load。要想榨干 Tensor Core，就必须让**"读下一批"和"算这一批"重叠**。

TileLang 通过**两层机制**做重叠：

```
┌──────────────────────────────────────────────────────────────────────┐
│  第 1 层：软件流水（software pipeline）                                │
│  --------------------------------------------------                  │
│  展开 K 循环，给 A_shared / B_shared 复制 N 份"slot"，                 │
│  同一个 warp 里 issue "预取 slot k+2" 后紧跟着 "算 slot k"，          │
│  靠 mbarrier / cp.async pipeline 硬件机制隐藏延迟。                   │
│                                                                      │
│  在 TIR 层就能做完（不依赖 Hopper 特有硬件），SM70 就能用。            │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  第 2 层：Warp Specialization（"角色分工"，Hopper+ 才有）              │
│  --------------------------------------------------                  │
│  把 256 个 threads 拆成 2 组 warp group（每组 128 线程）：             │
│     - Producer WG：只负责发起 TMA load（异步 DMA）                     │
│     - Consumer WG：只负责跑 WGMMA + epilogue                          │
│  两组之间靠 mbarrier 同步（"数据到了没？"）。                          │
│                                                                      │
│  好处：Producer 的调度器和 Consumer 的调度器不再互相"卡"，进一步压缩   │
│        Tensor Core 空闲时间。                                          │
└──────────────────────────────────────────────────────────────────────┘
```

> **先认全上图里的几个词**（后面整章都要用）：
> - **warp** = 32 个线程的硬件调度单位；**warp group（WG）** = **4 个 warp = 128 个线程**（NVIDIA 在 Hopper 上固定的分组）。所以上图"256 个 threads 拆成 2 组"= 每 128 线程一个 WG，一个当 producer、一个当 consumer。
> - **Producer WG / Consumer WG** = 把这些 warp group 分成"专门搬数据的"和"专门算的"两拨，各司其职。
> - **TMA** = Hopper 的硬件异步搬运引擎（Tensor Memory Accelerator）；**DMA** = 直接内存访问（不占用计算单元的搬运）。
> - **WGMMA** = warp-group MMA，即"一整个 warp group 一起做"的 Tensor Core 矩阵乘指令（Hopper 版）。
>
> 这些概念第 13 章会更系统地讲，这里先记住"WG=128 线程、producer 搬、consumer 算、mbarrier 通知到货"就够读下去了。

> 💡 **常见误解**：软件流水 ≠ warp specialization。
> - 只用软件流水（`num_stages=3`、SM80）：**同一批 warp** 同时负责 issue load 和 issue gemm，靠硬件的 issue 队列做重叠。
> - 加上 warp specialize（Hopper+）：**不同的 warp** 各干一件事，重叠更彻底。
> 你写 `T.Pipelined(K, num_stages=3)` 时得到的是 **第 1 层**；`T.use_swizzle` / target=SM90 + 未 disable 时，pipeline 之后**再套一层** warp specialize，得到 **第 1+2 层**。

## 6.2 第一步：`PipelinePlanning` —— 规划师

**它在做什么？**

它是一个"看菜下饭"的 pass：**把用户在 `T.Pipelined` 里可能没显式给的 `order` / `stage` 注解补齐**。
用户可以只写 `num_stages=3`，pass 会分析循环体内的每条语句（load / gemm / store），
按依赖关系给它们排出 `software_pipeline_stage` 和 `software_pipeline_order` 两个数组注解。

**输入**：

```
For(ko, extent=K/bK, kind=Serial, annotations={
    "software_pipeline_num_stages": 3,
})
  body = [ T.copy(A, A_shared),  T.copy(B, B_shared),  T.gemm(...) ]
```

**输出**：

```
For(ko, extent=K/bK, kind=Serial, annotations={
    "software_pipeline_stage":  [0, 0, 1],   # 前两条 copy 在 stage 0（读），第三条 gemm 在 stage 1（算）
    "software_pipeline_order":  [0, 1, 2],   # 语句执行顺序
})
  body = ... (body 本身不变)
```

> 📌 `software_pipeline_stage` 的取值可以对着源码测试核对（如
> `testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py`）：
> "两条 copy + 一条 gemm" 这种最常见的循环体，planning 出来就是 `[0, 0, 1]`。
> 注意 stage 数组表达的是"每条语句排在流水线第几档"，它和 `num_stages`（缓冲深度）是两个维度，
> 不要混为一谈。

**源码入口**：[`src/transform/pipeline_planning.cc`](../../src/transform/pipeline_planning.cc)。
这个 pass 有 49KB 大，主要复杂度在**依赖分析**（哪个 buffer 被谁读、谁写，从而决定哪句能提前）。
第 5 章说过它属于「阶段二通用 lowering」的一部分。

> 💡 **概念卡：`stage` vs `order`**
> - `order[i]` = "在循环体里执行的第 i 条语句，原本是 body[order[i]]"。相当于允许 pass 换序。
> - `stage[i]` = "这条语句在**流水线的哪个 stage** 发起"。stage 0 就是"最早发起、结果最晚使用"的那一档。
> - `num_stages` = 允许"同时在飞"的批次数量 = shared buffer 要复制几份 slot。

## 6.3 第二步：`InjectSoftwarePipeline` —— 施工队

它做的事**信息量最大**——把一个"带注解的普通循环"物理地展开成"prologue + 稳态循环 + epilogue"。

**输入**（上面 planning 的输出）→ **输出**（伪码）：

```
# prologue: 只 issue load，不算
for k in range(num_stages - 1):        # 也叫 "warm-up"
    load A[k] → A_shared[k]
    load B[k] → B_shared[k]
    commit_pipeline_barrier()

# steady state: load(k+2) 和 gemm(k) 交错
for k in range(K/bK - (num_stages - 1)):
    wait_pipeline_barrier(k)           # 等 slot k 的 load 完成
    gemm(A_shared[k % 3], B_shared[k % 3], C_local)
    load A[k + 2] → A_shared[(k+2) % 3]
    load B[k + 2] → B_shared[(k+2) % 3]
    commit_pipeline_barrier()

# epilogue: 只算，不再 load
for k in range(K/bK - (num_stages - 1), K/bK):
    wait_pipeline_barrier(k)
    gemm(A_shared[k % 3], B_shared[k % 3], C_local)
```

**这就是"3-stage 软件流水"的全貌**。有几个关键细节：

1. **`% 3` 那一步不是这个 pass 干的**，见下节 6.4——它是 `MultiVersionBufferRewriter` 干的。
2. **每个 stage 之间靠 pipeline barrier 同步**，SM80 上就是 `cp.async.commit_group` + `cp.async.wait_group`，
   Hopper 上升级为 **mbarrier**。
3. 稳态循环体的**长度**变了：从 `K/bK` 变成 `K/bK - (num_stages - 1)`。prologue 和 epilogue 各切走了几拍。

**源码入口**：[`src/transform/inject_pipeline.cc`](../../src/transform/inject_pipeline.cc)（146 KB）。
这里面有 30+ 个内部 helper，第一次读可以先只看顶层 `InjectPipeline::Rewrite`
和 `SoftwarePipelineRewriter::VisitStmt_(ForNode*)` 两个入口。

## 6.4 第三步：`MultiVersionBufferRewriter` —— 起 N 层楼

**问题**：上一步产出的伪码里已经有 `A_shared[k % 3]` 这样的多缓冲写法了——**这个 `% 3` 是谁加的？**

答案是 `MultiVersionBufferRewriter`。它做的事一句话说清：

> 找到"在 pipeline 循环里被读写、需要多缓冲的 shared / barrier buffer"，
> 把它们的**形状扩一维**（`(M, K)` → `(num_stages, M, K)`），
> 并在**每一次 buffer 访问**前面塞一个 stage 索引（`A_shared[i][j]` → `A_shared[k % N][i][j]`）。

**源码**：[`src/cuda/transform/multi_version_buffer_rewriter.cc`](../../src/cuda/transform/multi_version_buffer_rewriter.cc)，
入口 `ApplyMultiVersionBufferRewriter(PrimFunc f)` 在 919 行。

### 关键代码片段（6.9 节复盘 phase 对齐陷阱时要用到）

在 `VisitStmt_(const ForNode *op)` 里（约 718 行）。先说明代码里的 `loop_stack_`：它是**当前嵌套循环栈**，每个元素是一对 `(循环变量, extent)`——`.first` 取循环变量、`.second` 取循环长度（extent）。下面这段就是把多层嵌套循环的下标"压平"成一个一维索引 `linear_index`：

```cpp
PrimExpr linear_index = loop_stack_[0].first;          // 最外层 pipeline loop 的迭代变量
for (size_t i = 1; i < loop_stack_.size(); ++i) {
  linear_index = linear_index * loop_stack_[i].second + loop_stack_[i].first;
}
PrimExpr raw_version_index = FloorMod(linear_index, num_stages);   //  k % N
version_index_ =
    Call(raw_version_index->dtype, mvb_stage_index(), {raw_version_index});
    //  ^^^^^ 关键：包一层 mvb_stage_index() intrinsic 作为 "provenance tag"
parity_cycle_ = FloorMod(FloorDiv(linear_index, num_stages), 2);   //  (k / N) % 2
```

三个变量对应三个概念：

| 变量 | 表达式 | 用途 |
|---|---|---|
| `raw_version_index` | `k % N` | 选 slot |
| `version_index_`  | `mvb_stage_index(k % N)` | **打了 provenance tag** 的 slot 索引 |
| `parity_cycle_`   | `(k / N) % 2` | mbarrier 的 **phase**，见 6.5 |

**为什么 `version_index_` 要包一层 `mvb_stage_index()` intrinsic？** 这是本章最重要的设计模式：
它是**给下游 pass 看的 provenance 标记**。见 6.9。

### 生成的 IR 长这样

pipeline 循环内的原始表达式：`A_shared[i, j]`（`i, j` 是 tile 内坐标）。
经过 rewriter 后变成：

```
A_shared_v[ mvb_stage_index(k % 3), i, j ]
             ─────────┬─────────
             这个 Call 就是编译器留给下游的 "provenance"
```

同一次访问的 `A_shared_v` 声明也从 `Buffer(shape=[M, K])` 扩成 `Buffer(shape=[3, M, K])`。

## 6.5 关键概念：mbarrier 的 phase counter（跨 tile 最容易出错的地方）

### 6.5.1 mbarrier 是什么

CUDA 里 **mbarrier**（memory barrier）是一种硬件同步原语（Hopper 起支持得最完整）。
它有两个操作：

- `mbarrier.arrive.expect_tx()` —— "我保证之后会往这个 barrier 上写 tx 字节的数据"
- `mbarrier.wait_parity(bar, phase)` —— "阻塞，直到 `bar` 的 phase 翻转到 `phase`"

**它比 `__syncthreads()` 强的点**：能精确表达"某一批 TMA load 完成"，而不是"全体线程到达某处"。

### 6.5.2 什么是 phase？

一个 mbarrier 有一个内部计数器，每凑够指定次 arrive 就"翻转"一次。**翻转前后 phase 各是 0 和 1**。
一个 pipeline 循环里同一个 mbarrier 会被反复使用：

```
iter k=0: arrive → phase=1, wait(phase=0) 通过？不通过（初始也是 0，等翻到 1）
iter k=1: arrive → phase=0, wait(phase=1) 通过
iter k=2: arrive → phase=1, wait(phase=0) 通过
...
```

也就是说 **wait 的期望 phase 是 `(k) % 2`**。

### 6.5.3 但如果加了 `num_stages` 呢？

Multi-version 之后，同一个 mbarrier **每 `num_stages` 次迭代**才被重用一次
（每次 iter 用的是 `k % N` 那个 slot 上的 barrier）。所以 wait 期望的 phase 是：

```
       (k / num_stages) % 2       ← 就是 6.4 那段代码里 parity_cycle_ 的定义
```

**这行公式如果算错、或者在 persistent kernel / warp specialize 场景下**没有对齐 `num_stages` 的倍数，
Consumer 就会等错 phase，要么死锁，要么读到还没 arrive 的旧 slot——**数值静默出错**。
6.9 节会把这条陷阱的完整因果链拆开讲。

## 6.6 第四步：`ProducerConsumerWarpSpecialized`

**它做什么**（简称 PCWS）：把一个 threadblock 里的 warp 分成两组：

```
Threadblock (128 threads = 4 warps):

  Producer WG (1 warp = 32 threads):
    ┌────────────────────────────────────┐
    │ for k in pipeline_range:           │
    │   TMA load A[k] → shared[k%N]      │
    │   TMA load B[k] → shared[k%N]      │
    │   mbarrier.arrive(bar_full[k%N])   │  ← 通知 consumer "数据到了"
    │   mbarrier.wait(bar_empty, ...)    │  ← 等 consumer "slot 用完了"
    └────────────────────────────────────┘

  Consumer WG (3 warps = 96 threads):
    ┌────────────────────────────────────┐
    │ for k in pipeline_range:           │
    │   mbarrier.wait(bar_full, ...)     │  ← 等 producer "数据到了"
    │   wgmma(shared[k%N], shared[k%N])  │  ← 算
    │   mbarrier.arrive(bar_empty)       │  ← 通知 producer "slot 我用完了"
    └────────────────────────────────────┘
```

**源码入口**：[`src/cuda/transform/producer_consumer_ws.cc`](../../src/cuda/transform/producer_consumer_ws.cc)。
入口 pass 函数 `ProducerConsumerWarpSpecialized()` 在 2780 行。

**注意它内部会顺带调 `ApplyMultiVersionBufferRewriter`**（2807 行）—— 也就是说这个 pass 是
"warp specialize + multi version"的**组合**：先把 buffer 版本化，再切 producer / consumer。

### 6.6.1 disable 的方法（调试利器）

`PassContext` 里有 `tl.disable_warp_specialized`（[`src/op/builtin.cc:24`](../../src/op/builtin.cc)）。
Python 侧只要：

```python
kernel = matmul.compile(..., pass_configs={"tl.disable_warp_specialized": True})
# 或者用环境变量：TL_DISABLE_WARP_SPECIALIZED=1
```

这就退回到"**只有软件流水、没有 warp specialize**"的模式——**测试时对比这两种模式的数值**，
是定位"WS pass 是不是搞错了"的黄金方法（6.9.5 节会给出一个具体的对比测试模板）。

## 6.7 phase 表达式的 provenance 陷阱（本章最关键的一节）

回顾 6.4：`MultiVersionBufferRewriter` 把编译器生成的 slot index 用 `mvb_stage_index()`
**包起来**：

```cpp
version_index_ = Call(..., mvb_stage_index(), {FloorMod(k, num_stages)});
```

同时它自己会去改 `mbarrier_wait_parity` 的 parity 参数（约 836 行）：

```cpp
if (call->op.same_as(mbarrier_wait_parity()) && parity_cycle.defined()) {
  ...
  new_args.Set(1, new_parity);   // new_parity = (k / N) % 2
  return Call(..., mbarrier_wait_parity(), new_args, ...);
}
```

**下游** 的 `ProducerConsumerWarpSpecialized` 需要**再一次**改写这些索引和 parity
（因为它拆 producer / consumer 时会引入自己的**独立** phase counter）。**问题是**：它怎么识别
"这个 `FloorMod(x, N)` 是编译器生成的 slot 索引" 而不是"用户自己写的 `if (k+1)%N == 0`"？

**错误做法**：语法模式匹配——看到 `FloorMod(?, num_stages)` 就当成 slot 索引改写。
**这样会误伤用户源代码里合法出现的 `k%N`**——编译器根本分不清它是自己生成的还是用户写的。

**正确做法（现在的做法）**：识别 `Call(mvb_stage_index, ...)` 这一个 provenance tag。见
`MVBStageIndexReplacer`（[`producer_consumer_ws.cc:120`](../../src/cuda/transform/producer_consumer_ws.cc)）：

```cpp
class MVBStageIndexReplacer : public StmtExprMutator {
public:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(mvb_stage_index())) {          // ← 精确认这个 intrinsic
      ICHECK_EQ(op->args.size(), 1U);
      if (replacement_.defined()) {
        return tvm::cast(op->dtype, replacement_.value());
      }
      return VisitExpr(op->args[0]);                  // 剥掉外层 tag，回到原表达式
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};
```

> 💡 **概念卡：provenance vs syntax**
> - **provenance-based**：判断"这个节点是不是我上一个 pass 生成的"。工具是 **专属的 intrinsic 标签**（`mvb_stage_index`）。
> - **syntax-based**：判断"这个节点长得是不是像我要找的形状"。工具是 pattern match。
>
> 对**用户可写的语法子集**，永远用 provenance。对**纯编译器内部约定的形式**（比如某 pass 生成的
> `AttrStmt("software_pipeline_stage", ...)`），syntax 就足够，但意图仍然是"识别我自己生成的东西"。

在**这个 pass 的最后**，还必须**兜底剥掉所有 `mvb_stage_index` marker**，否则它会泄漏到 codegen——
后端不认识这个 intrinsic，会崩。这段逻辑在 pass 出口无条件跑一次 `MVBStageIndexReplacer::Replace(body, NullOpt)`
（`NullOpt` 表示"不替换成别的、就把 tag 剥掉"）。

## 6.8 亲手看一眼这些"多阶段"IR

```python
import tilelang, tilelang.language as T

@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int, num_stages: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by*block_M, ko*block_K], A_shared)
            T.copy(B[ko*block_K, bx*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*block_M, bx*block_N])
    return C

cfg = dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, num_stages=3)

# 1. 只启软件流水，禁 warp specialize：能看到"多缓冲后的 shared[stage][i][j]"
art_ws_off = tilelang.lower(
    matmul.get_tir(**cfg),
    target="cuda",
    pass_configs={"tl.disable_warp_specialized": True},
)
print("=== [WS OFF] device_mod ===")
print(art_ws_off.device_mod.script())

# 2. 完整 pipeline（默认，SM90+）：能额外看到 producer / consumer 分组、mbarrier_arrive/wait
art_ws_on = tilelang.lower(matmul.get_tir(**cfg), target="cuda")
print("=== [WS ON] device_mod ===")
print(art_ws_on.device_mod.script())
```

**你在 WS OFF 版本里应该看到的关键字**：`A_shared_v[<stage_expr>][...]`、`cp.async`、`cp.async.wait_group`。
**你在 WS ON 版本里应该额外看到的关键字**：`mbarrier_wait_parity(...)`、`producer_phase_cnt` /
`consumer_phase_cnt` 之类的 phase counter local 变量，以及 `if warp_group_id == 0` / `== 1` 的角色分派。

## 6.9 一个真实陷阱：Persistent + 深流水下的 K-trip 与 phase counter 对齐

现在把所有前面章节的概念串成一条因果链，看清"phase 没对齐"这类 bug 是怎么发生的。

> **先约定一个记号**：本节反复用到的 **"K-trip"** 指 **K 维外层循环的迭代次数**，即 `K-trip = ceildiv(K, block_K)`（trip count = 循环跑的圈数，是编译器里的常见说法；"K-trip" 是本书为了简写起的组合词）。例如 `K=1024, block_K=32` → K-trip = 32。下文说"K-trip 不是 num_stages 的倍数"，就是指这个圈数不能被流水级数整除。

### 6.9.1 表象

一个使用 `T.Persistent` + `T.Pipelined(K, num_stages=N)` + WS 的 GEMM：

- **当 `K/block_K` 是 `num_stages` 的倍数**（例如 K=1024、block_K=32、num_stages=4 → K-trip=32）：结果对
- **当 `K/block_K` **不是** `num_stages` 的倍数**（例如 K-trip=30、num_stages=4）：**前几个 tile 对，后续 tile 数值静默出错**

### 6.9.2 因果链

```
┌──────────────────────────────────────────────────────────────────┐
│  ① T.Persistent：一个 threadblock 循环处理多个 (i, j) tile        │
│     → 意味着 K 循环会被"跑很多次"                                  │
│                                                                  │
│  ② 每个 tile 内的 K 循环用 num_stages 做 pipeline，每次 iter 用   │
│     的 mbarrier phase = (k / num_stages) % 2                     │
│                                                                  │
│  ③ 一个 tile 结束时，如果 K-trip **不**是 num_stages 的倍数，    │
│     mbarrier 内部的 phase 会停在**奇数位**。                       │
│                                                                  │
│  ④ 进入下一个 tile 时，Consumer 认为自己应该从 phase=0 开始等，    │
│     但 mbarrier 实际停在 phase=1（stale）。                        │
│                                                                  │
│  ⑤ Consumer 的 wait 立刻通过（因为 phase 已经是 1 了），但 slot    │
│     里的数据其实还没被 Producer 更新——读到上一个 tile 的残留。     │
│                                                                  │
│  ⑥ 结果：数值静默错乱。前面的 tile 是对的（还没触发 K-trip 不齐），  │
│     后续 tile 会带着"错位的 phase"继续累加错误。                    │
└──────────────────────────────────────────────────────────────────┘
```

### 6.9.3 修复的原则

有两条路：

**路 A（错误）**：在 PCWS pass 里，用**语法匹配** `FloorMod(loop_var, num_stages)` 去发现和改写 slot 索引 / phase。
→ 会误伤用户合法源码里写的 `if (k+1) % 2 == 0` 之类：编译器生成的 slot 索引和用户手写的取模表达式长得一模一样，无法区分。

**路 B（正确 & 现在的做法）**：
- MVB 阶段把它生成的每一个 slot 索引都包一层 `mvb_stage_index()` intrinsic 作为**明确的 provenance tag**
- PCWS 阶段只用 `CallNode + op.same_as(mvb_stage_index())` 匹配这个 tag，`syntax` 上什么都不猜
- PCWS 结束前无条件跑 `MVBStageIndexReplacer` 把 tag 全剥干净，防止泄漏到 codegen
- 每个 tile 结束时，**主动把 phase counter 对齐到 `num_stages` 的倍数**（drain 掉多出来的 phase），
  下一个 tile 从 phase=0 干干净净地开始

### 6.9.4 为什么 K-trip 是 num_stages 倍数时表面上"对"

因为 `K_trip % num_stages == 0` 时，`(K_trip / num_stages) % 2` 天然回到 0，phase counter
恰好清零。**这不是修好了、只是掩盖了 bug**——它掩盖了"没有 drain phase counter"这个真正问题。

### 6.9.5 回归测试的写法

**只对比"数值 vs 数学 oracle"是不够的**——loose tolerance 会盖住细微的错乱。

- **wsoff-vs-ws diff**：用 `tl.disable_warp_specialized=True` 编一份参考 kernel，再编 WS 版，两个直接 `torch.allclose` 差一 ulp。差值大就报警。
- **生成源码指纹**：对生成的 CUDA 源码字符串**扫黑名单**——比如 `"producer_phase_cnt[0] %"` 只在**旧 buggy 代码路径**下才会出现，
  一旦出现就断言失败。这样能把 "IR 是否被误替换" 这件事从"隐晦数值差"提升为"确定性的字符串断言"。

> **术语说明**：本书后面几处把"对生成源码 grep 特征字符串来断言"这种手法简称为 **"硬签名"（hard signature）/ 硬 grep 断言**。它不是行业标准术语——业界更常见的叫法是 **snapshot / golden-file test（快照测试）** 或直接说"对生成源码做字符串断言"。本书用"硬签名"只是取其"确定性、非黑即白"之意，读者知道它指的就是这类字符串断言即可。

回归测试模板：

```python
def test_ws_persistent_misaligned_k_matches_wsoff_reference():
    cfg = dict(M=256, N=256, K=30 * 32, block_M=128, block_N=128, block_K=32,
               num_stages=4, use_persistent=True)

    k_ws_off = matmul.compile(**cfg, pass_configs={"tl.disable_warp_specialized": True})
    k_ws_on  = matmul.compile(**cfg)

    # 1) 硬签名断言：确认 WS 确实启动了、且没走到 buggy 分支
    src = k_ws_on.get_kernel_source()
    assert "mbarrier_wait_parity" in src               # WS 确实开启
    assert "producer_phase_cnt[0] %" not in src        # 没走到旧 buggy 代码路径

    # 2) 数值对比：ws-on 应该和 ws-off 严格一致
    a = torch.randn(cfg["M"], cfg["K"], device="cuda", dtype=torch.float16)
    b = torch.randn(cfg["K"], cfg["N"], device="cuda", dtype=torch.float16)
    c_off = k_ws_off(a, b)
    c_on  = k_ws_on(a, b)
    torch.testing.assert_close(c_on, c_off, rtol=0, atol=0)   # bit-exact
```

## 6.10 本章要带走的四件事

1. **软件流水 = "开 N 份 slot 让 load 和 gemm 重叠"**，靠 `MultiVersionBufferRewriter` 把 buffer 加一维、
   靠 `InjectSoftwarePipeline` 展开 prologue / steady / epilogue。
2. **mbarrier phase counter = `(k / num_stages) % 2`**，`num_stages` 不整除 K-trip 会导致 phase 漂移，
   跨 tile 静默出错。
3. **多 pass 之间传递"这是编译器生成的" 语义，只能靠 provenance**（专属 intrinsic 作 tag），
   **不能靠语法模式**。这是 TileLang / TVM 系统里做 pass 组合的**根本规则**。
4. **正确性测试要 wsoff-vs-ws + 硬签名双保险**——loose tolerance 会盖住 phase 漂移这类 bug，
   源码字符串扫黑名单能把"意图是否达成"从数值域提升到断言域。

---

下一章 [第 7 章 · Layout 系统与 Fragment](./07_layout_and_fragment.md)：
我们从"多缓冲"下探到"每个 tile 内部 32 个线程怎么瓜分数据"——
Fragment / Layout / Swizzle 到底描述的是什么物理事实。
