# 第 5 章 · Lowering Pipeline 逐 pass 巡礼（CUDA 后端）

> **TL;DR**：`tilelang.lower(pf, target="cuda")` 内部把 TIR 递给一段**顺序执行的 ~50 个 pass**。
> 这一章按**源码里真实的调用顺序**，把每个 pass 一句话讲清"它接手时看到什么、离开时留下什么"，
> 让你面对一段陌生 IR 时能立刻猜出"这是走到第几个阶段的样子"。
>
> **本章唯一权威来源**：[`tilelang/cuda/pipeline.py`](../../tilelang/cuda/pipeline.py)
> 里的 `CUDAPassPipelineBodyPrologue` + `CUDAPassPipelineBody` 两个函数。
> **本章不虚构任何 pass 名**——如果你 grep 本章某个 pass 名找不到，请告诉我。
>
> **前置**：[第 4 章](./04_pass_system.md)（知道 pass 是什么、`PassContext` 是什么）。

---

## 5.1 大局：pipeline 长成 4 段

CUDA 后端的完整 lowering 走这条路（每个方块是一段"主题相近的 pass 群"）：

```
      ┌───────────────────────────────────────────────────────────────┐
      │ 你的 PrimFunc / IRModule (刚从 tilelang DSL 解析出来)           │
      └───────────────────┬───────────────────────────────────────────┘
                          ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 【段 A · Prologue】 CUDAPassPipelineBodyPrologue                │
   │   目的：把 tile-op 高层 IR 变成"已经知道 layout、已经排好      │
   │   pipeline、tile-op 已经展开到 T.Parallel + BufferStore" 的 IR │
   └────────────────────────────┬───────────────────────────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 【段 B · 内存与 barrier 落地】                                  │
   │   目的：LowerSharedTmem / LowerSharedBarrier / MBarrier fusion  │
   │        + buffer allocation placement                           │
   └────────────────────────────┬───────────────────────────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 【段 C · 标准化 & 优化】                                        │
   │   FlattenBuffer / VectorizeLoop / StorageRewrite / UnrollLoop  │
   │   / Simplify / RemoveNoOp / HoistIfThenElse ...                │
   └────────────────────────────┬───────────────────────────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ 【段 D · Host/Device 拆分 + CUDA 收尾】                          │
   │   AnnotateDeviceRegions → SplitHostDevice → LowerLDGSTG /       │
   │   LowerHopperIntrin / MergeSharedMemoryAllocations /            │
   │   ThreadSync / MakePackedAPI / LowerDeviceKernelLaunch          │
   │   → PersistThreadblock                                          │
   └────────────────────────────┬───────────────────────────────────┘
                                ▼
                    codegen（第 8 章讲）
```

对照第 1 章那张"6 阶段流程图"：**段 A 就是阶段二的前半段**、**段 B/C 就是阶段二的后半段**、**段 D 就是阶段三**。
本章接下来把这 4 段拆开、pass 一条条列。

---

## 5.2 段 A · Prologue：从 tile-op 走到低层 TIR

以下每一条**都能在 [`tilelang/cuda/pipeline.py`](../../tilelang/cuda/pipeline.py) 的
`CUDAPassPipelineBodyPrologue` 函数里逐行对上**（顺序也一致）：

### 5.2.1 前置整理（"进厂洗一遍"）

| # | Pass | 作用（一句话） |
|---|---|---|
| 1 | `tirx.transform.BindTarget(target)` | 给每个 PrimFunc 挂上 `target = "cuda -arch=..."` 属性，后面 pass 才能 dispatch |
| 2 | `MaterializeKernelLaunch` | 把 `T.Kernel(gx, gy, threads=...)` 展开成一坨 `thread_binding` For（详见 [第 3 章 3.2](./03_tilelang_dsl.md#32-tkernel)） |
| 3 | `LetInline`（可选） | `pass_config` 里 `tl.force_let_inline=True` 时把所有 `Let` 展平；调试时偶尔用 |
| 4 | `AddWrapperForSingleBufStore` | 单条 `BufferStore` 外面套一层 SeqStmt/Block，方便后续 pass 一致处理 |
| 5 | `LegalizeNegativeIndex` | `A[-1, j]` 之类的负下标改写成规范形式 |
| 6 | `VerifyParallelLoop`（可选） | `pass_config` 里 `tl.enable_race_check=True` 时检查 `T.Parallel` 内的写冲突 |
| 7 | `InjectAssumes` | 在 body 顶部塞 `T.assume(...)`，帮 prover 简化边界（例如 M % block_M == 0） |
| 8 | `Simplify` | tilelang 增强版的 `arith::Analyzer` 化简，比 tvm 的官方 Simplify 更狠 |
| 9 | `LayoutReducer` | 给 reduction 用的中间累加器（`T.alloc_var` / reducer）挂上默认 layout |

### 5.2.2 高层调度：Warp Specialize + Pipeline Planning（**核心中的核心**）

| # | Pass | 作用 |
|---|---|---|
| 10 | `ProducerConsumerWarpSpecialized` | **只在 CUDA + TMA 目标 + 未 disable 时跑**。把 tile-op IR 里的 `copy` / `gemm` 按"producer warp（干搬运）/ consumer warp（干计算）"分角色，插入 mbarrier arrive/wait。**第 6 章会详细解剖它** |
| 11 | `LowerBlackwell2SM` | Blackwell (SM100+) 的 2-SM TCGEN5MMA lowering，非目标硬件上无副作用 |
| 12 | `IfStmtBinding` | 把"没 else 的 if"包成规范 SeqStmt，方便下一步 pipeline planning 抽 body |
| 13 | `PipelinePlanning` | 看 `T.Pipelined(..., num_stages=k)` 注解，给循环体每条语句排 stage/order（**软件流水调度**，第 6 章讲） |
| 14 | `InjectSoftwarePipeline` | 拿着 planning 结果，把原来 `for ko in Pipelined(...)` 单循环**改写成三段式**：prologue（开头几圈只预取数据）/ steady（稳定期边取边算）/ epilogue（末尾几圈只算不取），配上 multi-version buffer。这三段的直觉第 6 章 6.3 会画图讲 |
| 15 | `Simplify` | 打扫刚生成的一堆 index |

> 💡 **概念卡：为什么 Warp Specialize 要放这么靠前**
> 它必须**看到**原始的 `T.copy` / `T.gemm` 这类高层 tile-op 才能识别 producer/consumer。
> 一旦 `LowerTileOp`（下一节的第 17 号 pass）把 `T.copy` 展开成 `T.Parallel + cp.async` 就再也认不出来了。
> 所以它坚决要**排在 LowerTileOp 之前**。同样的道理，`PipelinePlanning` 也在 `LowerTileOp` 之前。

### 5.2.3 Layout 推断 + tile-op 展开

| # | Pass | 作用 |
|---|---|---|
| 16 | `LayoutInference` | 给每个 `Buffer`（尤其 `local.fragment` 和 `shared.dyn`）**推断出一个 layout**（fragment tile 分布 / swizzle pattern），是第 7 章主角 |
| 17 | `LowerTileOp` | **本 pipeline 中最"重"的 pass**。把所有 `Call("tl.tileop.copy")` / `Call("tl.tileop.gemm")` / `Call("tl.tileop.fill")` 展开成具体的 mma / cp.async / TMA + T.Parallel。见 [`src/transform/lower_tile_op.cc`](../../src/transform/lower_tile_op.cc) |

### 5.2.4 CUDA 特定 + vectorize 前打杂

| # | Pass | 作用 |
|---|---|---|
| 18 | `LowerL2Persistent` | 处理 L2 persistent cache 提示（`cudaAccessPropertyPersisting`），非 SM80+ 时是 no-op |
| 19 | `DecoupleTypeCast` | 把 `float16 → float32` 之类的 cast 从循环体里拎出来，方便后面向量化 |
| 20 | `LegalizeVectorizedLoop` | `T.Vectorized(n)` 循环合法性检查/改写 |
| 21 | `LegalizeSafeMemoryAccess` | 边界检查：给可能越界的访问加 `if (idx < N) ...` |
| 22 | `LowerAccessPtr` | 把前端的 `T.address_of` / pointer metadata 转成 `tvm_access_ptr` |
| 23 | `Simplify` | 又一次打扫 |
| 24 | `HoistNonRestrictParams` | 把 block 里挂的 `T.block_attr({...})` 挪到 PrimFunc.attrs |

> ⚠️ 段 A 的**出口 IR** 是我们后面调试最常打印的形态：
> - `T.copy` / `T.gemm` 都已消失，换成 `T.Parallel + BufferStore` / mma 内建
> - `T.Pipelined` 已被展开成 prologue/steady/epilogue 三段
> - 每个 `Buffer` 都已经有 layout
>
> 打印方式（真实可跑）：
> ```python
> import tilelang, tilelang.language as T
> from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
> from tvm.target import Target
> pf = matmul.get_tir(**cfg)                        # 见第 1 章
> mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})
> tgt = Target("cuda")
> out = CUDAPassPipelineBodyPrologue(mod, tgt)
> print(out.script())
> ```
> （`CUDAPassPipelineBodyPrologue` 是 tilelang.cuda.pipeline 模块的公开函数）

---

## 5.3 段 B · 内存与 barrier 落地

对应 `CUDAPassPipelineBody` 里 `LowerSharedTmem` 到 `HoistGlobalBufferAllocations` 那一段：

| # | Pass | 作用 |
|---|---|---|
| 25 | `LowerSharedTmem` | Blackwell TCGEN5 的 shared → tmem 拷贝落到具体 slot |
| 26 | `PlanAndUpdateBufferAllocationLocation` | 决定每个 `Allocate` 的最终位置——把 `alloc_shared` 提升到 kernel 顶部、把 `alloc_local` 就地 |
| 27 | `LowerSharedBarrier` | 把 `T.alloc_barrier()` 落到 CUTLASS `Barrier` 类型 + `tl_shuffle_elect` + `fence_barrier_init`（**仅 SM90+**；如果目标 < SM90 且有 `T.alloc_barrier`，pipeline 会**在这里 raise ValueError**）|
| 28 | `FuseMBarrierArriveExpectTx` | **只当 `tl.has_tma == True`** 时跑：把 `mbarrier.arrive` 和紧邻的 `expect_tx` 合成一条 PTX |
| 29 | `HoistGlobalBufferAllocations` | 把 `T.alloc_global` 全部提到函数顶（workspace） |

> 💡 **概念卡：mbarrier / expect_tx**
> Hopper 引入的 mbarrier 是 hardware 级的**多线程等待器**，寿命跨越 kernel 内多次 wait。
> `arrive` = 告诉 barrier "我到了"，`expect_tx = N` = 告诉 barrier "接下来会有 N 字节 TMA 到货，到齐才算数"。
> 把它俩合成一条 PTX 能少一次 barrier state 更新。第 6 章会讲 mbarrier phase counter，这里知道有这么个东西就够了。

---

## 5.4 段 C · 标准化 & 优化

这段的目标：**从"结构化 IR"变成"线性、扁平、可向量化的 IR"**，让 codegen 好干活。

| # | Pass | 作用 |
|---|---|---|
| 30 | `LowerOpaqueBlock` | 移除 opaque `SBlock`，剩下纯 For + BufferStore/Load |
| 31 | `Simplify` | 又一次 |
| 32 | `tirx.transform.NarrowDataType(32)` | 尽量把 index 类型从 int64 缩到 int32（省寄存器）|
| 33 | `FlattenBuffer` | `A[i, j, k]` → `A_flat[i*S1 + j*S2 + k]`，多维索引展平 |
| 34 | `ConfigIndexBitwidth` | 根据配置进一步调 index 位宽（必须在 FlattenBuffer 之后） |
| 35 | `tirx.transform.Simplify` | 用 TVM 官方 Simplify 打扫一遍 |
| 36 | `VectorizeLoop(enable_vectorize=...)` | 把 `T.Vectorized(n)` 循环编成 `A_flat[...] = broadcast(...)` 之类的向量化访存 |
| 37 | `StorageRewrite` | **共享内存 buffer 的复用与折叠**（重叠 lifetime 的多个 buffer 挤到同一段） |
| 38 | `LoopUnswitching` | 循环外提条件（loop invariant if 上升）|
| 39 | `UnrollLoop` | 展开 `T.Unroll(n)` 循环 |
| 40 | `s_tir.transform.RenormalizeSplitPattern` | 拆循环后重整索引 |
| 41 | `tirx.transform.Simplify` | 又一次 |
| 42 | `tirx.transform.RemoveNoOp` | 删死代码（例如 `if False: ...`） |
| 43 | `s_tir.transform.HoistIfThenElse` | 循环内不变的 `if` 提到循环外 |

---

## 5.5 段 D · Host / Device 拆分 + CUDA 收尾

进入这段时，PrimFunc 已经很像"直接可 codegen 的 CUDA IR"了。段 D 的核心是把
"一个大 PrimFunc"切成"host 函数 + device kernel"：

| # | Pass | 作用 |
|---|---|---|
| 44 | `tirx.transform.VerifyMemory` | 校验每次 BufferLoad/Store 的 scope 是否兼容（例如 kernel 内不能直接读 host memory）|
| 45 | `tirx.transform.AnnotateEntryFunc` | 标记 `tir.is_entry_func` |
| 46 | `s_tir.transform.InferFragment` | 给 `wmma.matrix_a/b/accumulator` fragment 属性推断 |
| 47 | `LowerThreadAllreduce` | `T.reduce`（cross-thread reduce）→ 具体的 warp shuffle / shared reduce |
| 48 | `LowerLDGSTG` | `ldg` / `stg`（load-global-with-cache-hint 之类）内建落到 PTX |
| 49 | `LowerHopperIntrin` | Hopper 特有指令（wgmma / async barrier / cluster launch）落到具体形式 |
| 50 | `AnnotateDeviceRegions` | 在 body 上打标记："这一段是 device code" |
| 51 | `SplitHostDevice` | **重量级 pass**：拆出两个 PrimFunc——host 侧留 kernel launch，device 侧是真正的 GPU kernel |
| 52 | `MarkCudaSyncCalls(have_pdl)` | 标记 pdl_sync / pdl_trigger（PDL = Programmatic Dependent Launch，SM90+）|
| 53 | `AnnotateReadOnlyParams` | 给 `__restrict__` / `const` 参数打标记 |
| 54 | `MergeSharedMemoryAllocations` | **多个 shared buffer 合并到一段 dyn shared**，用 `enable_aggressive_merge` / `disable_reuse` 调 |
| 55 | `InjectFenceProxy` | TMA / async proxy 之间插 fence.proxy.async（非 SM90+ target 上是 no-op）|
| 56 | `ThreadSync("shared")` / `ThreadSync("shared.dyn")` | 自动补 `__syncthreads()` |
| 57 | `InjectTcgen05Fence` | Blackwell 专属 fence（SM100+ 才有效）|
| 58 | `MergeIfStmt` | 相邻的 `if (cond)` `if (cond)` 合并 |
| 59 | `AnnotateWarpGroupRegAlloc` | **WS 场景**：给 producer / consumer warp 分别写上 register 数量（`setmaxnreg`）|
| 60 | `MakePackedAPI` | 把 PrimFunc 变成 `PackedFunc` ABI（`TVMValue*, int*, int` 参数列表）|
| 61 | `Simplify` | 又一次 |
| 62 | `LowerDeviceKernelLaunch` | host 侧的"launch device kernel"节点落到 `TVMFuncCall` |
| 63 | `PersistThreadblock` | **只当 kernel 内用了 `T.Persistent` 时才起作用**：把 launch config 从 "N 个 block" 换成 "wave_size 个 block"，见 [第 3 章 3.6](./03_tilelang_dsl.md#36-tpersistent) |

段 D 出口的 IRModule 里有**两个** PrimFunc：一个 `global_symbol="matmul_host"`，一个 `global_symbol="matmul_kernel"`，
后续 `Filter(_is_host_call)` 和 `Filter(_is_device_call)` 就把它俩分道扬镳（回到 [`lower.py`](../../tilelang/engine/lower.py)）。

---

## 5.6 亲手对着 IR 走一遍：**"pass 之前 vs pass 之后"**

单独跑 `CUDAPassPipelineBodyPrologue` 已经够看段 A 的出口了。想看**任何一个 pass 前后**的差异，
用下面这个模板（本仓库内直接可运行）：

```python
import tilelang, tilelang.language as T
from tilelang import tvm as tvm
from tvm import tirx
from tvm.target import Target

# 1) 从你的 kernel 拿 IRModule
pf  = matmul.get_tir(**cfg)                              # 见第 1 章
mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})
tgt = Target("cuda")

# 2) 把段 A 的前 15 个 pass 手动跑一遍，停在 InjectSoftwarePipeline 之前
mod = tirx.transform.BindTarget(tgt)(mod)
mod = tilelang.transform.MaterializeKernelLaunch()(mod)
mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
mod = tilelang.transform.LegalizeNegativeIndex()(mod)
mod = tilelang.transform.InjectAssumes()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.LayoutReducer()(mod)
mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)   # ← 想看这个 pass 的输入？在这行之前 print(mod.script())
mod = tilelang.cuda.transform.LowerBlackwell2SM()(mod)
mod = tilelang.transform.IfStmtBinding()(mod)
mod = tilelang.transform.PipelinePlanning()(mod)
print("===== BEFORE InjectSoftwarePipeline =====")
print(mod.script())

mod = tilelang.transform.InjectSoftwarePipeline()(mod)
print("===== AFTER  InjectSoftwarePipeline =====")
print(mod.script())
```

> 🐛 **调试技巧**：想在**已有** pipeline 内部某一步"叉一个断点"，最偷懒的做法是**改一份
> `CUDAPassPipelineBody` 的副本**，在感兴趣的 pass 前后加 `print(mod.script())`，然后
> 让 `resolve_pipeline` 返回你的副本。不要直接改 `pipeline.py` 里的顺序——这会让所有其他
> kernel 一起翻车。

---

## 5.7 每段的"必看" pass（把 60+ 个 pass 缩到 10 个）

如果只想背 10 个 pass，我推荐这 10 个（覆盖 90% 的 bug）：

1. `MaterializeKernelLaunch`（`T.Kernel` → thread_binding）
2. `ProducerConsumerWarpSpecialized`（WS 分工，第 6 章重点）
3. `PipelinePlanning` + `InjectSoftwarePipeline`（软件流水，第 6 章重点）
4. `LayoutInference`（fragment / swizzle 决定，第 7 章重点）
5. `LowerTileOp`（`T.copy` / `T.gemm` 展开的真正入口）
6. `PlanAndUpdateBufferAllocationLocation`（buffer 摆哪儿）
7. `FlattenBuffer`（多维索引扁平化）
8. `SplitHostDevice`（拆 host / device）
9. `MergeSharedMemoryAllocations`（shared 复用）
10. `ThreadSync`（自动补 `__syncthreads()`）

这 10 个 pass 位于图中不同段，覆盖了整条 pipeline 的关键关节。

---

## 5.8 本章要带走的三件事

1. **CUDA lowering 是 4 段 pipeline**：Prologue → 内存/barrier → 标准化/优化 → Host/Device 拆分。
   每段的目标和"打这一段之后 IR 长啥样"要有画面感。
2. **顺序是有意义的**：`ProducerConsumerWarpSpecialized` 必须在 `LowerTileOp` 之前，
   `MergeSharedMemoryAllocations` 必须在 `SplitHostDevice` 之后。改顺序不是"调优化"，是"埋 bug"。
3. **调试 pipeline 用两招**：
   - 手动串一段 pass 停在感兴趣的位置，`print(mod.script())`
   - 直接调 `CUDAPassPipelineBodyPrologue(mod, target)` 看段 A 的出口

---

下一章 [第 6 章 · 软件流水 + Warp Specialization 深挖](./06_pipeline_and_warp_specialize.md)：
把 5.2.2 那三个 pass 单独拎出来，讲清楚 mbarrier phase counter、
multi-version buffer、K-trip 对齐这些容易踩的坑。
