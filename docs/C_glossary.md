# 附录 C · 术语表

> 目的：给你一个"随时可回查"的术语字典。全书所有专有名词按字母排序（中文以拼音、缩写按首字母），每条给出：**中文** · **英文/缩写** · **一句话解释** · **首次出现章节**。
>
> 使用建议：读到不认识的词，Ctrl+F 搜一下；写 PR 描述时，用这里定义好的词更容易被 reviewer 一次读懂。

---

## A

**AST（抽象语法树）** · Abstract Syntax Tree · 编译器把源码解析后得到的树形数据结构，TVM 里的 TIR 本质就是一棵 AST。 · 第 2 章

**Analyzer（分析器）** · `arith::Analyzer` · TVM 的常量传播 / 范围推断 / 表达式化简三合一工具，pass 里手上常年拿着一个。 · 第 4 章

**Adapter（适配器）** · JIT Adapter · TileLang 中"把 kernel 参数从 torch.Tensor 转成 DLTensor 再喂给 PackedFunc"的桥接类。 · 第 9 章

**Attr（属性）/ AttrStmt** · Attribute / `AttrStmtNode` · 挂在某段代码上的元数据（比如 `thread_binding`、`pragma_unroll`、`storage_scope`），本身没运行时开销。 · 第 5 章

---

## B

**Bank Conflict（存储体冲突）** · Bank Conflict · 多个线程同时读写 shared memory 的同一个 bank 时被硬件串行化，性能骤降。Swizzle 就是为了绕开它。 · 第 7 章

**Barrier（屏障）** · Barrier · 让一组线程/warp 停下来等对方到齐才继续。TileLang 里最常见的是 `mbarrier`（Hopper 起硬件支持的 memory barrier）。 · 第 6 章

**BlockRealize** · `BlockRealizeNode` · "一次具体的块调用"——TIR 里 Block 是模板，BlockRealize 是把它放进循环某处的实例化。 · 第 2 章

**Buffer** · `BufferNode` · "如何访问一块内存"的描述符：shape / dtype / strides / scope。**不等于内存本身**，只是"访问方式"。 · 第 2 章

**Builtin（内建函数）** · Builtin · TVM/TileLang 定义的一批 `CallNode.op`，比如 `tl::builtin::mvb_stage_index`。区别于用户函数：builtin 是编译器内部认识的。 · 第 4/6 章

---

## C

**CallNode** · `CallNode` · 表达式里"调用某个 op/函数"的节点，`op` 字段是 `Op` 对象。 · 第 4 章

**Cluster（集群）** · Thread Block Cluster · sm_90 起引入的一层，介于 grid 和 block 之间，允许多个 block 共享 shared memory / 同步。 · 第 13 章

**CTA** · Cooperative Thread Array · **就是 CUDA 里的一个 thread block**（一个 `T.Kernel` 的 block），源码里 CTA 与 thread block 常混用。 · 第 3 章

**Codegen（代码生成）** · Codegen · 把 IR 打印成目标语言源码字符串的过程；TileLang 打印的是 CUDA C++。 · 第 8 章

**Cooperative Launch** · Cooperative Launch · CUDA 的一种 launch 方式，允许 grid 内所有 block 全局同步（`cooperative_groups::grid_group::sync()`）。 · 第 8 章

**CUDA Driver API vs Runtime API** · — · 前者更底层（`cuLaunchKernel`），TVM/TileLang 用的是 Driver API；后者是 `cudaLaunchKernel` 的高层封装。 · 第 9 章

**cubin** · CUDA Binary · NVCC 编译出的针对某一具体 SM 版本的二进制。`.fatbin` 是多 sm 版本打包。 · 第 8 章

**cp.async** · — · Ampere(SM80) 起的异步拷贝指令，让 global→shared 的搬运与计算重叠。没有 TMA（Hopper）时的主要手段。 · 第 3 章

**CuTe** · CuTe / CUTLASS · NVIDIA 的 tensor-core layout 抽象库。TileLang 的 layout 系统与它相容。 · 第 7 章

---

## D

**DLPack / DLTensor** · DLPack · 跨深度学习框架的 tensor 交换协议，`torch.Tensor` → `DLTensor` → TVM PackedFunc 就是通过它。 · 第 9 章

**DMA** · Direct Memory Access · 直接内存访问——不占用计算单元、由专门硬件完成的数据搬运。TMA 就是 Hopper 的一种硬件 DMA 引擎。 · 第 6 章

**DSL（领域特定语言）** · Domain Specific Language · TileLang 提供的 `T.Kernel`/`T.copy`/`T.gemm` 那套就是 DSL——建立在 Python 语法之上、专门写 GPU tile 计算。 · 第 3 章

---

## E

**Evaluate** · `EvaluateNode` · 一个只求值不保留结果的语句，一般包裹 side-effect intrinsic（如 `mbarrier_arrive`）。 · 第 6 章

**Expr（表达式）** · `PrimExpr` · 有值的 IR 节点，如 `IntImm`、`Var`、`Add`、`BufferLoad`、`Call`。 · 第 2 章

---

## F

**Fatbin** · Fat Binary · 多个 sm 版本的 cubin 打包成一个文件，运行时按 GPU 选。 · 第 8 章

**fp16 / bf16** · half / bfloat16 · 都是 16 位浮点。fp16=1 符号+5 指数+10 尾数；bf16=1+**8**+**7**，指数位更多所以范围大（≈fp32）、精度略低。 · 第 14 章

**fp8（e4m3 / e5m2）** · 8-bit float · 8 位浮点，名字即位数分配：**e4m3**=4 指数+3 尾数（精度略高），**e5m2**=5 指数+2 尾数（范围更大）。 · 第 14 章

**FFI（外部函数接口）** · Foreign Function Interface · TVM 里 `.def("xxx", CppFunc)` + `tvm.ffi.get_global_func("xxx")` 那套跨语言调用机制。 · 第 4/8 章

**Fragment（片段）** · `tl::Fragment` · Layout + "该由哪个线程处理"的合体。表达"一个 warp 上寄存器里如何摊平 tensor 值"。 · 第 7 章

**FloorDiv / FloorMod** · — · TVM/TileLang pass 里"整除、取模"的标准形式，行为定义得对负数也一致。 · 第 2/6 章

---

## G

**Gemm（矩阵乘）** · General Matrix Multiply · `T.gemm(A, B, C)`，TileLang 会根据 GPU 架构展成 mma / WGMMA / UMMA。 · 第 3 章

**GlobalVar** · `GlobalVarNode` · IRModule 里 PrimFunc 的名字对象。 · 第 5 章

**Grid / Block / Thread** · — · CUDA 的三级线程层次。TileLang 的 `T.Kernel(bx, by, ..., threads=N)` 决定 grid 和 block 尺寸。 · 第 3 章

---

## H

**Hopper** · sm_90(a) · NVIDIA H100/H200 的 GPU 架构代号，引入了 WGMMA / TMA / mbarrier / Thread Block Cluster。 · 第 6 章

**Host stub** · Host Stub · Device kernel 之外那份 Python/C++ 可调用的 host 函数，负责参数打包和 launch。 · 第 8/9 章

**Hard signature（硬签名）** · —（本书说法） · 对生成的 CUDA 源码 grep 特征字符串来做断言的测试手法（如 `assert "bug marker" not in src`）。业界更常见叫法是 snapshot / golden-file test。 · 第 6/8/10/11 章

---

## I

**IR（中间表示）** · Intermediate Representation · 编译器的中间数据结构。TIR 是 TVM 的 IR，专注 tensor / loop。 · 第 2 章

**IRModule** · `IRModuleNode` · 多个 PrimFunc 的容器，pass pipeline 的输入/输出单元。 · 第 2 章

**Intrinsic（内建指令）** · Intrinsic · 用户不能直接写、编译器内部生成的 op，如 `mvb_stage_index`、`mbarrier_arrive`。 · 第 6 章

---

## J

**JIT（即时编译）** · Just-In-Time compilation · `@tilelang.jit` 装饰后，第一次调用时才编译，编译结果缓存到磁盘。 · 第 9 章

---

## K

**Kernel Cache** · — · TileLang 缓存"参数 shape + pass config + IR hash → cubin"的映射，避免每次重编。 · 第 9 章

**K-trip** · —（本书说法） · K 维外层循环的迭代次数，即 `ceildiv(K, block_K)`（trip count = 循环圈数）。当它不是 `num_stages` 的倍数时，mbarrier phase 容易漂移。 · 第 6 章

---

## L

**Launch Params** · — · Kernel 启动时要传的额外参数：`grid_dim` / `block_dim` / `shared_bytes` / `cluster_dims` / cooperative flag 等。 · 第 8 章

**Layout（布局）** · `tl::Layout` · 逻辑坐标（例如 `(i, j)`）到物理索引（例如 shared memory offset）的映射函数。 · 第 7 章

**LetStmt / Let** · — · 命名一个中间值：`let x = a + b in <body>`。 · 第 4 章

**Legalize（合法化）** · Legalize · 把用户表达但不能直接 lower 的形式改写成 pass 认识的形式，比如 `frontend_legalize.cc`。 · 第 5 章

**Lowering（下降）** · Lowering · 把高层 IR 一层层降到硬件能执行的形式的过程。TileLang 的整个 pass pipeline 就是 lowering。 · 第 5 章

---

## M

**mbarrier** · Memory Barrier · Hopper 起硬件级的 shared memory barrier，配 phase counter 使用。（前言/第 1 章即出现，第 6 章详解。） · 第 6 章

**Multi-version Buffer（多版本 buffer）** · Multi-version Buffer · 软件流水下，同一逻辑 buffer 在物理上开成 `num_stages` 份（`A_shared[stage][...]`）。 · 第 6 章

**Mutator** · `StmtExprMutator` 等 · 只读遍历叫 Visitor；可写、返回新 IR 的叫 Mutator。 · 第 4 章

---

## N

**NVCC** · — · NVIDIA 官方 CUDA 编译器可执行文件，编译走独立进程。 · 第 8 章

**NVRTC** · NVIDIA Runtime Compilation · 在进程内把 CUDA 源码字符串编成 PTX/cubin 的库。 · 第 8 章

**nibble** · Nibble · 半个字节 = 4 位。"一个 uint8 装 2 个 int4"就是"一个 byte 装 2 个 nibble"。 · 第 14 章

**num_stages** · — · 软件流水的深度参数，决定多版本 buffer 开几份。 · 第 6 章

---

## O

**Op** · `tvm::Op` · TVM 里"具名 op"的对象。`CallNode.op` 是它。 · 第 4 章

---

## P

**PackedFunc / ffi::Function** · — · TVM 跨语言可调用对象，`kernel(A, B)` 最终会走到一个 PackedFunc。 · 第 8/9 章

**Pass** · — · IR → IR 的一次变换。TileLang 有 ~50 个 pass。 · 第 4 章

**Prologue / Steady-state / Epilogue** · — · 软件流水把一个循环改写成的三段：prologue（开头只预取数据）、steady-state（稳定期边取边算）、epilogue（末尾只算不取）。 · 第 5/6 章

**PassContext** · — · pass 运行时上下文，携带 config、trace hook 等。 · 第 4 章

**PassPipeline** · — · 组织多个 pass 的执行顺序的容器（TileLang 自定义的，非上游 TVM 类）。 · 第 4/5 章

**PDL（编程式依赖启动）** · Programmatic Dependent Launch · Hopper 起支持的机制，让上一个 kernel 未结束时下一个 kernel 就可以 launch。 · 第 8 章

**Persistent Kernel（持久化 kernel）** · — · 一个 kernel launch 处理多个 tile，避免反复 launch 开销。 · 第 6 章

**Phase Counter / Phase Bit** · — · mbarrier 的"当前是第几轮"计数器，每次 flip 一次 bit。跨 tile 不对齐时最容易出错（见第 6 章 6.9）。 · 第 6 章

**PrimExpr / PrimFunc / PrimFuncPass** · — · "Prim-"前缀表示 TIR 侧（区别于 Relay/Relax 的高层 IR）。 · 第 2 章

**Producer / Consumer Warp** · — · Warp Specialization 下，一部分 warp 只做 load（producer），另一部分只做 compute（consumer）。 · 第 6 章

**Provenance vs Syntax** · — · 判断"这个表达式是不是编译器生成的"要靠 provenance（谁生成的、打了什么 tag），不能靠 syntax（长成什么样）。是 pass 组合的一条根本规则（见第 6 章 6.7）。 · 第 6/10 章

**PTX** · Parallel Thread Execution · NVIDIA 的伪汇编中间形式，位于 CUDA 源码和 cubin 之间。 · 第 8 章

---

## R

**Reduction（归约）** · Reduction · `T.reduce_sum` 那一类"多输入变一个输出"的运算。 · 第 3 章

**Region** · `tl::Region` · tile 上的"区域"抽象，用来描述"从哪个位置开始、多大范围"的读写。 · 第 7 章

**Rewrite（重写）** · Rewrite · Mutator 的操作，把一段 IR 换成另一段 IR。 · 第 4 章

**runtime::Module** · — · TVM 编译产物容器，一个 module 里可以有多个 PackedFunc。 · 第 8/9 章

---

## S

**Scope（存储域）** · Storage Scope · Buffer 存在哪里：`"global"` / `"shared"` / `"local"` / `"wmma.matrix_a"` 等。 · 第 5 章

**SeqStmt** · `SeqStmtNode` · 语句序列容器。 · 第 4 章

**SM（流多处理器）** · Streaming Multiprocessor · GPU 上的一个物理核心，一个 kernel launch 会把 block 分派到多个 SM 上。**SM 号也是架构版本号**：SM70=Volta、SM80=Ampere(A100)、SM90=Hopper(H100)、SM100=Blackwell。 · 第 1 章

**SSA（静态单赋值）** · Static Single Assignment · 每个变量只赋值一次的 IR 形式，TVM 的很多 pass 假设或维护此性质。 · 第 8 章

**StmtVisitor / StmtMutator** · — · TVM 遍历/改写 Stmt 的基类。 · 第 4 章

**Swizzle** · Swizzle · 通过位交换重排 shared memory 索引，把不同 bank 错开避免冲突。 · 第 7 章

---

## T

**Target** · `Target` · 编译目标描述（例如 `cuda -arch=sm_90a`），决定用哪套 pass pipeline 和 codegen。 · 第 5 章

**Tensor Core** · Tensor Core · GPU 上专门做矩阵乘加（MMA）的硬件单元，比普通 CUDA core 快一个数量级。`T.gemm` 最终就是让它干活；调用它的指令有 mma / WGMMA / UMMA 等。 · 第 1 章

**TIR** · Tensor IR · TVM 的低层 IR，本书讨论的 IR 都是它。 · 第 2 章

**tirx** · Next-gen TIR · TileLang 用的是 TVM 下一代 TIR 分支（`tvm.tirx.transform.*`），API 与上游 `tvm.tir.transform` 有细微差异。 · 第 4 章

**TL_DISABLE_WARP_SPECIALIZED** · — · 环境变量/pass config，关闭 WS，用来生成"wsoff reference"和 WS 输出做数值 diff。 · 第 6 章

**TMA（Tensor Memory Accelerator）** · TMA · Hopper 引入的异步大 tile 传输硬件单元，用于 global ↔ shared。 · 第 6 章

---

## U

**UMMA** · — · Blackwell (sm_100) 的下一代矩阵乘硬件指令。 · 第 6 章

**Unroll（展开）** · Loop Unroll · 把循环体重复几遍以减少循环开销，`#pragma unroll` 就是这个。 · 第 4 章

---

## V

**Var / VarNode** · — · TIR 里的抽象变量，一切 loop_var / buffer var 都是它。 · 第 2 章

**Vectorize（向量化）** · Vectorize · 把标量运算合并成向量指令（如 `float4 = ld.v4`）。 · 第 5 章

**Visitor Pattern（访问者模式）** · — · "数据结构定义" vs "操作数据结构的算法"解耦的设计模式，codegen 和 pass 都用它。 · 第 4/8 章

---

## W

**Warp** · — · CUDA 32 线程为一组的调度单元。 · 第 6 章

**Warp Group（WG）** · — · Hopper 起的 4-warp（128 线程）分组单位，是 WGMMA 的执行粒度。 · 第 6 章

**Warp Specialization（WS，线程束特化）** · Warp Specialization · 让不同 warp 各司其职（有的搬数据、有的算），并通过 mbarrier 同步。 · 第 6 章

**WGMMA** · Warp-Group Matrix-Multiply-Accumulate · Hopper 的异步矩阵乘指令。 · 第 6 章

**wsoff reference** · — · 关闭 WS 编出来的 kernel，作为"正确性金标准"和 WS 输出做数值 diff。是验证 WS pass 正确性的关键测试范式（见第 6 章 6.9.5）。 · 第 6/10 章

---

## 缩写快速对照表

| 缩写 | 全称 | 领域 |
|---|---|---|
| AST | Abstract Syntax Tree | 编译器 |
| DSL | Domain Specific Language | 编译器 |
| FFI | Foreign Function Interface | 系统 |
| IR | Intermediate Representation | 编译器 |
| JIT | Just-In-Time compilation | 编译器 |
| MMA | Matrix-Multiply-Accumulate | GPU |
| PDL | Programmatic Dependent Launch | CUDA |
| PTX | Parallel Thread Execution | CUDA |
| SM | Streaming Multiprocessor | GPU |
| SSA | Static Single Assignment | 编译器 |
| TIR | Tensor IR | TVM |
| TMA | Tensor Memory Accelerator | Hopper |
| UMMA | (Next-gen) MMA | Blackwell |
| WGMMA | Warp-Group MMA | Hopper |
| WS | Warp Specialization | CUDA |

（如果读全书遇到本表未收录的术语，欢迎在此追加。）
