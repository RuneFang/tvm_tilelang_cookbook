# 附录 A · TVM/TIR 关键类速查表

> 目的：读源码或看正文时突然看到某个类不知道是干嘛的，来这里查。按主题分组，附上定义位置和典型出场章节。
>
> 表中"位置"若以 `3rdparty/tvm/` 开头，说明是 TVM 上游的类；`src/` 或 `tilelang/` 开头则是 TileLang 自己定义或注册的。

## A.1 IR 节点类

| 类名 | 定义位置 | 一句话说明 | 首次出场 |
|---|---|---|---|
| `PrimFunc` / `PrimFuncNode` | `3rdparty/tvm/include/tvm/tir/function.h` | 一个函数，pass 的操作单位 | 第 2 章 |
| `IRModule` | `3rdparty/tvm/include/tvm/ir/module.h` | 多个 PrimFunc 的容器 | 第 2 章 |
| `Buffer` / `BufferNode` | `3rdparty/tvm/include/tvm/tir/buffer.h` | "如何访问一块内存"的描述符（shape/dtype/scope/strides） | 第 2 章 |
| `Var` / `VarNode` | `3rdparty/tvm/include/tvm/tir/var.h` | 抽象变量（不可变），任何 loop_var / buffer var 都是 Var | 第 2 章 |
| `Stmt` 基类 | `3rdparty/tvm/include/tvm/tir/stmt.h` | 无值有副作用的节点 | 第 2 章 |
| `PrimExpr` 基类 | `3rdparty/tvm/include/tvm/tir/expr.h` | 有值的节点 | 第 2 章 |
| `GlobalVar` | `3rdparty/tvm/include/tvm/ir/expr.h` | IRModule 里 PrimFunc 的名字 | 第 5 章 |
| `Target` | `3rdparty/tvm/include/tvm/target/target.h` | 编译目标描述（`"cuda -arch=sm_90a"`） | 第 5 章 |

## A.2 常见 Stmt 节点

| 类名 | 语义 | 典型出场 |
|---|---|---|
| `SeqStmt` | 语句序列 | 到处都是 |
| `For` | 循环，有 kind 字段（Serial / Parallel / Vectorized / Unrolled / ThreadBinding） | 第 4/6 章 |
| `IfThenElse` | 条件 | 第 4 章 |
| `Block` / `BlockRealize` | 结构化块（第 2 章讲过）；BlockRealize 是"块的一次具体调用" | 第 2 章 |
| `BufferStore` | `buffer[indices] = value` | 第 2 章 |
| `Allocate` | 分配存储（scope 决定 shared / local / global） | 第 4 章 |
| `AttrStmt` | 挂元数据到某段代码上（thread binding / pragma / storage_scope 等） | 第 5 章 |
| `Evaluate` | 只求值一个 Expr，用于纯 side-effect intrinsic call | 第 6 章 |
| `LetStmt` | 命名一个中间值 | 第 4 章 |
| `DeclBuffer` | 声明一个 buffer 变量（不是分配，只是"告诉 IR 这里有个 buffer"） | 第 5 章 |
| `AssertStmt` | 运行时断言 | 第 5 章 |
| `While` | while 循环 | 少见 |

## A.3 常见 Expr 节点

| 类名 | 语义 | 备注 |
|---|---|---|
| `IntImm` / `FloatImm` / `StringImm` | 立即数 | 常量 |
| `Var` | 变量 | |
| `BufferLoad` | `buffer[indices]` | 有值版的读 |
| `Add` / `Sub` / `Mul` / `Div` / `Mod` | 常规算术 | 有符号版本 |
| `FloorDiv` / `FloorMod` | 向下取整算术 | pass 里几乎都用这两个，避免负数歧义 |
| `EQ` / `NE` / `LT` / `LE` / `GT` / `GE` | 比较 | |
| `And` / `Or` / `Not` | 逻辑 | |
| `Cast` | 类型转换 | |
| `Select` | 三元 `cond ? a : b` | |
| `Call` | 调用（函数 / intrinsic）；`op` 字段是 `Op`，如 `builtin::if_then_else` | 第 6 章 |
| `Ramp` / `Broadcast` | 向量类节点 | 向量化后大量出现 |
| `Shuffle` | 向量重排 | 向量化 |
| `Let` | 有值版本的 let | |

## A.4 Pass 相关（TVM 上游）

| 类名 | 位置 | 用途 | 出场 |
|---|---|---|---|
| `StmtVisitor` | `3rdparty/tvm/include/tvm/tir/stmt_functor.h` | 只读遍历 Stmt | 第 4 章 |
| `ExprVisitor` | `3rdparty/tvm/include/tvm/tir/expr_functor.h` | 只读遍历 Expr | 第 4 章 |
| `StmtExprVisitor` | `stmt_functor.h` | Stmt+Expr 一体只读 | 第 4 章 |
| `StmtMutator` / `ExprMutator` / `StmtExprMutator` | 同上 | 可写版本，`Stmt/Expr → Stmt/Expr` | 第 4 章 |
| `IRMutatorWithAnalyzer` | `3rdparty/tvm/src/arith/ir_mutator_with_analyzer.h` | Mutator + `arith::Analyzer` 常量传播 | 第 6/10 章 |
| `PassContext` | `3rdparty/tvm/include/tvm/ir/transform.h` | pass 运行时上下文（config / trace hook） | 第 4 章 |
| `Pass` | 同上 | pass 的顶层类 | 第 4 章 |
| `PrimFuncPass` | `3rdparty/tvm/include/tvm/tir/transform.h` | 作用于单个 PrimFunc 的 Pass 子类 | 第 4 章 |
| `Sequential` | `3rdparty/tvm/include/tvm/ir/transform.h` | 把一串 Pass 串联成一个大 Pass | 第 5 章 |
| `arith::Analyzer` | `3rdparty/tvm/include/tvm/arith/analyzer.h` | 常量传播 / 范围推断 / 表达式化简三合一 | 第 4/6 章 |

## A.5 Pass 相关（TileLang 特有）

| 类名 / 函数 | 位置 | 用途 | 出场 |
|---|---|---|---|
| `tvm.tirx.transform.prim_func_pass` | `3rdparty/tvm/python/tvm/tirx/transform/function_pass.py` | TileLang 用的 pass 装饰器（区别于上游 `tvm.tir.transform`） | 第 4 章 |
| `PassPipeline` | `tilelang/backend/pass_pipeline/pipeline.py` | 组织多个 pass 的执行顺序（按 target 分派） | 第 4/5 章 |
| `PassConfigKey` | `tilelang/transform/pass_config.py` | 全书那些 `tl.*` 配置项的键名枚举 | 第 4 章 |
| `resolve_pipeline(target)` | `tilelang/backend/pass_pipeline/pipeline.py` | 拿到某后端的 pipeline 实例（按 `target.kind.name` 分派） | 第 5 章 |

## A.6 Codegen 相关

| 类名 | 位置 | 用途 | 出场 |
|---|---|---|---|
| `CodeGenC` | `3rdparty/tvm/src/target/source/codegen_c.h` | 上游"TIR → 类 C 源码"的访问器基类 | 第 8 章 |
| `CodeGenTileLangCUDA` | `src/cuda/codegen/codegen_cuda.h` | TileLang 的 CUDA codegen，继承 `CodeGenC` | 第 8 章 |
| `runtime::Module` | `3rdparty/tvm/include/tvm/runtime/module.h` | 编译产物容器（一个 cubin + 一堆 PackedFunc） | 第 8/9 章 |
| `PackedFunc` / `ffi::Function` | `3rdparty/tvm/include/tvm/ffi/function.h` | 跨语言可调用对象，`kernel(A, B)` 的最终形态 | 第 8/9 章 |
| `DLTensor` | `3rdparty/tvm/3rdparty/dlpack/include/dlpack/dlpack.h` | 跨框架 tensor 描述符（torch/numpy/tvm 共用） | 第 9 章 |

## A.7 TileLang 特有 IR / Op

| 类名 / 常量 | 位置 | 用途 | 出场 |
|---|---|---|---|
| `tl::builtin::mvb_stage_index` | `src/op/builtin.h` | 多版本 buffer 的 stage 索引 marker（provenance tag，见第 6 章 6.7） | 第 6 章 |
| `tl::builtin::mbarrier_wait_parity` | `src/op/builtin.h` | mbarrier 等待某相位 | 第 6 章 |
| `tl::builtin::mbarrier_arrive` / `mbarrier_arrive_expect_tx` | `src/op/builtin.h` | mbarrier 到达 / 到达+expect_tx | 第 6 章 |
| `tl::builtin::create_barriers` | `src/op/builtin.h` | 分配 mbarrier 数组 | 第 6 章 |
| `tl::Gemm` / `tl::GemmSP` / `tl::GemmSR` 等 | `src/op/gemm.cc` 等 | tile-level gemm intrinsic 的 C++ 内部 op | 第 3 章 |
| `tl::Copy` | `src/op/copy.cc` | tile-level copy intrinsic | 第 3 章 |
| `tl::Reduce` / `tl::Scan` / `tl::Fill` | `src/op/*.cc` | 对应 tile-level 计算 op | 第 3 章 |
| `tl::attr::kHasGridSync` | `src/op/builtin.h` | PrimFunc 属性：需要 grid 级同步 | 第 8 章 |

## A.8 Layout / Fragment 相关

| 类名 | 位置 | 用途 | 出场 |
|---|---|---|---|
| `tl::Layout` | `src/layout/layout.h` | 逻辑坐标 → 物理索引的映射 | 第 7 章 |
| `tl::Fragment` | `src/layout/layout.h` | Layout + "该由哪个线程处理"的合体 | 第 7 章 |
| `tl::Swizzle` | `src/layout/swizzle.h` | Bank conflict 规避的位交换 | 第 7 章 |

## A.9 读 C++ pass 时的常见惯用法（不是类，是"套路"）

看 `src/**/*.cc` 时会反复撞见下面这些名字，它们大多**自解释**，先混个眼熟：

| 写法 | 含义 | 备注 |
|---|---|---|
| `GetRef<For>(op)` | 从裸指针 `const ForNode* op` 拿回带引用计数的 wrapper `For` | 和 `.get()` 相反（第 2 章） |
| `Downcast<PrimFunc>(x)` | 把一个基类引用**向下转型**成具体类型，转不成会报错 | 类似 C++ 的 `static_cast` + 检查 |
| `x->IsInstance<PrimFuncNode>()` | 运行时**判断节点类型**，返回 bool | 配 `Downcast` 用："先判断再转" |
| `f->GetAttr<Integer>("key")` | 从 PrimFunc / IRModule **取一个属性**（可能不存在，返回 `Optional`） | 属性即第 2 章的 `attrs` |
| `node.same_as(other)` | 判断两个引用**是不是同一个对象**（指针相等） | Mutator 里"没改就返回原节点"的短路依据（第 2 章） |
| `ICHECK(cond) << "msg"` | 断言，不满足直接抛错并打印 msg | TVM 版的 `assert`（第 2 章） |
| `CopyOnWrite()` | 拿到一个"可安全就地改"的可变副本指针 | 因为节点默认不可变（第 2/10 章） |

（若你在源码里遇到本表未列出的核心类或惯用法，欢迎在此追加。）
