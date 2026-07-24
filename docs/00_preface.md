# 第 0 章 · 前言与阅读指南

## 为什么写这本书

你正拿着一个叫 TileLang 的东西，它对外像 Python，对内却包含了：

- 一个自己的 **DSL 前端**（`@T.prim_func` + `T.Kernel` + `T.gemm` …）
- 一整套 **中间表示（IR）**，也就是 TVM 家的 **TIR**（Tensor IR）
- **几十个 Pass**，每个 Pass 都是一个"把 IR 树改一改"的小函数
- 一个 **Codegen 后端**，把改到最后的 IR 打印成 **CUDA 源码字符串**
- 一个 **Runtime**，把 CUDA 源码丢给 **NVRTC** 编成 `cubin`，再从 Python 侧调进 GPU

任何一层单独拎出来都够写一本教材。已有的 TVM 官方文档假定你已经理解了"张量编译器"这个概念、
"IR"这个词、"Pass"这套架构；而市面上大部分讲 CUDA 的教程又完全不谈编译器。
本书填的就是这条缝。

## 读者画像

我假设你：

- 会写基本的 Python（会用装饰器、会读类）
- 听过 CUDA、知道"线程 / block / warp / shared memory"这些词，但不一定手写过 kernel
- 完全没读过编译器代码，或者只知道"parser → AST → codegen"三个词

我**不假设**你：

- 学过编译原理课程
- 会 C++ 模板元编程
- 用过 TVM / Halide / MLIR / Triton

## 本书的三条主线

读的时候脑子里请一直挂着这三张图：

### 主线 1：数据形态的变化

```
Python 源码
   │  (Python AST + tilelang.language.parser 解析)
   ▼
TIR: PrimFunc（一棵 Stmt/Expr 树）
   │  (几十个 Pass 依次改写这棵树)
   ▼
TIR: 已经 lower 好、贴近硬件的 PrimFunc
   │  (codegen_cuda 把它打印成字符串)
   ▼
CUDA 源码字符串
   │  (NVRTC 编译)
   ▼
cubin / PTX
   │  (CUDA Driver 加载)
   ▼
GPU 上真正运行的 kernel
```

**记住**：从第二步到第三步中间是 TileLang 最核心、也是最值得学的部分。

### 主线 2：目录 ↔ 阶段 的对应

```
tilelang/language/    ← 前端 DSL（把 Python 变成 TIR）
tilelang/engine/      ← lowering 编排（决定 pass 顺序）
src/transform/        ← 通用 pass（跨后端）
src/cuda/transform/   ← CUDA-only 的 pass
src/op/               ← Tile-level op 的定义（copy / gemm / reduce…）
src/cuda/codegen/     ← 把最终 TIR 打印成 CUDA 源码
src/cuda/runtime.cc   ← 运行时（NVRTC、cubin 加载、参数打包）
tilelang/jit/         ← JIT 缓存、Python 层入口
```

书里每一章基本对应上面的一个目录，你可以先记住这张表，后面看到 "咦，这个 pass 应该在哪" 就来这里对号入座。

### 主线 3：一个 Pass 的通用套路

不管 pass 长什么样，它一定回答三个问题：

1. **输入是什么形状的 IR？**（例如："输入的 PrimFunc 里所有 `T.Pipelined` 都还未展开"）
2. **输出是什么形状的 IR？**（例如："展开后的 PrimFunc，`T.Pipelined` 被换成了带 mbarrier（一种硬件同步器，第 6 章详解）的双缓冲循环"）
3. **中间它扫描 / 改写了哪些节点？**（这决定了 pass 该用 `StmtExprVisitor` 只看，还是 `StmtExprMutator` 会改）

第 4 章会把这套套路讲透，之后每一章都在拿具体 pass 往这套模板里套。

## 阅读姿势建议

1. **一定要打开 IDE**（VSCode / Cursor）在旁边，本书的每一个源码引用都可以 `Ctrl+P` 跳过去
2. **每章开头有 `TL;DR` 一句话概括，末尾有"本章要带走的"/"小结"**——先读开头建立预期，读完对着末尾自查有没有漏
3. 正文里带 `⚠️ 常见误解` 框的地方**重点看**——那都是初学者最容易想错、且一旦想错就会写出 bug 的地方
4. 大部分章节配有"亲手做一遍 / 亲手看一眼"小节和 `examples/` 里的可运行示例，**读完概念一定要挑一个跑/dump 一次**——不跑一次你会以为自己懂了但其实没懂
5. 遇到不认识的类，先翻 [附录 A：TVM/TIR 关键类速查表](./A_tvm_tir_cheatsheet.md)
6. 术语不熟去 [附录 C：术语表](./C_glossary.md)
7. 读完一章问自己："如果我要给这一层加个 feature，我会改哪个文件？" 回答不出来说明没读透，回去再看

## 用什么环境

写这本书时用的是：

- tilelang 版本：`0.1.12`（仓库根目录 `VERSION` 文件里可查；源码演进后类名 / 函数名一般稳定，行号可能有偏移）
- Python 3.10+
- CUDA 12.x（书里的 CUDA-specific 讨论以 Hopper / SM90 为主，SM100 / Blackwell 会额外标注）
- 一张 GPU（不然第 1 章跑不起来）

如果你没有 GPU，也可以把 Kernel 编译到 CUDA 源码字符串停下——**大部分学习价值在编译阶段，
不在运行阶段**。第 10 章会讲怎么只做 lowering 不做 launch。

---

好了，翻页去 [第 1 章 · 一个最小的 TileLang 例子跑起来了发生什么](./01_hello_tilelang.md)。
