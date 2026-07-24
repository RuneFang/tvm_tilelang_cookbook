# 第 1 章 · 一个最小的 TileLang 例子跑起来了发生什么

> **TL;DR**：本章把 [`examples/quickstart.py`](../../examples/quickstart.py) 这 88 行代码从头到尾拆开，
> 让你看到"用户 Python 代码"→"GPU 上真正在跑的 kernel"这条路径上一共有几个阶段，每个阶段的名字是什么、
> 由哪个文件负责。后面所有章节都是在放大这张地图上的某个阶段。
>
> **你会读到的真实源码**：
> - `examples/quickstart.py`
> - `tilelang/jit/__init__.py`
> - `tilelang/jit/kernel.py`
> - `tilelang/engine/lower.py`
> - `src/cuda/codegen/codegen_cuda.cc`
> - `src/cuda/runtime.cc`

---

## 1.1 我们要跑的代码

打开 [`examples/quickstart.py`](../../examples/quickstart.py)，核心只有 3 段：

```python
@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
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
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by*block_M, ko*block_K], A_shared)
            T.copy(B[ko*block_K, bx*block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)  # relu
        T.copy(C_local, C[by*block_M, bx*block_N])
    return C

kernel = matmul.compile(M=1024, N=1024, K=1024,
                        block_M=128, block_N=128, block_K=32)

a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = kernel(a, b)
```

粗看：这就是个 tile-based 的 matmul + relu，用 `T.Pipelined` 做 3-stage 软件流水，用 `T.gemm` 调用 Tensor Core。

> **先看懂开头这几行"声明"**（它们的写法和普通 Python 不一样，第一眼容易懵）：
>
> - **`M, N, K = T.const("M, N, K")`** —— 声明三个**编译期符号常量**（表示矩阵维度）。`T.const` 接收一个字符串，**按逗号/空格拆成多个名字**，一次返回一个元组，所以能直接解包给 `M, N, K`。这些维度的**具体数值不写死在函数里**，而是等你 `matmul.compile(M=1024, ...)` 或用真实 tensor 调用时才被填入（推断）。这样同一个 `matmul` 定义能服务任意 shape。（这是 `@tilelang.jit` 的 eager 模式专用写法。）
> - **`A: T.Tensor((M, K), dtype)`** —— 用 Python 的**类型注解语法（冒号 `:`）**声明一个**输入** tensor 的形状和 dtype。它不产生新数据，只是告诉编译器"参数 `A` 长这样"。
> - **`C = T.empty((M, N), dtype)`** —— 用**赋值语法（等号 `=`）**声明一个**输出** tensor。区别在于：输入 `A/B` 是外面传进来的（用 `:` 注解），输出 `C` 是这个 kernel **要新建并返回**的（用 `=` 创建），最后 `return C`。
>
> 记住这个对应关系：**`:` 注解 = 传进来的输入**，**`= T.empty(...)` = 这里新建的输出**。剩下的 `T.Kernel` / `T.copy` / `T.gemm` 等是"计算体"，1.3 起逐一展开。

现在的问题是：**从 `@tilelang.jit` 到 `c = kernel(a, b)` 之间发生了什么？**

## 1.2 全景地图：一次 `.compile()` 会经历的 6 个阶段

> **说明**：下面把整条编译链路划成"6 个阶段"（阶段一…阶段六）只是**本书为了讲解方便自造的分段**，不是 TileLang / 编译器领域的官方术语。你和别人交流时用官方说法（前端解析 / lowering / codegen / NVRTC 编译 / host stub / launch）即可，"阶段二"这类编号只在本书内部生效。

先给你张全景图，接下来的小节按阶段顺序讲：

```
┌──────────────────────────────────────────────────────────────────────┐
│  用户 Python 源码                                                       │
│      def matmul(...):                                                  │
│          with T.Kernel(...) as (bx, by):                               │
│              for ko in T.Pipelined(..., num_stages=3):                 │
│                  T.copy(...); T.gemm(...)                              │
└──────────────────────────────────────────────────────────────────────┘
                     │  ①  DSL 解析：Python AST → TIR
                     │       负责人：tilelang/language/parser/
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  TIR 初始形态 (PrimFunc + IRModule)                                     │
│      T.Kernel、T.Pipelined、T.copy、T.gemm 都还在                        │
│      以"高层 tile-level intrinsic"的样子存在                              │
└──────────────────────────────────────────────────────────────────────┘
                     │  ②  Lowering Pipeline（几十个 Pass 依次运行）
                     │       负责人：tilelang/engine/lower.py
                     │              +  src/transform/*.cc
                     │              +  src/cuda/transform/*.cc
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  TIR 低阶形态                                                            │
│      T.copy 变成 async_copy + mbarrier                                 │
│      T.gemm  变成 wgmma 指令                                            │
│      T.Pipelined 变成显式双缓冲 + arrive/wait                            │
│      warp specialization 已注入（如果开）                                 │
└──────────────────────────────────────────────────────────────────────┘
                     │  ③  Codegen：TIR → CUDA 源码字符串
                     │       负责人：src/cuda/codegen/codegen_cuda.cc
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  一段真正合法的 CUDA C++ 源码字符串                                        │
│      __global__ void matmul_kernel_0(...) { ... }                       │
└──────────────────────────────────────────────────────────────────────┘
                     │  ④  NVRTC 编译：CUDA 源码 → cubin
                     │       负责人：tilelang/contrib/nvcc.py + nvrtc.py
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cubin (二进制)                                                          │
└──────────────────────────────────────────────────────────────────────┘
                     │  ⑤  Host stub 生成 + 打包成可调用对象
                     │       负责人：tilelang/jit/kernel.py
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Python 里的 kernel 对象（callable）                                     │
└──────────────────────────────────────────────────────────────────────┘
                     │  ⑥  运行：kernel(a, b)
                     │       负责人：src/cuda/runtime.cc + Driver API
                     ▼
                    GPU 上真正执行的 kernel
```

后面每一节挑一个阶段细讲。**你现在只需要记住这张图**，读到后面章节任何一段代码你都能对应到某个阶段。

> **图里那些没见过的硬件/编译名词，现在都不用懂**，只要知道它们大致"是什么类别的东西"即可，后面各章会展开：
>
> | 名词 | 一句话（先有个印象即可） | 详解 |
> |---|---|---|
> | **Tensor Core** | GPU 上专门做矩阵乘加的硬件单元，比普通核心快很多 | 第 6/7 章 |
> | **wgmma / mma** | 调用 Tensor Core 的机器指令；`wgmma` 是 Hopper 上"一个 warp group 一起做"的版本 | 第 6 章 |
> | **mbarrier** | Hopper 起的硬件同步器（memory barrier），异步搬数据时用它通知"到货了" | 第 6 章 |
> | **NVRTC** | NVIDIA 的运行时编译库，把 CUDA 源码字符串直接编成 GPU 机器码 | 1.6 / 第 8 章 |
> | **PTX / cubin** | CUDA 编译的中间产物：`PTX` 是可读的类汇编，`cubin` 是最终二进制 | 第 8 章 |
> | **SM90 / SM100** | GPU 的**架构版本号**：SM80=Ampere(A100)、**SM90=Hopper(H100)**、SM100=Blackwell。本书以 SM90 为主 | 全书 |
>
> 换句话说：**这一章你只需建立"编译分几个阶段"的骨架**，具体硬件概念遇到时再回来查这张表或去对应章节。

## 1.3 阶段一 · DSL 解析：从 Python AST 到 TIR

`@tilelang.jit` 是入口装饰器，定义在 [`tilelang/jit/__init__.py`](../../tilelang/jit/__init__.py) 里。
它会返回一个 `JITImpl` 包装器对象（**不是** `JITKernel`——`JITKernel` 要等你调 `.compile()` 才拿到），
**并不立刻编译**——真正的编译发生在你调用 `.compile(...)` 时。

> 📌 小澄清：**`JITImpl` vs `JITKernel`**
> - `JITImpl` = 被 `@tilelang.jit` 包好的**函数入口**。它记住了配置（target、pass_configs…），
>   还能延迟到你给具体参数的时候再决定要不要触发编译。上面 1.3 提到的 `.get_tir()` / `.compile()` /
>   `.get_kernel_source()` 都是它身上的方法。
> - `JITKernel` = 编译**完成之后**得到的"可调用 kernel 对象"。`kernel(a, b)` 里的 `kernel` 就是它。
>   它身上有 `.get_kernel_source()` / `.get_host_source()` / `.get_profiler()` 这些方法。

那 Python 函数体里的 `T.Kernel(...)`、`T.Pipelined(...)` 又是什么？它们**不是普通 Python 函数调用**——
它们是运行在一个特殊的"解析模式"下的构造器。这套机制来自 TVM 的 TVMScript 解析器，
本仓库在 [`tilelang/language/parser/`](../../tilelang/language/parser) 里做了扩展。

解析的效果是：**Python 函数体不会真的执行**，它的 AST 会被扫描，
每遇到一个 `T.xxx` 就生成一个对应的 TIR 节点。最终生成的对象叫 `PrimFunc`。

一个大致简化后的心智模型：

| Python 代码里的 | 生成的 TIR 节点大致是 |
|---|---|
| `A: T.Tensor(...)` | 一个 `Buffer` 声明 |
| `with T.Kernel(nx, ny, threads=T) as (bx, by):` | 一个带 `thread_binding` 的 `For` 序列 |
| `T.alloc_shared(...)` | 一个 `Allocate` 节点，指向 `shared` 存储 |
| `T.alloc_fragment(...)` | 一个 `Allocate` 节点，指向 `local.fragment` |
| `for ko in T.Pipelined(N, num_stages=3):` | 一个带 `software_pipeline_*` 注解的 `For` |
| `T.copy(src, dst)` | 一个自定义的 tile-level intrinsic call |
| `T.gemm(A, B, C)` | 一个自定义的 tile-level intrinsic call |

**"自定义 intrinsic"** 是什么？就是本仓库自己定义的一批"函数名"，
它们在这个阶段还只是名字（`Call`节点），没有具体实现。等到 Lowering 走到某个 pass 时，
再根据当前的硬件和 layout 展开成真正的指令序列。

> 💡 **想自己看一眼？** 下面这些 API 全都在源码里能查到（不是我编的）：
>
> - `matmul.get_tir(**kwargs)` → 阶段一结束时的 `PrimFunc`（仅前端解析后、还没跑任何 pass）。
>   定义在 [`tilelang/jit/__init__.py`](../../tilelang/jit/__init__.py) 的 `JITImpl.get_tir`。
> - `tilelang.lower(prim_func, target="cuda")` → 阶段二结束时的 `CompiledArtifact`，
>   里面有 `host_mod` / `device_mod` / `kernel_source` 三个字段。
>   定义在 [`tilelang/engine/lower.py`](../../tilelang/engine/lower.py) 的 `lower()`，
>   返回类型在 [`tilelang/engine/param.py`](../../tilelang/engine/param.py) 的 `CompiledArtifact`。
> - `matmul.compile(**kwargs)` → 完整走完 6 个阶段，返回一个可调用的 `JITKernel`。
> - `kernel.get_kernel_source()` → 阶段三输出的 CUDA 源码字符串。
> - `kernel.get_host_source()` → 阶段五生成的 host 侧代码。
>
> 一段完整的"看每个阶段中间产物"的脚本长这样（保存为 `look_around.py` 就能跑）：
>
> ```python
> import tilelang
> import tilelang.language as T
>
> @tilelang.jit
> def matmul(A, B, block_M: int, block_N: int, block_K: int):
>     # ... 和 quickstart 里一样 ...
>     M, N, K = T.const("M, N, K")
>     dtype = T.float16
>     accum_dtype = T.float32
>     A: T.Tensor((M, K), dtype)
>     B: T.Tensor((K, N), dtype)
>     C = T.empty((M, N), dtype)
>     with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
>         A_shared = T.alloc_shared((block_M, block_K), dtype)
>         B_shared = T.alloc_shared((block_K, block_N), dtype)
>         C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
>         T.clear(C_local)
>         for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
>             T.copy(A[by*block_M, ko*block_K], A_shared)
>             T.copy(B[ko*block_K, bx*block_N], B_shared)
>             T.gemm(A_shared, B_shared, C_local)
>         T.copy(C_local, C[by*block_M, bx*block_N])
>     return C
>
> cfg = dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
>
> # ---- 阶段一的产物：解析出来的 TIR ----
> prim_func = matmul.get_tir(**cfg)
> print("===== TIR (after parse, before any pass) =====")
> print(prim_func.script())
>
> # ---- 阶段二的产物：lower 后的 host / device IRModule ----
> artifact = tilelang.lower(prim_func, target="cuda")
> print("===== device_mod (after lowering pipeline) =====")
> print(artifact.device_mod.script())
>
> # ---- 阶段三的产物：CUDA 源码字符串（lower() 内部已经跑过一次 codegen） ----
> print("===== kernel_source (CUDA C++) =====")
> print(artifact.kernel_source)
>
> # ---- 完整走完 6 个阶段：拿到 JITKernel，可以直接调 ----
> kernel = matmul.compile(**cfg)
> print("===== kernel.get_kernel_source() =====")
> print(kernel.get_kernel_source())
> print("===== kernel.get_host_source() =====")
> print(kernel.get_host_source())
> ```
>
> 你会看到一段像 Python 又不像 Python 的东西——那就是 TIR 的文本形态。

## 1.4 阶段二 · Lowering Pipeline：几十个 Pass 依次改写 TIR

这是本书的**主线**——从第 4 章到第 7 章都在讲这一阶段。这里先给你一个鸟瞰。

编排器在 [`tilelang/engine/lower.py`](../../tilelang/engine/lower.py) 和
[`tilelang/backend/pass_pipeline/`](../../tilelang/backend/pass_pipeline) 里。
它做的事情用一句话说清楚：

> 拿到刚解析出来的 `PrimFunc`，按预定义顺序应用几十个 `Pass`，每个 Pass 都产出新的 `PrimFunc`。

真正的 Pass 分两类：

- **通用 pass**（对所有后端都跑），在 [`src/transform/`](../../src/transform) —— 一部分来自 TVM 上游（`3rdparty/tvm/src/tir/transforms/`），
  一部分是 TileLang 自己加的（比如 `inject_pipeline.cc`、`storage_rewrite.cc` 的 TL 版本）
- **CUDA-only pass**，在 [`src/cuda/transform/`](../../src/cuda/transform)
  —— 比如 `producer_consumer_ws.cc`（warp specialization）、`multi_version_buffer_rewriter.cc`（多版本 buffer 展开）

Pass 的运行顺序**极其重要**（第 5 章会详细讲），因为每个 pass 都对输入 IR 有前置假设。
举个例子：`ProducerConsumerWarpSpecialized` 依赖 `InjectPipeline` 先把 `T.Pipelined` 展开成显式的双缓冲循环，
才能识别出哪些是 producer 步骤、哪些是 consumer 步骤。

## 1.5 阶段三 · Codegen：把 TIR 打印成 CUDA 源码字符串

Lowering 结束后，`PrimFunc` 已经"贴地"了：没有高级 tile-op、只剩下 CUDA / PTX 层面能对应的指令。
这时候由 [`src/cuda/codegen/codegen_cuda.cc`](../../src/cuda/codegen/codegen_cuda.cc)（226 KB 的大文件）
把它**一行一行打印**成 CUDA C++ 源码字符串。

打印的原理是：一个访问者对每种 IR 节点写一段 `PrintExpr(...)` / `PrintStmt(...)`。
比如遇到 `IfThenElse`，就打印 `if (cond) { ... } else { ... }`。

> 想验证？quickstart 最后就打印了这段字符串：
> ```python
> cuda_source = matmul_relu_kernel.get_kernel_source()
> print("Generated CUDA kernel:\n", cuda_source)
> ```

## 1.6 阶段四 · NVRTC：CUDA 源码 → cubin

`codegen_cuda.cc` 输出的是**字符串**，还得有人把它编译成 GPU 能加载的机器码。
本仓库不用 `nvcc` 命令行（那太慢），而是用 NVIDIA 提供的运行时编译库 **NVRTC**（`libnvrtc.so`），
通过 [`tilelang/contrib/nvcc.py`](../../tilelang/contrib/nvcc.py) 和
[`tilelang/contrib/nvrtc.py`](../../tilelang/contrib/nvrtc.py) 里的包装完成。

编出来的是 `cubin`（或 `fatbin`，即多 arch 打包），二进制形态。

> **想看这一阶段的产物？** cubin 是二进制看不了，但它的上一级中间表示 PTX 是可读文本，`JITKernel` 提供了导出接口：
> ```python
> matmul_relu_kernel.export_ptx("/tmp/matmul.ptx")   # 阶段四的产物（PTX 汇编）
> # 打开 /tmp/matmul.ptx，能看到 .visible .entry matmul_kernel_0(...) 之类
> ```
> 对照阶段三打印出的 CUDA 源码，你会看到同一个 kernel 从 C++ 源码变成了更底层的 PTX 汇编。

## 1.7 阶段五 · Host stub + JIT 包装

到这一步 GPU 上的 kernel 已经就绪，但 Python 侧还得知道：

- 参数怎么打包（`torch.Tensor` 里拿出 data pointer、stride、shape）
- 什么时候把 kernel launch 上去
- 生成的 cubin 缓存到哪里，避免每次都重编

这些事全在 [`tilelang/jit/kernel.py`](../../tilelang/jit/kernel.py) 和
[`tilelang/cache/kernel_cache.py`](../../tilelang/cache/kernel_cache.py) 里完成。
最终你拿到的 `kernel` 对象是一个 `JITKernel`，`kernel(a, b)` 就等价于
"打包参数 → 找到对应 cubin → 通过 CUDA Driver launch"。

> **想看这一阶段生成的 host stub？** 用 `get_host_source()` 就能把 Python 侧那段"打包参数 + launch"的代码打印出来：
> ```python
> print(matmul_relu_kernel.get_host_source())   # 阶段五生成的 host 侧胶水代码
> ```
> 你会看到它从 `torch.Tensor` 里取 data pointer / stride / shape，再调底层的 launch 接口——这正是上面三条要点的落地。（1.9 会把这些产物连同前几个阶段一起打印一遍。）

## 1.8 阶段六 · 真正的运行

`kernel(a, b)` 的底层通过 [`src/cuda/runtime.cc`](../../src/cuda/runtime.cc) 里的封装
调 CUDA Driver 的 `cuLaunchKernel`（或 PDL / cluster launch 的变体）。
从这里开始就是 GPU 硬件层面的事情了，本书基本不再讨论——除非某个 pass 要专门配合硬件某个特性
（比如 SM90 的 wgmma、SM100 的 tcgen05）。

---

## 1.9 亲手做一遍

强烈建议你现在就跑一遍 `quickstart.py`，并顺便**打印每个阶段的中间产物**。
在 tilelang 仓库根目录：

```bash
python examples/quickstart.py
```

跑通之后，在 `quickstart.py` 里 `matmul_relu_kernel = matmul.compile(...)` 之后插入下面几行
（`M / N / K / block_M / block_N / block_K` 用原脚本里已经定义的那些变量即可）：

```python
# 阶段一之后的 TIR（前端解析后、未 lower）
print(matmul.get_tir(M=M, N=N, K=K,
                     block_M=block_M, block_N=block_N, block_K=block_K).script())

# 阶段三之后的 CUDA 源码（原脚本最后一行本来就在打印它）
print(matmul_relu_kernel.get_kernel_source())

# 阶段五生成的 host 侧代码
print(matmul_relu_kernel.get_host_source())
```

对着这两段输出，回头再看 1.2 那张图，你会觉得**"编译"这个词具体了不少**。

## 1.10 本章要带走的三件事

1. **一个 kernel 的编译不是一步完成，而是"6 个阶段 + 几十个 pass"** 的流水线。
2. **每个阶段都有明确的负责文件**——记不住图没关系，看目录名就够猜个八九不离十。
3. **中间形态可以看**——三个 API 一辈子够用了：
   - `matmul.get_tir(**cfg).script()` 看**阶段一之后**的 TIR（未 lower）
   - `tilelang.lower(pf, target="cuda").device_mod.script()` 看**阶段二之后**的 device 侧 IR
   - `kernel.get_kernel_source()` / `kernel.get_host_source()` 看**阶段三和阶段五之后**生成的源码

   这三个是你后面调 pass / 看 bug 的**最重要武器**。

---

下一章 [第 2 章 · TVM / TIR 基础概念](./02_tvm_tir_basics.md) 我们把阶段一输出的东西—— `PrimFunc` / `Buffer` / `Stmt` / `Expr` —— 一个个拆开看。
