# 附录 F · Eager JIT 与 CuTe DSL 分支

> **本附录目标**
> 讲清楚 TileLang 里两条**存在但正文没提**的"支路"：
> 1. **Eager JIT** — 一种不用显式 `return PrimFunc` 的写法，看起来"跟写普通 Python 一样"
> 2. **CuTeDSL codegen 分支** — 一条完全不同的后端路径，把 TileLang IR 翻译成 NVIDIA 的 CuTe DSL 而不是直接生成 CUDA C
>
> 都不是**正常写 kernel** 必须了解的东西，但当你读到 `tilelang/language/eager/`、`tilelang/contrib/cutedsl/`、或者看到 `TILELANG_TARGET=cutedsl` 时，希望这份附录能让你 5 分钟对号入座。

---

## F.1 Eager JIT vs Lazy JIT：TileLang 的两种 `@tilelang.jit`

翻开 [tilelang/jit/__init__.py](../../tilelang/jit/__init__.py)（第 547-627 行 `jit` 定义），你会在 docstring 里看到一段极关键的话：

> Supports two execution modes (automatically inferred):
> - **lazy**: Function returns PrimFunc explicitly. Returns compiled kernel object.
> - **eager**: Function uses DSL builder pattern. Executes kernel immediately.

也就是说 **`@tilelang.jit` 是同一个装饰器，但函数体的写法决定它走 lazy 还是 eager**——不用你显式选。

### Lazy 模式（就是正文里一直用的）

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(...) as (bx, by):
            ...
    return main       # ← 显式 return PrimFunc

kernel = matmul(1024, 1024, 1024, 128, 128, 32)   # 拿到编译好的 kernel 对象
C = kernel(A, B)                                    # 后续多次调用不重编译
```

**特点**：
- 外层函数是**工厂**，接收编译期形状参数
- 内层 `@T.prim_func` 才是 kernel
- 调用工厂 → 拿到 `kernel` 对象 → 用它去跑

### Eager 模式（正文没提的另一种）

Eager 模式**没有内层 `@T.prim_func`，也没有 `return`**——你直接在函数体里"操作 tensor"，装饰器帮你把它 trace 成 IR 并立刻编译执行：

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def add(A, B):
    # 注意：没有 @T.prim_func、没有 return
    C = T.empty_like(A)
    with T.Kernel(T.ceildiv(A.shape[0], 128), threads=128) as (bx,):
        offset = bx * 128
        for i in T.Parallel(128):
            C[offset + i] = A[offset + i] + B[offset + i]
    return C          # ← 这里 return 的不是 PrimFunc，是"输出 tensor"

# 直接用，第一次调用会 trace + compile + execute
C = add(A, B)
```

**特点**：
- 更像 PyTorch 的 eager 感觉
- 一次调用 = trace + compile + run，之后再调同样 signature 是走缓存
- 教程 notebook 在 [examples/eager_jit/](../../examples/eager_jit/)：`eagerjit.zh.ipynb` 是完整中文教程

### 底层实现：`eager_jit="phase1" / "phase2" / "none"`

来看 [tilelang/language/eager/builder.py](../../tilelang/language/eager/builder.py) 的关键片段：

- `prim_func(func, eager_jit=True)` 是 eager 入口
- Builder 内部维护 3 个 phase：`"none"` / `"phase1"` / `"phase2"`
- **phase1** = 第一遍走一遍函数体，收集"这个 Python 函数长什么 IR 骨架"
- **phase2** = 拿到具体输入 tensor 后，把 shape/dtype specialize 进去
- 组合起来达到"看起来一次调用就完成 trace + compile"的效果

一般用户**不用管这三个 phase**——它们是内部实现。你只需要知道：**eager 模式下同一个函数首次调用慢（要 trace + compile），之后按输入 signature 缓存**。

### Eager 模式的 Ref / OutTensor

Eager builder 引入两个额外类型：

- `Ref` — 对可变 tensor 的引用（因为 eager 允许写"读原地改"）
- `OutTensor` — 声明"这个 tensor 是函数的输出"

`T.empty_like` / `T.zeros_like` 之类的 builder 会自动帮你生出 `OutTensor`。

### 什么时候用哪个

| 场景 | 选 |
|---|---|
| 需要**多种形状复用同一份 kernel** | **Lazy** —— 编译一次，多个 shape 走同一份工厂（用第 12 章的 `T.dynamic`） |
| 需要**pass config、compile flags** 精细控制 | **Lazy** |
| 想快速原型、像写 PyTorch 一样 | **Eager** |
| 教学 / notebook 演示 | **Eager** |
| 正式项目里的 hot-path kernel | **Lazy**（更可控） |

Eager 目前不建议用来堆生产 kernel——builder 里还有些边角 API 会变。

---

## F.2 CuTeDSL Codegen：不是 DSL，是"另一条 codegen 后端"

### 它是什么

正常路径：**TileLang PrimFunc → 一堆 pass → CUDA C 源码 → nvcc**

CuTeDSL 分支：**TileLang PrimFunc → 一堆 pass → CuTe DSL（cutlass.cute）→ CUTLASS 编译**

真实事实（这个仓库里能验证的）：

- **源码入口**：[src/cuda/codegen/codegen_cutedsl.cc](../../src/cuda/codegen/codegen_cutedsl.cc)（137KB）
- **Python 侧**：[tilelang/contrib/cutedsl/](../../tilelang/contrib/cutedsl/)（一堆 Python 模块封装 CUTLASS DSL 原语）
- **触发方式**（可靠的通用做法是指定 execution backend）：
  ```python
  # 装饰器上指定
  @tilelang.jit(execution_backend="cutedsl")
  # 或 compile 时指定
  kernel = tilelang.compile(func, execution_backend="cutedsl")
  ```
  ```bash
  # 或用环境变量设默认后端
  TILELANG_EXECUTION_BACKEND=cutedsl python your_script.py
  ```
  > ⚠️ 你可能在 CI / examples 里看到 `TILELANG_TARGET=cutedsl`。它**不是**通用的后端开关——
  > 那是 `examples/conftest.py` 和部分示例测试读取的**测试约定**，用来在 cutedsl 那轮 CI 里
  > 自动 skip / xfail 掉不支持的用例。要真正切后端，用上面的 `execution_backend`。
- **依赖**：`import cutlass` / `import cutlass.cute as cute`（不是 CUTLASS C++ 头文件，是 CUTLASS 的 Python DSL）

### 为什么会存在这条路径

TileLang 本来就是"高层 tile IR"，而 CuTe（CUTLASS 的核心概念）也是"layout + tile" 抽象。两者其实在**同一个概念层次**。所以有一条 codegen 分支把它们直接对接起来，好处是：

1. **复用 CUTLASS 已经写好的 Hopper / Blackwell 微内核**（`tcgen05` MMA、TMA descriptor 生成等）
2. **减少 TileLang 自己维护 PTX 生成的负担**
3. **能利用 CUTLASS Python DSL 的调试和 profile 工具**

### 但——它有实际限制

来自真实 conftest（[examples/conftest.py](../../examples/conftest.py) 第 41-46 行）和真实 example 的 skip 标记：

```python
# CuTeDSL backend: known failures / unsupported cases
CUTEDSL_KNOWN_FAILURES = {
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn",   # flaky
    "deepseek_v4/test_tilelang_example_deepseek_v4.py::test_example_act_quant",  # 未实现 FP4 quant lowering
}

# 一堆 test 里显式 skipif：
@pytest.mark.skipif(_is_cutedsl,
                    reason="CuTeDSL backend does not support alloc_global yet")
```

**已知限制清单**（截至我读到的版本）：

| 限制 | 影响 |
|---|---|
| 不支持 `T.alloc_global` | 用到 global scratch 的 kernel（如 flash-decoding）跑不了 |
| 不支持 DeepSeek V4 FP4 activation quant | 部分极致量化 kernel 走不通 |
| 某些 handle tensor 类型不兼容 | grouped GEMM 的一些变种要禁用 |
| 并行 pytest 下 flaky | 有些测试单独跑没事、并行跑挂 |

因此**CuTeDSL 分支目前更像"实验性 codegen 后端"**——TileLang 主 devs 在探索"完全走 CUTLASS 路径能不能更省心"，但还没到"日常推荐"程度。

### `tilelang/contrib/cutedsl/` 里到底装了什么

按文件功能拆解（16 个模块）：

| 模块 | 干啥 |
|---|---|
| `utils.py` | 类型、shape 转换工具 |
| `cpasync.py` | `cp.async` 对应的 CuTe 封装 |
| `gemm_v1.py` / `gemm_v2.py` | 两代 GEMM 微内核 wrapper |
| `gemm_tcgen05.py` | Blackwell tcgen05 MMA wrapper |
| `ptx_mma.py` | PTX 级 MMA 指令 |
| `ldsm.py` | `ldmatrix` 硬件指令 |
| `reduce.py` | warp / block 级规约 |
| `math.py` / `ieee_math.py` | 数学 intrinsic |
| `atomic.py` | atomic 操作对接 |
| `quantize.py` | 量化解包（呼应第 14 章） |
| `warp.py` | warp 级原语 |
| `threadblock_swizzle.py` | tile swizzle 布局 |
| `grid_sync.py` | grid-level 同步 |

从 `__init__.py` 还可以看到它 re-export 了 `cutlass.cute` 里一堆东西：

```python
from cutlass.cute.arch import sync_threads, sync_warp
from cutlass.cute.arch import alloc_smem, get_dyn_smem
from cutlass.cute.arch import warpgroup_reg_alloc, warpgroup_reg_dealloc
from cutlass.cute import make_tensor, make_rmem_tensor, recast_ptr
...
```

**含义**：这条分支下**TileLang 的 codegen 输出的不是 CUDA C，而是 Python 代码**——那份 Python 代码调用 `cutlass.cute.*`，运行时由 CUTLASS DSL 编译到 PTX。

### 使用方式速查

```python
# 方式 A：装饰器上指定
@tilelang.jit(execution_backend="cutedsl")
def matmul(...): ...
```

```python
# 方式 B：手动 compile 时指定
kernel = tilelang.compile(func, execution_backend="cutedsl")
```

```bash
# 方式 C：环境变量设默认后端（对整个进程生效）
TILELANG_EXECUTION_BACKEND=cutedsl python your_matmul.py
```

**推荐流程**：写一个 kernel，先默认路径（CUDA C）跑通，再用 `execution_backend="cutedsl"` 跑一遍看是否可用——**能用就用**（部分情况会有性能收益）；报错就 fallback 回默认路径，别在 cutedsl 分支上死磕。

---

## F.3 两条支路的定位对照

正文 10 章 + 前面附录 A-E 涵盖的是 **99% 用户的日常路径**：

```
用户 Python DSL
    │
    ▼
[Lazy JIT @tilelang.jit + @T.prim_func]
    │
    ▼
TileLang PrimFunc → passes → CUDA C → nvcc → .cubin → 跑
```

Eager JIT 是**同一条路径的 Python 语法糖**：

```
用户 Python DSL（更像 PyTorch）
    │
    ▼
[Eager JIT @tilelang.jit，无 return PrimFunc]
    │
    ▼ (builder trace)
TileLang PrimFunc → passes → CUDA C → nvcc → .cubin → 跑
```

CuTeDSL 是**替换 codegen 的另一条路径**：

```
TileLang PrimFunc → passes
                        │
                        ├─ 默认路径 → CUDA C → nvcc
                        │
                        └─ execution_backend="cutedsl" → Python + cutlass.cute
                                                        → CUTLASS 内部编译
                                                        → PTX
```

三者**互不排斥**：你可以同时用 Eager JIT + CuTeDSL 后端（`@tilelang.jit(execution_backend="cutedsl")` 而且函数体是 eager 风格）——只是这种组合目前踩坑最多，不推荐生产用。

---

## F.4 什么时候要读这一节

如果你遇到下面情况，回来翻这份附录：

- 你在 `examples/eager_jit/` 或社区教程里看到「没有 `@T.prim_func`、没有 `return main`」的写法 → **F.1**
- 你 pull 了最新代码发现 CI 里一堆 `TILELANG_TARGET=cutedsl` 的用例标记（examples 的测试约定）→ **F.2**
- 你想在自己 kernel 上试试 CUTLASS Python DSL 的 profiler → **F.2 使用方式速查**
- 你在 `tilelang/contrib/cutedsl/` 里看到 `wgmma_wait_group`、`warpgroup_reg_alloc` 之类 → 那是 CUTLASS DSL 名字，跟 TileLang 的 `T.ws` / `T.no_set_max_nreg` 概念对应

---

## F.5 小结

- **Eager JIT vs Lazy JIT**：`@tilelang.jit` **自动推断**，看你有没有 `return PrimFunc`。Eager 更适合原型 / notebook，Lazy 更适合生产。
- **CuTeDSL 分支**：另一条 codegen 后端，用 `execution_backend="cutedsl"`（或环境变量 `TILELANG_EXECUTION_BACKEND=cutedsl`）打开。目前**实验性**，`alloc_global` 等还不支持，部分 kernel 会 fallback。
- 这两条支路都不是"必须掌握"，但当你翻到 `tilelang/language/eager/` 或 `tilelang/contrib/cutedsl/` 时，希望你知道它们在做什么。

至此 cookbook 涵盖了 TileLang 从**入门 → PrimFunc → pass → 软件流水 & WS → layout → codegen → JIT/runtime → 贡献 → 调试 → 控制流/reduce/atomic → Hopper 特性 → 量化 → Eager + CuTeDSL**的完整学习路径。相信读到这里的你，已经可以：

1. 独立读懂 TileLang 里几乎任何一份 kernel 源码
2. 定位并 debug 编译 pass 层面的 bug
3. 写自己的高级 kernel（如 flash attention 变种、W4A16 GEMM）
4. 参与 TileLang 社区代码贡献

祝玩得开心 🚀
