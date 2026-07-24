# 第 4 章 · Pass 系统与 Pass Pipeline

> **TL;DR**：本章打开"编译"这个黑盒的**中层**——pass。
> 通过阅读 TileLang 里几个真实存在的最小 pass，回答四个问题：
> 
> 1. **什么是 pass？** —— 一个 `IRModule → IRModule` 的函数。
> 2. **怎么写一个 pass？** —— 装饰器 `@prim_func_pass` + Visitor / Mutator。
> 3. **pass 怎么被串起来？** —— `PassPipeline` 容器 + 后端注册。
> 4. **pass 怎么读用户传下来的开关？** —— `PassContext.current().config.get(...)`。
> 
> **本章会读到的真实源码**：
> 
> - [`3rdparty/tvm/python/tvm/tirx/transform/function_pass.py`](../../3rdparty/tvm/python/tvm/tirx/transform/function_pass.py)（`prim_func_pass` 装饰器）
> - [`3rdparty/tvm/python/tvm/ir/transform.py`](../../3rdparty/tvm/python/tvm/ir/transform.py)（`Pass` / `PassContext`）
> - [`tilelang/analysis/nested_loop_checker.py`](../../tilelang/analysis/nested_loop_checker.py)（真实 **只读** pass 样本）
> - [`tilelang/metal/transform/mark_host_metal_context.py`](../../tilelang/metal/transform/mark_host_metal_context.py)（真实 **改写型** pass 样本）
> - [`tilelang/backend/pass_pipeline/pipeline.py`](../../tilelang/backend/pass_pipeline/pipeline.py)（`PassPipeline` 容器）
> - [`tilelang/cuda/pipeline.py`](../../tilelang/cuda/pipeline.py)（**CUDA 后端**完整 pass 顺序表；本章简单引用，下一章逐行拆）
> - [`tilelang/engine/lower.py`](../../tilelang/engine/lower.py)（顶层 `lower(...)` 怎么调 pipeline）
> 
> **前置**：读完 [第 2 章](./02_tvm_tir_basics.md)（PrimFunc / Stmt / Expr / IRModule）。

---

> 📌 **读之前先明确一件事：本章前半 API 都是 TVM 的，不是 TileLang 自己造的。**
> 
> TileLang **没有重写** pass 框架，而是**直接沿用 TVM 的那一套**——所以 4.1–4.4 你会看到大量
> `tvm.tirx.transform.prim_func_pass` / `PyStmtExprVisitor` / `PyStmtExprMutator` /
> `@functor.mutator`，它们全部来自 vendored 在 `3rdparty/tvm/` 里的 TVM（tirx 分支），
> 是**写 TileLang pass 时必须掌握的底层工具**，不掌握后面第 5、6、10 章的源码你根本读不动。
> 
> 真正**属于 TileLang 自己**的东西集中在后半：
> 
> - **4.5 `PassPipeline`**：一个 30 行的轻量容器，负责把几十个 pass **按顺序、按后端**串起来
> - **4.6 `PassConfigKey`**：把跨 pass 的全局开关（`tl.disable_warp_specialized` 等）收敛成枚举
> - **4.7 亲手写 pass**：Visitor + `tl.tileop.copy` 这个 TileLang 自定义 intrinsic 合起来用
> 
> 所以本章的读法建议是：**4.1–4.4 当"TVM pass 框架速成"，4.5 之后才是"TileLang 里 pass 系统的独特之处"**。
> 至于 pass 具体都在干什么、CUDA 后端跑了哪几十个——那是 [第 5 章](./05_lowering_pipeline.md) 的主题。

---

## 4.1 pass 是什么：一个 `IRModule -> IRModule` 的函数

在 TVM / TileLang 里，一个 **pass** 就是一个"接受 IRModule、返回 IRModule"的函数——
或者更准确地说：**一个能作用于 IRModule 上的、有名字有元信息的可调用对象**。

```
       ┌───────────────┐              ┌───────────────┐
       │  IRModule A   │──►  pass  ──►│  IRModule B   │
       └───────────────┘              └───────────────┘
        (输入是一棵 TIR)                (输出还是一棵 TIR，
                                        但结构 / 语义已经变了)
```

TVM 提供两种粒度的 pass：

| 类型                 | 每次处理什么                           | 装饰器                                  |
| ------------------ | -------------------------------- | ------------------------------------ |
| **`ModulePass`**   | 一整个 `IRModule`（可跨函数）             | `@tvm.transform.module_pass`         |
| **`PrimFuncPass`** | 一次一个 `PrimFunc`（TVM 帮你遍历模块里每个函数） | `@tvm.tirx.transform.prim_func_pass` |

> ⚠️ **注意 tirx**：本仓库 vendored 的 TVM 是"下一代 TIR"分支，
> Python 命名空间是 `tvm.tirx.transform.prim_func_pass`**（不是 `tvm.tir.transform`）**。
> C++ 侧节点类型也是 `tirx::PrimFunc` 等（第 2 章已经提过）。

TileLang 里**几乎所有 Python pass** 都是 `PrimFuncPass`——因为大多数变换都以函数为单位。

## 4.2 最小可运行的 pass：一个空 pass

先来看最简单的"空 pass"（什么也不做，只把函数原样返回）：

```python
from tvm.tirx.transform import prim_func_pass

@prim_func_pass(opt_level=0)
def IdentityPass(func, mod, ctx):
    # func: tvm.tirx.PrimFunc  —— 当前正在处理的函数
    # mod : tvm.IRModule        —— 它所在的模块（有时候要跨函数查东西才用）
    # ctx : tvm.ir.transform.PassContext —— 全局配置
    return func     # 原样返回
```

调用它：

```python
new_mod = IdentityPass(mod)   # pass 对象可以直接 (mod) 调用
```

### 概念卡：**pass 是不可变（immutable）变换**

你**不能**"就地"改 IR——TIR 里所有节点都是 immutable（`class Stmt` 底层是 `ObjectRef`，指向 unique_ptr）。
所以每个 pass 都必须**构造新节点**并返回。

这听起来低效，其实很自然：

- 你在写 Python 时**引用**同一个子树多次时是"共享"（引用计数 +1），不产生拷贝
- 只有你**改**的那一小部分才会造新节点
- 天然线程安全、天然可 diff（`stmt_a.same_as(stmt_b)` 直接看引用是否相同）

### 概念卡：**你写的是 Python，跑的是 C++——TVM 的两层架构**

后面你会看到零零散散冒 C++（`tirx::PrimFunc`、"C++ 侧宏"、"C++ 侧的调度表"…），别慌，这不是跑题。TVM/TileLang 是**双语言双层架构**，理解一次以后全书通用：

```
┌────────────────────────────────────────────────────────────────┐
│  Python 层（用户接触到的一切）                                    │
│  ─ tvm.tirx.PrimFunc / Stmt / Call / For …  （类型的"壳"）        │
│  ─ PyStmtExprVisitor / PyStmtExprMutator    （能被继承的基类）    │
│  ─ @prim_func_pass / @functor.mutator       （装饰器）            │
└────────────────────────────────────────────────────────────────┘
                       ▲          │
                       │  FFI（外部函数接口）—— pybind11 / ctypes 那一套
                       │          ▼
┌────────────────────────────────────────────────────────────────┐
│  C++ 层（真正干活的地方）                                          │
│  ─ tirx::PrimFuncNode / StmtNode / CallNode …                    │
│  ─ StmtVisitor / StmtExprMutator             （C++ 基类）        │
│  ─ Pass / PassNode                            （pass 的"本体"）  │
│  ─ 遍历调度表（visit_for_ 派发到 VisitStmt_(For)）                │
└────────────────────────────────────────────────────────────────┘
```

**Python 类都是 C++ 类的"壳"（thin wrapper）**：`tvm.tirx.PrimFunc` 里没有 body、params 这些数据，它只是持有一个 C++ `PrimFuncNode` 的智能指针（`ObjectRef`）。**你在 Python 里能做的所有事，最终都会走 FFI 掉到 C++ 里执行**——包括构造 IR 节点、遍历 AST、`.with_attr(...)`、`pass_obj(mod)`。

这一层架构会**在 pass 系统里以三种形式冒头**（这也是为什么本章不断出现 C++）：

1. **命名约定被 C++ 决定**：`visit_for_` 的下划线来自 C++ 侧宏 `TVM_FFI_DEF_TVM_FFI_METHOD`；`PyStmtVisitor` 的 `Py` 前缀是"从 C++ 侧 `StmtVisitor` 派生出来给 Python 用的版本"。
2. **装饰器把 Python 类"注册"进 C++ 派发表**：`@functor.mutator` / `@tirx.functor.visitor` 干的事就是——**把你 Python 里写的 `visit_for_` 方法登记到 C++ 侧的节点派发表里**，这样 C++ 遍历 AST 走到 `For` 节点时能反过来回调你的 Python 方法。忘写这个装饰器就会 raise `Unregistered functor`。
3. **性能敏感的地方在 C++**：整棵 AST 的递归下降、pass pipeline 的调度、真正的 IR 转换（`Simplify` / `LowerTileOp` / `FlattenBuffer`）**都在 C++ 实现**。Python 侧写的 pass 是"少数几个只读检查 + 顶层胶水"。

**读本章 / 全书时的实用推论**：
- 看到 `tirx::XxxNode` 这种带 `::` 的名字 = **C++ 侧的定义**，去 `3rdparty/tvm/include/tvm/tir/` 或 `src/tir/` 找源码。
- 看到 `tvm.tirx.Xxx` 或 `PyXxx` = **Python 侧的壳**，去 `3rdparty/tvm/python/tvm/tirx/` 找。
- **同一个 IR 节点在两层都有对应类**，`For` ↔ `ForNode`、`PrimFunc` ↔ `PrimFuncNode`。
- 大多数时候你**不需要**碰 C++，只在两种情形下需要：**（a）你的 pass 性能不够、要下沉到 C++**（第 10 章会讲）；**（b）你在读一个不完全懂的 pass 源码、想去 C++ 侧看真实实现**。

后面遇到"C++ 侧的调度表"、"C++ 侧的类型"这种表述，回来看这张图就懂了。

## 4.3 只读 pass：用 Visitor 检查非法结构

**用途**：不改 IR，只是"扫一遍看看有没有非法模式"。
[`tilelang/analysis/nested_loop_checker.py`](../../tilelang/analysis/nested_loop_checker.py) 就是一个真实例子。

关键结构（截取自源码）：

```python
from tvm import tirx
from tvm.tirx import For, Call, PrimFunc, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass


@tirx.functor.visitor              # ← ① 装饰器：告诉 TVM 这是一个 visitor
class _NestedLoopCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.in_parallel_context = False

    def visit_for_(self, op: For) -> None:   # ← ② 重载"访问 For 节点"
        if op.kind == tirx.ForKind.PARALLEL:
            ...
            self.in_parallel_context = True
            super().visit_for_(op)           # ← ③ 递归下降到 body
            self.in_parallel_context = False
            return
        elif is_pipelined_for(op):
            if self.in_parallel_context:
                raise ValueError("Pipelined inside Parallel not allowed")
        super().visit_for_(op)

    def visit_call_(self, op: Call) -> None:  # ← ② 重载"访问 Call 节点"
        if self.in_parallel_context and is_tile_op(op):
            raise ValueError("tile-op inside Parallel not allowed")


def NestedLoopChecker():
    def pass_fn(func: PrimFunc, mod, ctx):
        _NestedLoopCheckVisitor().visit_stmt(func.body)
        return func                             # ← ④ 只读，直接返回原函数
    return prim_func_pass(pass_fn, opt_level=0)
```

### 概念卡：**Visitor 模式**

你可能听说过"访问者模式"。在编译器里它就是：

```
┌─────────────────────────────────────────────────────────┐
│ 每个 IR 节点类型都对应一个 visit_xxx_ 方法               │
│                                                         │
│  Stmt 节点:                                             │
│    visit_for_        visit_if_then_else_                │
│    visit_allocate_   visit_attr_stmt_                   │
│    visit_evaluate_   visit_buffer_store_                │
│    visit_block_      ...                                │
│                                                         │
│  Expr 节点:                                             │
│    visit_call_       visit_buffer_load_                 │
│    visit_add_        visit_var_                         │
│    ...                                                  │
│                                                         │
│ 你只重载"你关心的那几个"，其他节点默认递归下降。          │
└─────────────────────────────────────────────────────────┘
```

**为什么方法名后面都带下划线 `visit_for_` 而不是 `visit_for`？**
因为 TVM 的 C++ 侧宏 `TVM_FFI_DEF_TVM_FFI_METHOD("visit_for_", ...)` 用带下划线的名字
来避开 Python 关键字冲突（`for` 是关键字），慢慢就成了约定。

### 概念卡：**PyStmtExprVisitor vs PyStmtVisitor vs PyExprVisitor**

TVM 提供三个 base class，按你想访问什么选一个：

| Base                | 能访问              |
| ------------------- | ---------------- |
| `PyStmtVisitor`     | 只 Stmt           |
| `PyExprVisitor`     | 只 Expr           |
| `PyStmtExprVisitor` | Stmt + Expr（最常用） |

> 💡 **上一轮我提到过的 `StmtVisitor`** 是 C++ 侧 base class 的名字，
> Python 侧对应就是 `PyStmtVisitor`（`Py` 前缀 = "从 Python 继承的版本"）。

### 试着自己跑一次

保存下面这段为 `try_visitor.py` 就能跑：

```python
import tilelang, tilelang.language as T
from tilelang.analysis.nested_loop_checker import NestedLoopChecker

# 一个合法的 matmul
@tilelang.jit
def good(A, B):
    M = T.const("M")
    A: T.Tensor((M, M), T.float16)
    B: T.Tensor((M, M), T.float16)
    C = T.empty((M, M), T.float16)
    with T.Kernel(1, threads=32) as bx:
        for i, j in T.Parallel(M, M):
            C[i, j] = A[i, j] + B[i, j]
    return C

pf = good.get_tir(M=128)
NestedLoopChecker()(pf.with_attr("global_symbol", "good"))   # ✅ 不 raise
print("passed")
```

> 📌 **`pf.with_attr("global_symbol", "good")` 是啥？**
> 
> - **`global_symbol`** 是 `PrimFunc` 的一个属性（元信息），值是这个函数在 IRModule 里的**全局名字**。TVM 的 pass pipeline **要通过这个名字定位函数**（比如 `mod[global_symbol]` 才能取到），没打这个属性 pass 走到某些环节会 raise。
> 
> - **`.with_attr(key, value)` 到底做了什么？**
>   
>   - **签名**：`PrimFunc.with_attr(key: str, value) -> PrimFunc`
>   - **行为**：**不修改原 `pf`**，而是"拷贝一份、把 attrs 字典里加上 `key=value` 这一项、返回新 `PrimFunc`"。原来的 `pf` 对象一个字节都不会变（第 2 章说过 `PrimFunc` 是 **immutable** ObjectRef，你没法"就地"改它）。
>   - **等价的心智模型**（**只是心智模型，别当真去这样写**）：
>     
>     ```python
>     def with_attr(self, key, value):
>         new_attrs = {**self.attrs, key: value}   # 拷贝 attrs、加/覆盖一项
>         return PrimFunc(params=self.params, body=self.body,
>                         ret_type=self.ret_type, buffer_map=self.buffer_map,
>                         attrs=new_attrs)         # 返回新对象
>     ```
>   - **正确用法一定是"接住返回值"**：`pf = pf.with_attr(...)`。写成 `pf.with_attr(...)` 但没赋值，等于白改（新对象被立刻丢弃）。
>   - **同族方法**：`.with_body(new_body)`（换 body）、`.with_attrs(dict)`（批量塞 attrs）、`.without_attr(key)`（删一项）——4.4 表格里出现的 `.with_body` 就是同一族。**所有以 `.with_` 开头的方法都遵循这个"造副本"约定**，是 TVM/TileLang 里改 IR 节点的标准写法。
> 
> - **为什么要在这里手动打？** `get_tir()` 返回的 `PrimFunc` 有时候没带 `global_symbol`（视上层 pass 而定）；直接喂给 pass 会挂。所以本书大多数示例（第 5/10/11 章都有）都有类似防御性写法：
>   
>   ```python
>   if "global_symbol" not in pf.attrs:
>       pf = pf.with_attr("global_symbol", "main")
>   ```
>   
>   这里为了紧凑，直接一行 `pf.with_attr(...)` 强行覆盖成 `"good"`。
> 
> - 

## 4.4 改写型 pass：用 Mutator 造新节点

**用途**：真正修改 IR。方法：继承 `PyStmtExprMutator`，重载 `visit_xxx_` **返回一个新节点**。

[`tilelang/metal/transform/mark_host_metal_context.py`](../../tilelang/metal/transform/mark_host_metal_context.py) 是绝佳短例：

```python
from tvm import tirx as tir
from tvm.ir import Op
from tvm.tirx import AttrStmt, Evaluate, PyStmtExprMutator, functor
from tvm.tirx.transform import prim_func_pass

_tvm_call_packed_lowered = Op.get("tirx.tvm_call_packed_lowered")


@functor.mutator                              # ← ① mutator 装饰器
class _MarkHostMetalContextMutator(PyStmtExprMutator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_in_compute_scope = False

    def visit_attr_stmt_(self, stmt):
        switch = stmt.attr_key == "compute_scope"
        old_value = False
        if switch:
            old_value, self.is_in_compute_scope = self.is_in_compute_scope, True
        s = self.visit_stmt(stmt.body)         # ← ② 递归改写 body
        if switch:
            self.is_in_compute_scope = old_value
        return s

    def visit_evaluate_(self, op: Evaluate):
        if self.is_in_compute_scope and isinstance(op.value, tir.Call) \
                and op.value.op.same_as(_tvm_call_packed_lowered):
            return AttrStmt(0, "metal_context", "", op)   # ← ③ 造一个新节点返回
        return op                                          # ← ③ 或者返回原节点


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        mutator = _MarkHostMetalContextMutator()
        new_body = mutator.visit_stmt(func.body)          # ← ④ 拿到新 body
        return func.with_body(new_body)                    # ← ⑤ func 换 body
    return prim_func_pass(pass_fn, opt_level=0)
```

### 关键约定

| 约定                                     | 含义                                                          |
| -------------------------------------- | ----------------------------------------------------------- |
| **每个 `visit_xxx_` 都必须 return 一个节点**    | 视你要不要改而定：`return op`（不改）或 `return NewNode(...)`（改）          |
| **父节点自动更新**                            | 你 return 了新的 `stmt.body`，父 `AttrStmt` 会自动"clone 后把 body 换掉" |
| **函数级别更新用 `func.with_body(new_body)`** | `PrimFunc` 也是 immutable，用 `.with_body` / `.with_attr` 造一个副本 |
| **配合 `functor.mutator` 装饰器**           | 它会把 C++ 侧的调度表和 Python 类连起来；忘写会 raise `Unregistered functor` |

## 4.5 pass 是怎么被"串起来"的：`PassPipeline`

一个真实 kernel 要跑**几十个 pass**。TileLang 用一个非常轻的容器把它们串起来
（[`tilelang/backend/pass_pipeline/pipeline.py`](../../tilelang/backend/pass_pipeline/pipeline.py)）：

```python
class PassPipeline:
    def __init__(self, name: str, lower: LowerFunc):
        self.name = name
        self._lower = lower                 # 一个 (mod, target) -> mod 的函数

    def lower(self, mod, target):
        return self._lower(mod, target)


_PIPELINES: dict[str, PassPipeline] = {}


def register_pipeline(pipeline): _PIPELINES[pipeline.name] = pipeline
def resolve_pipeline(target):    return _PIPELINES[target.kind.name]
```

**就 30 行**——它不像 TVM 官方那样引入 `tvm.transform.Sequential`，
而是直接把整个 pass 顺序表**装在一个 Python 函数里**（每一步 `mod = SomePass()(mod)`）。

每个后端在自己的目录下写一个 `pipeline.py`：

```
tilelang/
├── cuda/pipeline.py        ← register_pipeline(PassPipeline("cuda",  CUDAPassPipelineBody))
├── rocm/pipeline.py        ← register_pipeline(PassPipeline("hip",   ROCMPassPipelineBody))
├── metal/pipeline.py       ← register_pipeline(PassPipeline("metal", MetalPassPipelineBody))
├── webgpu/pipeline.py      ← register_pipeline(PassPipeline("webgpu", WebGPUPassPipelineBody))
└── cpu/pipeline.py         ← register_pipeline(PassPipeline("cpu",   CPUPassPipelineBody))
```

顶层 `tilelang.lower(mod, target="cuda")` 里的关键两行就是：

```python
# tilelang/engine/lower.py:290
pipeline = resolve_pipeline(target)      # 按 target 拿到 "cuda" 那份 pass 表
mod = pipeline.lower(mod, target)         # 一次性跑完那份 pass 表
```

### 概念卡：**为什么要有 pipeline 抽象？**

不同硬件的 pass 顺序**不一样**：

- CUDA 需要 `LowerHopperIntrin` 处理 wgmma、`RewriteWgmmaSync` 之类；ROCm 不需要
- Metal / WebGPU 需要 `MetalFragmentToSimdgroup` 等；CUDA 不需要
- CPU 完全不走 shared memory / warp specialization 那套

把它们各自封在一个 `PassPipeline` 里，`lower()` 就不用写一大坨 `if target.kind == "cuda": ...`。

第 5 章会**逐 pass** 展开 CUDA 那份表（[`tilelang/cuda/pipeline.py`](../../tilelang/cuda/pipeline.py) 的 `CUDAPassPipelineBody`）。

## 4.6 pass 之间怎么共享"配置开关"：`PassContext`

同一个 pass 有时候要根据用户传的参数改行为。例如"是否禁用 warp specialization"、
"是否打印中间 IR"、"是否检查 data race"……这些**跨 pass 的全局开关**放在
`PassContext.current().config` 里。

### 用户侧：怎么传配置

```python
kernel = matmul.compile(
    M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32,
    pass_configs={
        "tl.disable_warp_specialized": True,
        "tl.disable_tma_lower": True,
    },
)
```

这些 kwarg 最终会灌到 `PassContext.override_instructions({...})`，
在 `pipeline.lower(...)` 期间生效。

### Pass 里：怎么读配置

看 [`tilelang/cuda/pipeline.py:35`](../../tilelang/cuda/pipeline.py) 的真实用法：

```python
def CUDAPassPipelineBody(mod, target):
    pass_ctx = PassContext.current()
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    ...
    if not disable_warp_specialized:
        mod = ProducerConsumerWarpSpecialized()(mod)
```

或者一个更集中的 helper（[`tilelang/backend/pass_pipeline/pipeline_utils.py`](../../tilelang/backend/pass_pipeline/pipeline_utils.py)）：

```python
def is_warp_specialize_disabled(pass_ctx: PassContext | None = None) -> bool:
    pass_ctx = pass_ctx or PassContext.current()
    return bool(pass_ctx.config.get(PassConfigKey.TL_DISABLE_WARP_SPECIALIZED, False))
```

### 概念卡：**`PassConfigKey`** —— 别用魔法字符串

字符串名字（如 `"tl.disable_warp_specialized"`）容易拼错。TileLang 在
[`tilelang/transform/pass_config.py`](../../tilelang/transform/pass_config.py) 里把所有已知开关
封装成了 `PassConfigKey` 的枚举/常量。**新加开关时先在那里注册。**

## 4.7 亲手写一个 pass：数一数 kernel 里有多少 `T.copy`

来把 4.3 和 4.4 学的东西合起来，写一个**只读**分析 pass，
统计当前 PrimFunc 里 `T.copy` intrinsic 的调用次数。

`tmp/count_copies.py`：

```python
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm          # tilelang 自带的 tvm（下面 tvm.IRModule 要用）
from tvm import tirx
from tvm.tirx import Call, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass


@tirx.functor.visitor
class _CopyCounter(PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.n = 0

    def visit_call_(self, op: Call):
        # 3.1 讲过：T.copy 在解析后是 Call("tl.tileop.copy", ...) intrinsic
        if str(op.op) == "tl.tileop.copy":
            self.n += 1
        # 别忘了继续下降：Call 里可能嵌套 Call
        super().visit_call_(op)


def CountCopies():
    counter = _CopyCounter()

    def pass_fn(func, mod, ctx):
        counter.visit_stmt(func.body)
        print(f"[CountCopies] {func.attrs['global_symbol']}: {counter.n} T.copy calls")
        return func

    return prim_func_pass(pass_fn, opt_level=0), counter


# --- 用它 ---
@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_K, block_N), T.float16)
        C_local  = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by*block_M, ko*block_K], A_shared)     # +1
            T.copy(B[ko*block_K, bx*block_N], B_shared)     # +1
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by*block_M, bx*block_N])          # +1
    return C

pf = matmul.get_tir(M=1024, N=1024, K=1024,
                    block_M=128, block_N=128, block_K=32)
mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})

pass_obj, counter = CountCopies()
pass_obj(mod)
# 期望输出： [CountCopies] main: 3 T.copy calls
```

> **`mod = tvm.IRModule({pf.attrs["global_symbol"]: pf})` 这句在干嘛？** 第 2 章说过 pass 的输入/输出单位是 `IRModule`（一个 `{函数名: PrimFunc}` 的容器）。`get_tir(...)` 返回的是单个 `PrimFunc`，所以这里用一个字典把它包成 `IRModule`——**键**是这个函数的全局名字（从 `pf.attrs["global_symbol"]` 取，通常是 `"main"`），**值**就是 `pf`。这个"字典构造 IRModule"的写法后面第 5、7、10 章会反复用到，记住即可。

**关键点**：

- 我们的 pass 只在**解析后（阶段一）**运行才能看到 `Call("tl.tileop.copy", ...)`。
  阶段二之后这些 Call 已经被 `LowerTileOp` 展开成 `cp.async` / TMA 了，就数不到了。
  → **pass 什么时候插入 pipeline，决定它能看到什么形态的 IR**。这也是第 5 章的主题。

## 4.8 陷阱清单：新手最容易踩的 5 个坑

1. **忘了 `super().visit_xxx_(op)` 递归下降**
   → visitor 只访问了根节点，深层子树被跳过。
2. **在 Mutator 里 return 了 `None`**
   → TVM 会崩掉说"got NoneType"。必须 `return op` 或新节点。
3. **在 `visit_xxx_` 里改 `self.xxx` 后忘记还原**
   → 递归回来时状态就错了。看 4.4 里 `old_value` 备份还原的模式。
4. **用 `==` 比较 `Op`**
   → 应该用 `op.op.same_as(target_op)`。`==` 走的是结构相等，会退化到很慢的 SEqualReduce。
5. **给已经 lower 完的 IR 加只读 pass 却看不到 tile-op**
   → 因为 tile-op 已经被 `LowerTileOp` pass 展开掉了。把你的 pass 插到更前面。

> ⚠️ **常见误解**
> 
> - **"我 override 了 `visit_xxx_`，别的节点它就不管了"** —— 恰恰相反。Visitor / Mutator 默认会**递归下降遍历整棵树**；你 override 某个节点类型只是"在经过这类节点时插一脚"。**危险的反面**是：如果你 override 了某个节点却**忘了继续递归子节点**（Mutator 里忘了对子节点调 `visit_*`、或没返回递归后的结果），那么这棵子树里你真正关心的节点就再也走不到了——第 6 章讲的一类真实 bug 就是"只处理了 `BlockRealize` 却漏了 `For`，导致嵌套在 For 下的目标节点被整个跳过"。**规则：override 时要么显式递归子节点，要么调用基类实现兜底。**
> - **"Visitor 也能顺手改一下 IR"** —— 不能。`PyStmtExprVisitor` 是**只读**的，没有返回值语义；要改写必须用 `PyStmtExprMutator`（每个 `visit_*` 返回新节点）。想"检查"用 Visitor，想"改写"用 Mutator，别混用。
> - **"pass 顺序无所谓，反正都会跑"** —— 大错。pass 是**流水线**，前一个 pass 的输出是后一个的输入。你的只读 pass 想看的节点（比如 tile-op）可能已经被前面的 pass 展开没了；反过来，你依赖的某个 attr 也可能还没被注入。**插 pass 前先想清楚它要跑在 pipeline 的哪一段**（第 5 章会给出完整顺序表）。

## 4.9 本章要带走的三件事

1. **pass = `IRModule -> IRModule` 的函数**；用 `@prim_func_pass(opt_level=0)` 装饰即可。
2. **两种主流写法**：`PyStmtExprVisitor`（只读，扫描/检查）
   和 `PyStmtExprMutator`（改写，构造新节点返回）。
3. **多个 pass 组织成 `PassPipeline`**（每个后端一份），
   `PassContext.current().config` 是跨 pass 的"全局配置总线"。

---

下一章 [第 5 章 · Lowering Pipeline 逐 pass 巡礼](./05_lowering_pipeline.md)：
打开 [`tilelang/cuda/pipeline.py`](../../tilelang/cuda/pipeline.py) 的 `CUDAPassPipelineBody` 大表格，
把 CUDA 后端从**阶段一后**到**阶段二结束**要跑的每一个 pass 都讲一遍，
配上"每个 pass 前后 IR 的样子"的对比图。
