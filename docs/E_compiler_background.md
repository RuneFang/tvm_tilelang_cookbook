# 附录 E · 编译器背景速成 —— 为小白铺垫 TVM/TileLang 假定你已会的那部分

> 目的：正文里我们频繁说 "AST"、"IR"、"pass"、"visitor"、"lowering"、"SSA"、"FFI"、"JIT"…… 但这些概念并不是"看名字就能懂"。这份附录用**尽量少的术语堆叠**、**尽量多的类比和图**，把它们一次讲清，让你回头读 4/5/6/8 章时不再"每个词都认识连起来看不懂"。
>
> 阅读方式：**不用一口气读完**。写 pass / 追 bug 时被某个词卡住，回来查对应一节；每节末尾都指向"回到本书哪一章能立刻用上"。

---

## 目录

| # | 主题 | 一句话概括 | 回本书哪里立刻用上 |
|---|---|---|---|
| E.1 | 编译器到底做什么？ | 把"人写的高层描述"翻译成"机器能执行的低层描述" | 第 1 章 6 阶段流水 |
| E.2 | AST：所有编译器的公共起点 | 源码是一棵树，不是一串字符 | 第 2 章 |
| E.3 | IR：为什么中间还要多几层 | 越翻越像机器、越翻越丢语义 | 第 2/5 章 |
| E.4 | Pass：IR → IR 的一小步 | 每次只改一件事，串起来就是编译 | 第 4/5 章 |
| E.5 | Visitor 模式：怎么"走"一棵 IR 树 | 数据结构和算法解耦 | 第 4/8 章 |
| E.6 | SSA：每个变量只赋值一次 | 让编译器分析变得容易 | 第 8 章 |
| E.7 | Lowering vs Legalizing vs Rewriting | 三个近义词的精确区别 | 第 5 章 |
| E.8 | Dataflow / Control-flow / Loop 分析 | pass 优化前先"看懂代码" | 第 6 章 |
| E.9 | Analyzer / Simplifier / 常量折叠 | 编译器算算术的方式 | 第 4/11 章 |
| E.10 | Intrinsic vs Builtin vs Op | 三个"内建"到底哪家的 | 第 6 章 |
| E.11 | Codegen：从 IR 到目标语言字符串 | 就是一个"深度优先打印器" | 第 8 章 |
| E.12 | JIT / AOT / 解释器 | 什么时候翻译、翻译几次 | 第 9 章 |
| E.13 | FFI：跨语言互调的胶水 | 让 Python 能调 C++ 也能被 C++ 调 | 第 8/9 章 |
| E.14 | Runtime、ABI、Calling Convention | 编译产物怎么被"启动" | 第 9 章 |
| E.15 | Symbolic / Dynamic shape | 编译期还不知道形状怎么办 | 第 12 章 |
| E.16 | GPU 视角下的编译栈 | CUDA C++ → PTX → SASS → cubin 是怎么回事 | 第 8 章 |
| E.17 | 常用词汇最终辨析 | 一次性把易混词摆一起 | 全书 |

---

## E.1 编译器到底做什么？

一句话：**把"人容易写的高层描述"翻译成"机器容易跑的低层描述"**。

它不是一次翻完，而是像流水线一样，翻一小段、简化一下、再翻一小段：

```
你写的代码 → [解析器] → AST → [多个 pass] → 低层 IR → [codegen] → 目标语言/机器码
   高层                                                             低层
```

关键洞察：**编译器几乎所有工作都发生在中间那段"多个 pass"**。前后两端（解析和 codegen）反而是最机械的部分。

> 类比：翻译一本英文小说到中文。你不会一次从英文原文直接口译中文成品——你会先做逐句直译（AST），然后修辞润色（若干 pass），最后按目标读者习惯排版（codegen）。

**在 TileLang 里对应**：

- 解析器 = `@T.prim_func` / `@tilelang.jit` 装饰器 + Python AST 转换
- 多个 pass = 第 5 章讲的 ~50 个 pass
- codegen = 第 8 章讲的 `CodeGenTileLangCUDA`

→ 回第 1 章 §1.3 看六阶段流水的整体大图。

---

## E.2 AST：所有编译器的公共起点

**AST = Abstract Syntax Tree = 抽象语法树**。

当你写下：

```python
c = a + b * 2
```

计算机不会把它当作"13 个字符"，而是解析成一棵树：

```
    (=)
   /   \
  c    (+)
      /   \
     a    (*)
         /   \
        b    2
```

- **树的每个节点都代表一种"结构"**：赋值 `=`、加法 `+`、乘法 `*`、变量、常量……
- **叶子是原子**（变量名、字面量），**内部节点是运算/结构**。
- **"抽象"的意思**是：忽略源码里没有语义的东西（空格、括号、分号）。

> 类比：语文课的"句法树"——主语、谓语、宾语的层级结构。程序也是"有语法的语言"，同样可以画句法树。

**在 TVM 里 AST 就是 TIR**。你在 `T.prim_func` 里写的每一句：

- `for i in range(N):` → `tir.For`
- `A[i] = B[i] + 1` → `tir.BufferStore(A, tir.Add(BufferLoad(B), IntImm(1)))`
- `if x > 0:` → `tir.IfThenElse`

书里说的"遍历 IR"/"改写 IR"，本质就是遍历/改写这棵树。

→ 回第 2 章 §2.2 亲手看 `add_one` 的 IR 树打印。

---

## E.3 IR：为什么中间还要多几层

前面说 "AST 就是 IR"，但更准确的说法是：**IR 是一组 AST 中的某个"抽象等级"**。编译器往往有**多种 IR**，从高到低层层降解。

以 TVM 为例：

```
Relax / Relay         ← 高层：整张神经网络计算图
   ↓ 若干 pass
TIR (Tensor IR)        ← 中层：loop + buffer + primitive op
   ↓ 若干 pass         ← 本书主战场
Lowered TIR            ← 低层：thread_binding + mbarrier + intrinsic 都露出来
   ↓ codegen
CUDA C++ / LLVM IR     ← 最低层：接近硬件
```

**为什么不一步到位？**

因为不同优化在不同抽象层做起来更容易：

- 在高层（Relay/Relax）：容易做"融合两个算子"这种整图优化
- 在中层（TIR）：容易做"循环并行化 / 分块"这种数学优化
- 在低层（lowered TIR）：容易做"寄存器分配 / 指令选择"这种硬件优化

**类比**：装修房子先画平面图（高层），再决定水电走线（中层），最后砌墙铺砖（低层）。你不会在画平面图时就选择砖块颜色。

→ 回第 2 章 §2.1 看 TIR 在整个 TVM stack 中的定位。

---

## E.4 Pass：IR → IR 的一小步

**Pass = IR 的一次原子变换**。

数学签名：

```
Pass : IR → IR
```

- 输入一个 IR 树，输出一个新的 IR 树
- **每个 pass 只做一件事**：InjectPipeline、Simplify、ThreadSync、Vectorize……
- 多个 pass **串起来**就是完整的 lowering pipeline

> 类比：Photoshop 里每个滤镜（模糊、锐化、去噪）都是一个 pass，把它们按顺序作用于同一张图，就是"完整修图流程"。

**Pass 的两种典型形态**：

| 类型 | 只读 | 可写 | TVM 基类 |
|---|---|---|---|
| Analysis pass | ✅ | ❌ | `StmtExprVisitor` |
| Transform pass | ✅ | ✅ | `StmtExprMutator` |

Analysis pass 不改 IR，只**计算某些信息**（比如 "有几个 T.copy"），可以给后续 pass 用。

**Pass 顺序为什么重要？**

先做 A 再做 B ≠ 先做 B 再做 A。比如：

- 先 `Simplify`（消掉冗余算术）再 `InjectPipeline`：pipeline 判断的表达式简单，好分析
- 反过来：pipeline 已经把循环拆成三段了，Simplify 就要处理三倍的代码

TileLang 的 pass 顺序被封在 `PassPipeline` 里，第 5 章会带你走一遍。

→ 回第 4 章 §4.7 亲手写一个 pass；第 5 章看真实 pipeline 长什么样。

---

## E.5 Visitor 模式：怎么"走"一棵 IR 树

假设你要数一段 TIR 里有几个 `T.copy` 调用。这段 IR 是一棵**任意深度的树**，你怎么"走遍每一个节点"？

**朴素做法**：`for` 循环 + `if isinstance(node, ...)` 手写递归 —— 一遍下来上百行 `elif`，维护崩溃。

**Visitor 模式的做法**：

1. 定义一个 **visitor 类**，为每种节点类型写一个 `visit_XXX(self, node)` 方法。
2. 基类负责递归下降（内置分派 —— 见到什么节点调什么 `visit_`）。
3. 你只需要**覆盖你关心的节点类型**，其他节点由基类默默走完。

```python
class CountCopies(PyStmtExprVisitor):
    def __init__(self):
        self.n = 0

    def visit_call_(self, op):
        if str(op.op) == "tl.tileop.copy":
            self.n += 1
        super().visit_call_(op)     # 关键：让基类继续下降
```

> 类比：让邮递员挨家挨户送信。邮递员不需要"预先知道"每家住的是谁——他只按门牌顺序走，遇到你订过的杂志才停下来投递。基类是"按门牌顺序走"，你写的 `visit_XXX` 是"投递给关心的门牌"。

**Mutator = Visitor + 返回值**：

- Visitor: `visit_XXX(node) → None`（只读）
- Mutator: `visit_XXX(node) → new_node`（可写，返回新节点替换旧的）

TileLang 的 codegen（第 8 章）本质上也是一个 Visitor —— 每 `visit_` 一个节点就往 `std::ostringstream` 里写一段 CUDA 源码。

→ 回第 4 章 §4.3 / §4.4 看真实 pass 的 visitor 骨架；第 8 章 §8.4 看 codegen 怎么"走完一棵 IR 打印出 CUDA"。

---

## E.6 SSA：每个变量只赋值一次

**SSA = Static Single Assignment = 静态单赋值**。

普通代码：

```python
x = 1
x = x + 1     # 又赋值了一次
x = x * 2
```

在 SSA 形式下，会被重写成：

```python
x_1 = 1
x_2 = x_1 + 1
x_3 = x_2 * 2
```

**每个"版本"的变量只赋值一次**，用途都指向"这个版本"。

**为什么麻烦一步？** 因为很多优化在 SSA 形式下变得**平凡**：

- **常量传播**：`x_1 = 1` 之后，只要看到 `x_1` 都可以替换成 `1`，永远不用担心 "会不会中间被改过"。
- **死代码消除**：如果 `x_2` 后面不再被引用，`x_2 = x_1 + 1` 整行都能删掉。
- **别名分析**：两个变量名字不同意味着"绝对不是同一份数据"。

**TIR 是不是 SSA？** 大体上是。TileLang 的多数 pass **假定**并 **维护** SSA 性质。当你手写 pass 时，如果 rewrite 后同一个 `Var` 被赋值两次，就打破了 SSA，后续 pass 可能出错。

→ 回第 8 章看 codegen 里 `let` 表达式怎么打印。

---

## E.7 Lowering vs Legalizing vs Rewriting —— 三个近义词的精确区别

正文里这三个词经常混着用，其实**语义不同**：

| 词 | 含义 | 例子 |
|---|---|---|
| **Rewrite（重写）** | 把 IR 的某段替换成另一段，抽象层级**不变** | `x*2 → x<<1`（都是 TIR，只是形式变了） |
| **Legalize（合法化）** | 用户能写但下一个 pass 不认识的形式改写成能认识的 | `A[..., -1] → A[..., N-1]`（同一抽象层级但更"规范") |
| **Lower（下降）** | 把高层抽象降到低层，**抽象等级下降** | `T.gemm(...) → for i, j, k: C += A*B` |

**画个图**：

```
   高层
    │
    │ Legalize（同层内部整形）
    │ ↕
    ▼
    ├── Lower（跨层下降）  ──► 低层
    │
   Rewrite（任意层都在发生的通用改写）
```

**在 TileLang 里**：
- `frontend_legalize.cc`：把用户写法规范化（Legalize）
- `LowerHopperIntrinsics.cc`：把 `T.gemm` 展成 WGMMA 指令（Lower）
- `Simplify` pass 里的常量折叠：不改层级（Rewrite）

→ 回第 5 章看 pass pipeline 每个阶段分别属于哪种。

---

## E.8 Dataflow / Control-flow / Loop 分析

Pass 在改代码之前，通常要先"看懂"代码。三种最基础的看法：

**Data-flow analysis（数据流分析）**：追踪"某个值从哪里来、被谁用"。

```python
a = load(A[i])       # a 从这里"产生"
b = a * 2            # a 在这里"被用"
c = a + b            # a 在这里"被用"（第二次）
```

图上：`a` 有一条出边到 `b`、一条出边到 `c`。这个"谁产生谁消费"的图叫 **use-def chain**。

**Control-flow analysis（控制流分析）**：追踪 "程序执行会走哪条路径"。

```python
if x > 0:            #    ┌─── x>0 分支
    y = 1            #    │
else:                #    │
    y = 2            #    └─── x<=0 分支
z = y + 1            #  两条路汇合
```

控制流是**图**（**CFG = Control Flow Graph**），每个基本块是节点，跳转是边。

**Loop analysis（循环分析）**：识别循环的"归纳变量"（每次 +1 的那个 `i`）、循环边界、循环体的副作用。

**为什么 pass 需要这些？** 举例：
- 想做 `Vectorize`，先要用循环分析找出**归纳变量**和**循环体的 memory access pattern**。
- 想做 `InjectPipeline`，先要用数据流分析确认 "producer 和 consumer 是不是访问了同一份 buffer"。
- 想做 `IfStmtBinding`，先要控制流分析知道 "这个 if 会不会被 thread axis 摊平"。

→ 回第 6 章看 `MultiVersionBufferRewriter` 怎么用 use-def 决定"哪个 buffer 需要开多版本"。

---

## E.9 Analyzer / Simplifier / 常量折叠

编译器需要**自己会算算术**：

- 看到 `4*3+1` 立即知道结果是 `13`（**常量折叠**）
- 看到 `x+0` 立即化简成 `x`（**代数化简**）
- 看到 `if 5 > 3:` 立即知道条件恒真，删掉 else 分支（**分支消除**）
- 看到 `for i in range(1024): if i < 512:` 立即知道 "前一半 else 分支永远不走"（**范围推断**）

TVM 把这些能力封装在 `arith::Analyzer` 里。你写 pass 时手上常年拿着一个 analyzer：

```cpp
arith::Analyzer analyzer;
analyzer.Bind(loop_var, Range::FromMinExtent(0, extent));
if (analyzer.CanProve(x < 512)) { ... }
PrimExpr simplified = analyzer.Simplify(expr);
```

**为什么这些能力值得单独讲？** 因为 pass 里 90% 的 bug 都源自 "分析器没能证明 XXX"。比如：

- 你写了个 pass 依赖 "`i < N` 能被简化成 true"，但 analyzer 里没绑定 `N` 的范围，就化简不成。
- 你的 IR 里表达式形式不标准（比如 `-1 + N` 而不是 `N - 1`），analyzer 就认不出来。

→ 回第 11 章 §11.5 看 Analyzer 怎么算 roofline 上限；第 4 章看 pass 里 analyzer 的正确用法。

---

## E.10 Intrinsic vs Builtin vs Op —— 三个"内建"到底哪家的

这仨词在 TVM/TileLang 里各有专用含义：

| 词 | 定义 | 例子 | 谁能写 |
|---|---|---|---|
| **Op** | TVM 里"具名的运算"对象（`tvm::Op`），是 `CallNode.op` 字段的类型 | `tir.add`, `tl.tileop.copy` | 编译器/用户都行 |
| **Builtin** | Op 的一种：由**编译器/框架**注册、有内部语义 | `tvm_thread_allreduce`, `tl.mvb_stage_index` | 只编译器 |
| **Intrinsic** | 通常指"直接对应硬件指令的 Op"（有时和 builtin 混用） | `ldmatrix`, `mma_sync`, `mbarrier_arrive` | 只编译器 |

**记忆口诀**：
- Op = 名字空间里的运算（大概念）
- Builtin = 编译器内部专用 Op（不给用户直接写）
- Intrinsic = 硬件指令级 Op（builtin 的一个子集）

**为什么区分重要？** 第 6 章 6.7 讲的核心教训：

- 用户可以写任意合法的 `FloorMod(k, 4)` 表达式
- 编译器 MultiVersionBufferRewriter 也会**生成** `FloorMod(k, 4)` 形式的表达式作为 stage 索引
- 后续 pass 无法用"表达式长什么样"来区分二者
- **解法**：让编译器生成方**用一个 builtin/intrinsic 把它包起来**（例如 `tl.mvb_stage_index(FloorMod(k, 4))`），这样消费方可以用 op identity 精确识别

这就是 "provenance 而非 syntax" 的核心机制。

→ 回第 6 章 §6.7 看 `TIR_DEFINE_TL_BUILTIN` 宏怎么注册一个 builtin。

---

## E.11 Codegen：从 IR 到目标语言字符串

Codegen（**Code Generation**）就是**把 IR 树打印成目标语言的源码字符串**。

**它到底有多简单？** 本质是一个 visitor + `std::ostringstream`：

```cpp
class CodeGenCUDA : public StmtExprVisitor {
  std::ostringstream os;

  void VisitStmt_(const ForNode* op) {
    os << "for (int " << op->loop_var->name_hint
       << " = 0; " << op->loop_var->name_hint
       << " < " << PrintExpr(op->extent) << "; ++"
       << op->loop_var->name_hint << ") {\n";
    VisitStmt(op->body);
    os << "}\n";
  }
  // ... 每种节点一个 VisitStmt_ / VisitExpr_
};
```

看到 `ForNode` 打印 `for (...) {`、看到 `BufferStore` 打印 `A[i] = ...;`、看到 `IfThenElseNode` 打印 `if (...) { } else { }`。

**难点不在遍历，而在细节**：

- **优先级和括号**：`a + b * c` 打印成 `(a) + ((b) * (c))` 太丑，聪明地判断"外层运算符优先级更高就需要括号"。
- **变量重命名**：TIR 里两个作用域可以有同名 `Var`，打印到 C++ 就得改名避免冲突。
- **类型转换**：TIR 有自己的 dtype，映射到 C++ 的 `float / __half / int8_t` 得对齐。
- **硬件专用指令**：某些 op（如 `mma_sync`）不是 C++ 内建，得打印成 `asm volatile("mma.sync..." ...)` 内联汇编。

→ 回第 8 章看 `CodeGenTileLangCUDA` 的具体实现。

---

## E.12 JIT / AOT / 解释器 —— 什么时候翻译？

同样一段"高层代码"，可以有三种执行方式：

| 方式 | 何时翻译 | 何时执行 | 例子 |
|---|---|---|---|
| **解释器** | 边翻边执行 | 每次执行都重翻 | Python 官方 CPython |
| **JIT**（Just-In-Time）| 第一次执行时翻，翻完缓存 | 之后直接跑缓存 | Java HotSpot / JAX / TileLang |
| **AOT**（Ahead-Of-Time）| 编译期一次性翻完 | 运行时不再翻 | C++/Rust；TensorRT engine |

**TileLang 是 JIT**：`@tilelang.jit` 装饰的函数**第一次调用时**才走 lowering + nvcc，编译结果缓存到磁盘和进程内 dict。

**为什么选 JIT 而不是 AOT？**

- 你需要根据**运行时形状**决定 tile size（AOT 编译时不知道 M/N/K）
- 你想编译一次跑多次（比 AOT 快启动、比解释器快执行）
- 缓存粒度可以精确到 (formal params, target, pass_configs)

**为什么不选纯解释器？** GPU kernel 不可能解释执行——最终必须编成 cubin 才能 launch。

→ 回第 9 章看 TileLang JIT 的完整调用链和缓存机制。

---

## E.13 FFI：跨语言互调的胶水

**FFI = Foreign Function Interface = 外部函数接口**。

你在 Python 里写 `kernel(a, b)`，最终要**调用 nvcc 编出来的 C++ 函数**——这跨越了 Python 和 C++ 两个世界。中间的"桥"就是 FFI。

**TVM 的 FFI 设计**（`tvm.ffi`）：

- **`ffi::Function`**（旧名 `PackedFunc`）：C++ 侧的**跨语言可调用对象**，任何签名 `(TVMFFIAny*, int, TVMFFIAny*) → int` 的 C++ 函数都可以包装成它。
- **全局注册表**：C++ 侧 `TVM_FFI_REGISTER_GLOBAL("mypkg.myfunc").set_body(...)`，Python 侧 `tvm.ffi.get_global_func("mypkg.myfunc")` 就能拿到并调用。
- **参数用 `TVMFFIAny`**：一个 tagged union，能装 int/float/string/Tensor/PackedFunc 等，从而支持任意签名。

**在 TileLang 里对应**：
- Python 侧 `@tilelang.jit` 拿到 kernel 后，通过 FFI 调 C++ 的 `target.build.tilelang_cuda`
- C++ 侧 codegen 完毕，把生成的 `runtime::Module` 通过 FFI 传回 Python
- Runtime 时 `kernel(A, B)` 最终会走一个 PackedFunc 调 CUDA driver 的 `cuLaunchKernel`

**类比**：翻译电话（同声传译）—— Python 和 C++ 各说各的话，FFI 是中间那位翻译。

→ 回第 8 章 §8.2 看 `target.build.tilelang_cuda` 的 FFI 注册；第 9 章看 Adapter 层如何把 `torch.Tensor` 转成 `DLTensor` 喂给 PackedFunc。

---

## E.14 Runtime、ABI、Calling Convention —— 编译产物怎么被"启动"

编译器产出的 cubin **不能自己蹦起来**——它只是一段字节，需要有人：

1. 加载到 GPU（`cuModuleLoadData`）
2. 找到函数入口（`cuModuleGetFunction`）
3. 把 host 端参数打包成 GPU 能读的格式
4. 分配 grid/block 尺寸、shared memory 大小
5. 调 `cuLaunchKernel`

这一整套动作叫 **runtime**（运行时）。TVM 的 runtime 就是那份很薄的 C++ 代码（`tvm/runtime/`），负责跨设备（CPU/CUDA/Metal/…）统一暴露 "load、find、launch" 三件事。

**ABI = Application Binary Interface = 应用二进制接口**：编译好的代码之间**互相调用的约定**。包括：

- 参数怎么放（哪些用寄存器、哪些用栈）
- 返回值放哪
- 名字怎么 mangling（比如 C++ 的 `_Z3fooii`）
- 栈是谁清理（caller 还是 callee）

**Calling Convention**：ABI 里"参数怎么放"这一小块。CUDA 里常见的是 `__global__` 函数的参数直接进 `constant memory`。

**为什么小白也要知道？** 因为你会看到 kernel signature 里那些奇奇怪怪的东西：`__grid_constant__`、`__launch_bounds__(128, 1)`、`extern "C"`—— 都是在满足 ABI/calling convention 约定。

→ 回第 8 章看 host stub 怎么组装 launch 参数；第 9 章看 `ffi::Function` 底层的 ABI 约定。

---

## E.15 Symbolic / Dynamic shape —— 编译期还不知道形状怎么办

**Static shape**：编译期就是常数，如 `M=1024, N=1024, K=1024`。

**Dynamic shape**：编译期不知道，运行时才定，如 `M=?, N=?, K=?`。

编译器怎么处理？答案是引入 **符号变量**（symbolic variable）：

- 编译期 `M` 是一个符号 `Var("m", "int32")`
- IR 里所有涉及 M 的表达式都用这个符号（`M * 4`, `T.ceildiv(M, block_M)`……）
- Analyzer 会记录 `M >= 0` 之类的**范围约束**，尽量简化
- Codegen 阶段把 `M` 打印成一个 host 端传进来的参数

**代价**：
- 有些优化在符号情况下做不了（不知道 tile 能不能整除 M）
- 生成的 kernel 里会多一些"处理边界"的 if

**TileLang 里的写法**：`M = T.dynamic("m")`（旧名 `T.symbolic`）。

→ 回第 12 章 §12.4 看动态 shape GEMM 的完整例子。

---

## E.16 GPU 视角下的编译栈 —— CUDA C++ → PTX → SASS → cubin

CUDA 端的"编译"其实是**多阶段**的，每一阶都是"IR 到更低层 IR"：

```
你的 CUDA C++ 源码 (kernel.cu)
   ↓ nvcc 前端（+ clang 前端）
LLVM IR (bitcode)
   ↓ nvptx backend
PTX (Parallel Thread Execution) —— NVIDIA 的伪汇编，与具体 SM 无关
   ↓ ptxas（也可以运行时用 driver 编）
SASS (Streaming Assembler) —— 针对具体 SM 版本的真汇编
   ↓ 打包
cubin (CUDA Binary) —— 单 SM 版本
   ↓ 可多份拼装
fatbin —— 多 SM 版本合一
```

**分层的好处**：

- PTX 是"跨 SM 版本"的中间层：你的 PTX 编好一次，可以在新硬件出来后由 driver 现场翻译成新的 SASS。
- SASS 是"真汇编"：调优到极致的库（cuBLAS/CUTLASS）经常直接看 SASS 找机会。

**在 TileLang 里对应**：
- `.get_kernel_source()`：返回**CUDA C++ 源码字符串**
- `.export_ptx()`：返回 PTX
- `.export_sass()`：返回 SASS（需要本机装了 `cuobjdump`）
- `.export_sources()`：给出编译中间产物

→ 回第 8 章 §8.5 / §8.6 看 nvcc 和 nvrtc 两条路径；第 11 章 §11.7 看怎么在调 bug 时下钻到 PTX/SASS。

---

## E.17 常用词汇最终辨析（易混词一次讲清）

**Compile 编译 vs Build 构建 vs Assemble 汇编**
- Compile：源码 → 目标代码（可能中间有 IR 层）
- Assemble：汇编源码 → 机器码（compile 的最后一小步）
- Build：包含 compile + link + package 的完整工程流程

**Interpret 解释 vs Execute 执行**
- Execute：跑起来（谁跑都算）
- Interpret：不预编译，边翻译边跑

**Emit 发射 vs Lower 下降 vs Generate 生成**
- Emit：codegen 时"把这段 IR 输出成字符串"的动作
- Lower：把 IR 从高层降到低层
- Generate：宽泛，通常指 codegen 的整体动作

**Trace 追踪 vs Log 日志 vs Dump 转储**
- Trace：程序运行时"每步做什么"的动态记录
- Log：应用主动打印的信息
- Dump：把某个内部状态**整体**输出（如 dump IR 到 .py 文件）

**Op 与 Operator（算子）**
- 在 TVM 语境里通常等价。C++ 中 `tvm::Op` 是那个类；日常口语里"op"可以指任何"一次运算"（`add`、`gemm`……）

**Kernel（内核）**
- 在 CUDA 语境里指"GPU 上一次并行执行的函数（`__global__` 修饰的那个）"
- 在编译器语境里可能指"某个 primitive 计算"（易混）
- **在本书里，"kernel" 一律指前者**

**Fragment（片段）**
- 通用 CS：一小段代码
- TileLang 专用：`T.alloc_fragment` 表示"寄存器 tile + 分线程"，见附录 C 和第 7 章

---

## 想再深挖一层？

如果你被这份附录勾起了对编译原理的兴趣，可按顺序找如下资源（都是免费或经典必读）：

1. **《Compilers: Principles, Techniques, and Tools》**（俗称"龙书"）—— 编译原理经典，重点看第 1、6、8、9 章。
2. **《SSA-based Compiler Design》**（Rastello et al., 免费 pdf）—— SSA 深挖。
3. **《LLVM Kaleidoscope tutorial》**（LLVM 官方教程）—— 手写一个玩具编译器，理解 pass / IR 从 0 到 1。
4. **《TVM Deep Dive》**（Chen 2018, ASPLOS）—— TVM 的原始论文。
5. **《FlexFlow》/《Roller》/《Ansor》**：三篇张量编译器方向经典论文，对应 TVM/TileLang 的思想血脉。
6. **《CUDA C++ Programming Guide》** 附录 C（PTX ISA）—— 从 CUDA 视角理解编译栈。

---

## 结语：这些概念真的都用到了吗？

**是的**。回顾一下：
- 第 1 章的"6 阶段流水" = E.1
- 第 2 章的 PrimFunc/Buffer/Stmt = E.2 + E.3
- 第 4 章的 pass_context / mutator = E.4 + E.5 + E.9
- 第 5 章的 pipeline 分四段 = E.7
- 第 6 章的 provenance vs syntax = E.10
- 第 8 章的 codegen + FFI = E.11 + E.13
- 第 9 章的 JIT 缓存 = E.12 + E.14
- 第 12 章的 T.dynamic = E.15
- 全书的 CUDA/PTX/SASS 对照 = E.16

**这就是"背景知识变成读代码的能力"的完整路径**：先建立起这些概念的**心智模型**，再看正文里 `MultiVersionBufferRewriter::VisitStmt_` 之类的 C++ 代码就不再"每行字都懂但连起来不懂"。

→ 你现在已经建立了完整的地图。回到 [README](./README.md) 挑一章继续往下读吧。
