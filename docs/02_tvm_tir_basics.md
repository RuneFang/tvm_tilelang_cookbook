# 第 2 章 · TVM / TIR 基础概念

> **TL;DR**：本章把第 1 章"阶段一"输出的那个 `PrimFunc` 掰开揉碎讲清楚。
> 读完你要能盯着一段 IR 打印出来的文本说出："这是个 `For` 里包了一个 `BufferStore`，
> 表达式那边是个 `Add(BufferLoad, IntImm)`"——**能读 IR 是理解 pass 的前提**。
> 
> **你会读到的真实源码**：
> 
> - `3rdparty/tvm/include/tvm/tir/stmt.h`
> - `3rdparty/tvm/include/tvm/tir/expr.h`
> - `3rdparty/tvm/include/tvm/tir/buffer.h`
> - `3rdparty/tvm/include/tvm/tir/function.h`
> - `3rdparty/tvm/include/tvm/tir/stmt_functor.h`（这是所有 pass 的父类）
> 
> **前置**：读完 [第 1 章](./01_hello_tilelang.md)。

---

## 2.1 为什么我们需要"中间表示（IR）"

先回答一个哲学问题：为什么不直接把 Python 变成 CUDA？

答案是：**做不到，也不该做**。

- **做不到**：Python 里的 `for ko in T.Pipelined(...)` 里究竟有几个 buffer 要开双缓冲、
  要插几个 mbarrier、要跟哪个 producer/consumer 对齐——这些问题需要**看全整个函数的结构**才知道。
  一行行边解析边生成 CUDA，等价于逐行翻译一门语言到另一门，无法做全局优化。

- **不该做**：Nvidia GPU、AMD GPU、CPU、Metal 各有各的指令集，如果每种都单独实现一遍
  "Python → 目标代码"，代码量会炸掉。我们想要的是：**Python → 一份中间表示 → 各种目标**。

这份中间表示就是 **IR (Intermediate Representation)**。TVM 家的 IR 有两级：

- **Relax** / **Relay**：图级 IR，描述"张量 A × B 后接 relu"这种 op 级别的东西。
  TileLang 基本不用它——TileLang 直接从 Python 生成下一级。
- **TIR (Tensor IR)**：**语句 / 表达式级** IR，形态非常接近 C。
  一个 `for` 循环就是一个 `For` 节点，一个 `a[i] = b[i] + 1` 就是一个 `BufferStore` 节点。
  **TileLang 的一切都在这一级上做**。

## 2.2 IR 长什么样：先看一眼

打开 Python 起个 REPL：

```python
from tilelang import tvm
from tvm import tir
from tvm.script import tir as T

@T.prim_func
def add_one(A: T.Buffer((16,), "float32"),
            B: T.Buffer((16,), "float32")):
    for i in range(16):
        B[i] = A[i] + 1.0

print(add_one.script())
```

输出（就是 TIR 的文本形态）：

```
@T.prim_func
def add_one(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
    for i in range(16):
        B[i] = A[i] + T.float32(1.0)
```

看起来跟 Python 差不多，但**这不是 Python**——它是**一棵 IR 树**的文本渲染。
树长这样：

```
PrimFunc(A, B)                       ← function.h: PrimFunc
└── For(loop_var=i, extent=16)       ← stmt.h  : For
    └── BufferStore(B, [i], value=?) ← stmt.h  : BufferStore
        └── Add                       ← expr.h  : Add
            ├── BufferLoad(A, [i])   ← expr.h  : BufferLoad
            └── FloatImm(1.0)        ← expr.h  : FloatImm
```

**记住这个心智模型**：TIR 是一棵树。所有 pass 都在做同一件事：**递归遍历这棵树，
选择性地把某些节点换成新节点**。

## 2.3 IR 树的三大骨架：Stmt、Expr、PrimFunc

TIR 里所有东西都能归到这三类中的一个。

### 2.3.1 `Expr`（表达式）—— 有值

**定义位置**：`3rdparty/tvm/include/tvm/tir/expr.h`

**特征**：一个 `Expr` 求值后**必然有一个值**（有 `dtype`）。就像 C 里的表达式，能出现在 `=` 右侧。

常见的 `Expr` 节点：

| 节点类                                                            | 对应 C 语法          | 例子                 |
| -------------------------------------------------------------- | ---------------- | ------------------ |
| `IntImm`                                                       | 整数字面量            | `42`               |
| `FloatImm`                                                     | 浮点字面量            | `1.0f`             |
| `Var`                                                          | 变量               | `i`                |
| `BufferLoad`                                                   | 数组读              | `A[i]`             |
| `Add` / `Sub` / `Mul` / `Div` / `FloorDiv` / `FloorMod`        | 算术               | `a + b`            |
| `EQ` / `NE` / `LT` / `LE` / `GT` / `GE` / `And` / `Or` / `Not` | 逻辑               | `i < n`            |
| `Cast`                                                         | 类型转换             | `(float)i`         |
| `Select`                                                       | 三元               | `a ? b : c`        |
| `Call`                                                         | 函数调用 / intrinsic | `tl.gemm(A, B, C)` |

**关键点：所有 `Expr` 都有 `dtype`**。这是 TVM 严格约束的东西：
`Add(a, b)` 要求 `a.dtype == b.dtype`，如果一个 int32 一个 int64，构造时的 `ICHECK` 就会失败。
写 pass 手工构造表达式（尤其涉及索引运算）时，混用 int32 / int64 是常见的报错来源。

### 2.3.2 `Stmt`（语句）—— 有副作用，没值

**定义位置**：`3rdparty/tvm/include/tvm/tir/stmt.h`

**特征**：一个 `Stmt` **执行会产生副作用**（写内存、控制流、分配缓冲），但它**没有值**。
就像 C 里的语句。

常见的 `Stmt` 节点：

| 节点类                      | 对应 C 语法                       | 用途                                          |
| ------------------------ | ----------------------------- | ------------------------------------------- |
| `SeqStmt`                | `{ ...; ...; }`               | 顺序执行一串 stmt                                 |
| `For`                    | `for (int i = 0; i < n; ++i)` | 循环                                          |
| `IfThenElse`             | `if (c) { A } else { B }`     | 条件                                          |
| `BufferStore`            | `A[i] = v`                    | 数组写                                         |
| `Allocate`               | `float A[16]`                 | 分配缓冲区                                       |
| `AttrStmt`               | 编译器元数据注解                      | 挂 target、thread_binding、software_pipeline 等 |
| `Evaluate`               | 只是求个表达式                       | 调用 intrinsic（例如 mbarrier arrive）            |
| `LetStmt`                | `int x = expr; ...`           | 命名一个中间值                                     |
| `BlockRealize` / `Block` | TVM 特有的"结构化块"                 | 分离 iteration domain 和 body                  |

**`For` 有个 `kind` 字段**，决定循环怎么执行：

- `ForKind::kSerial` —— 普通串行循环
- `ForKind::kParallel` —— 并行循环（TileLang 里 `T.Parallel`）
- `ForKind::kVectorized` —— 向量化
- `ForKind::kUnrolled` —— 展开
- `ForKind::kThreadBinding` —— 循环变量直接绑定到 `blockIdx.x` / `threadIdx.x` 之类

`T.Kernel(...)` 最终就是产生一堆 `ForKind::kThreadBinding` 的 For。

### 2.3.3 `PrimFunc`（函数）—— 根节点

**定义位置**：`3rdparty/tvm/include/tvm/tir/function.h`

一个 `PrimFunc` 是 IR 树的**根**，大致等于 C 里的一个函数：

```cpp
class PrimFuncNode {
  Array<Var> params;                              // 参数变量
  Map<Var, Buffer> buffer_map;                    // 参数变量 → Buffer 对象
  Stmt body;                                      // 函数体（一棵 Stmt 树）
  Type ret_type;                                  // 返回类型（TileLang 一般是 void）
  DictAttrs attrs;                                // 元信息，如 target / global_symbol
};
```

多个 `PrimFunc` 打包起来叫 **`IRModule`**（`3rdparty/tvm/include/tvm/ir/module.h`）。
你 `.compile()` 一次 TileLang 函数，得到的顶层容器就是一个 `IRModule`，里面通常有：

- 一个 device function（GPU kernel）
- 一个 host function（负责 launch）

## 2.4 `Buffer` 是什么

**定义位置**：`3rdparty/tvm/include/tvm/tir/buffer.h`

`Buffer` 不是数据本身，而是**"如何访问数据"的描述符**——它记录：

- `data`：底层存储的 `Var`（指针）
- `shape`：逻辑形状
- `strides`：步长
- `dtype`：元素类型
- `scope`：存储域（`"global"` / `"shared"` / `"local"` / `"local.fragment"` / `"shared.dyn"` 等）
- `elem_offset`：起始偏移

`BufferLoad(buf, [i, j])` 是"读取 `buf` 的 `[i, j]` 元素"，
经过后续 pass **拍平** 之后会变成对底层 `data` 指针的一次 `Load(data, i * strides[0] + j * strides[1])`。

> 💡 一个特别常见的困惑：为什么 IR 里既有 `Var A` 又有 `Buffer A`？
> —— `Var` 是抽象的指针"符号"，`Buffer` 是包着这个 `Var` 的访问语义。
> 早期 pass 阶段读写用 `BufferLoad/BufferStore`（形状信息还在），
> 后期 pass 用 `Load/Store`（形状已展开、只剩线性偏移）。
> `FlattenBuffer` pass 就是负责这个转换的（`src/transform/flatten_buffer.cc`）。

## 2.5 遍历与改写 IR：`StmtVisitor` / `StmtExprMutator`

**定义位置**：`3rdparty/tvm/include/tvm/tir/stmt_functor.h`

这是本仓库**每个 pass 的父类**。看懂它们你就看懂了 pass 的写法。

> **先扫清两个前置概念**（否则下面的代码会看不懂）：
> 
> **① `ForNode` 是什么？** 就是 2.4 表格里那个 `For` 节点（`for` 循环）**在 C++ 里的底层类型**。TVM 的约定是：每种 IR 节点都有一对类——带 `Node` 后缀的是底层对象（如 `ForNode`），不带后缀的是它的智能指针 wrapper（如 `For`）。写遍历代码时我们拿到的是指向底层节点的指针 `const ForNode* op`，通过 `op->extent`、`op->body` 访问它的字段。（这对类的完整区别 2.6 节会讲，这里先知道"`ForNode` = For 循环节点"即可。）
> 
> **② 为什么"每碰到一个 For 就会自动跑一次 `VisitStmt_(const ForNode* op)`"？** 这是 `StmtVisitor` 的**分发（dispatch）机制**：它有一个总入口 `VisitStmt(stmt)`，会先看这个 `stmt` **运行时到底是哪种节点**，再调用对应的 `VisitStmt_(const XxxNode*)`。伪码大致是：
> 
> ```cpp
> void StmtVisitor::VisitStmt(const Stmt& stmt) {
>   if (stmt 是 ForNode)        VisitStmt_(static_cast<const ForNode*>(...));
>   else if (stmt 是 IfThenElseNode) VisitStmt_(...);
>   else if (stmt 是 BufferStoreNode) VisitStmt_(...);
>   // ... 每种节点类型一个分支
> }
> ```
> 
> 这里要分清**两个不同的函数**（TVM 故意用命名区分）：
> 
> - **`VisitStmt(stmt)`（不带下划线）= 分发器**：只负责"看类型 → 转发到对应的 `VisitStmt_`"，它自己**不知道任何节点有哪些子节点**。
> - **`VisitStmt_(const ForNode* op)`（带下划线）= 针对 For 的具体处理**：TVM 提供的**基类默认实现**里，正是这一层**知道 For 有 `min/extent/body` 这些子节点**，并挨个对它们调 `VisitStmt`/`VisitExpr` 往下走。
> 
> 所以整棵树是这样被**深度优先**走完的：分发器 `VisitStmt` 把每个节点转发到它的 `VisitStmt_`；而每个 `VisitStmt_` 的默认实现再把自己的子节点交回分发器……如此往复。于是你只要 override `VisitStmt_(const ForNode*)`，**遍历一路走下来每经过一个 For 就进你这份实现一次**——这就是"每碰到一个 For 就跑一次"的由来。而你在里面手写的 `StmtVisitor::VisitStmt_(op)`，意思是"我处理完自己这层，**调用基类那份'懂 For 结构'的默认实现**去递归子节点"，**漏了它，子树就不会再被遍历**。

### 2.5.1 `StmtVisitor`：只读遍历

```cpp
class MyAnalyzer : public StmtVisitor {
 public:
  // 遍历经过每一个 For 节点时，都会被分发到这里执行一次
  void VisitStmt_(const ForNode* op) final {
    num_loops_++;                 // 数一个循环
    StmtVisitor::VisitStmt_(op);  // 调用基类默认实现：继续递归 op 的子节点
  }
  int num_loops_ = 0;
};
```

> ⚠️ **这里千万别写成 `StmtVisitor::VisitStmt(op)`**（不带下划线的分发器）。
> `VisitStmt` 是分发器：你把这个 `op`（它是个 `ForNode`）交给它，它一看"是 For"，又转发回 `VisitStmt_(const ForNode*)`——也就是**回到你自己这个函数**，`num_loops_++` 再来一遍……结果要么**无限递归**、要么重复计数。
> 想在 override 里"继续往下递归子节点"，就得调**基类的同名带下划线实现** `StmtVisitor::VisitStmt_(op)`，因为只有它知道 For 有哪些子节点、该怎么走。
> **源码实证**：本节稍后引用的 `producer_consumer_ws.cc` 里，`PipelineLoopFinder::VisitStmt_(const ForNode* op)`（约 2201 行）内部递归用的正是 `StmtVisitor::VisitStmt_(op)`（约 2208 行）；全仓上百处 override 都是这个模式。
>
> 顺便看一眼**基类那份默认实现**长什么样（示意）——它证明了"取子节点是基类干的、你只需把 `op` 交回去"：
>
> ```cpp
> void StmtVisitor::VisitStmt_(const ForNode* op) {
>   this->VisitExpr(op->min);      // 递归访问子表达式：循环下界
>   this->VisitExpr(op->extent);   // 递归访问子表达式：循环长度
>   this->VisitStmt(op->body);     // ★递归访问子语句：循环体
> }
> ```

**怎么启动它**：构造一个对象，把它作用到根节点上，遍历就从这里开始自动铺开：

```cpp
MyAnalyzer analyzer;
analyzer(func->body);   // 等价于 analyzer.VisitStmt(func->body)，从函数体根节点开始走
// 走完之后，analyzer.num_loops_ 就是整个 kernel 里 For 循环的总数
```

**用途**：统计信息、找特征、验证合法性。**不改 IR**。

例子：[`src/cuda/transform/producer_consumer_ws.cc`](../../src/cuda/transform/producer_consumer_ws.cc) 里
的 `FindPipelineLoop` 就是纯 `StmtVisitor`，只负责找出所有带 pipeline 注解的 For。

### 2.5.2 `StmtExprMutator`：可改写遍历

> **先认全 `ForNode` 的字段**（下面重建 `For` 时要一个个用到）。一个 `ForNode` 描述的就是 `for (loop_var = min; loop_var < min + extent; ++loop_var) { body }`：
>
> | 字段 | 含义 |
> |---|---|
> | `op->loop_var` | 循环变量（如 `i`） |
> | `op->min` | 循环**起始值**（下界，通常是 0） |
> | `op->extent` | 循环**长度 / 迭代次数**——**注意不是终点**，终点是 `min + extent` |
> | `op->kind` | 循环类型（serial / parallel / unrolled / 线程绑定…） |
> | `op->body` | 循环体（一个 `Stmt`） |
> | `op->thread_binding` / `op->annotations` | 线程绑定、注解（如软件流水标记） |
>
> 所以循环变量的取值范围是 **`[min, min + extent)`**（源码里 `simplify.cc` 对它加的约束正是 `loop_var >= min` 且 `loop_var < min + extent`）。

```cpp
class MyRewriter : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    // 先递归改写子节点（注意：这里传的是子节点 op->body，不是 op 本身；
    //   VisitStmt 是分发器，会把 body 转发到它对应类型的 VisitStmt_）
    Stmt body = VisitStmt(op->body);
    // 如果 body 没变（还是同一个对象），就原样返回旧的 For，不造新节点
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    // body 变了，才用「旧字段 + 新 body」拼一个全新的 For 返回
    return For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding, op->annotations);
  }
};
```

> **和 2.5.1 的 Visitor 有意不同**：Visitor 里递归写的是 `StmtVisitor::VisitStmt_(op)`（传 `op` 本身、让基类默认实现去取所有子节点）；这里 Mutator 因为要**精确重建**，是**自己**取出 `op->body` 单独递归改写（`VisitStmt(op->body)`），好拿到"新 body"再决定怎么拼。两种写法都能递归，选哪种取决于你是"只读遍历"还是"要返回新节点"。

**用途**：几乎所有真正做事的 pass。

**心法**：

1. **子节点先递归**：`Stmt body = VisitStmt(op->body)` 让子树先被改，你只处理"我这一层的新逻辑"。
2. **不可变 + 改写=造新节点**：TVM 的 IR 节点几乎都是**不可变的**——造好之后**不允许原地改字段**（不能 `op->extent = 128`）。想"改"一个节点，只能**保留旧节点不动，用它的旧字段 + 你要换的那部分，拼一个全新节点返回**（上面 `return For(..., body, ...)` 就是：只有 `body` 换成新的，其余字段照抄旧 `op`）。这也是为什么 Mutator 的每个 `VisitStmt_` 都要 **return 一个 Stmt**（而 Visitor 只读、不返回）。
   > **为什么要这么麻烦？** 因为 IR 子树常被**多处共享**（同一个节点被多个父节点引用）。若允许原地改，就会"改了 A 顺带改到 B"，制造隐蔽 bug。不可变 = 谁都改不到别人，共享绝对安全。
3. **`same_as` 短路**：`same_as` 判断两个引用**是不是同一个对象**（比指针，`O(1)`，不比内容）。上面 `if (body.same_as(op->body))` 的意思是"这棵子树根本没被改动"，于是**直接返回旧 `op`、不造新节点**。有了它，一次 pass **只会重建"从改动点到根"那一条链**，没碰到的子树全部复用旧对象——所以"造新节点"在实践中并不昂贵。
4. **返回类型 & 命名约定**：`VisitStmt_(ForNode*)` 返回 `Stmt`；`VisitExpr_(AddNode*)` 返回 `PrimExpr`。方法名结尾的 `_` 是 TVM 约定——不带 `_` 的 `VisitStmt(Stmt)` 是**分发器**，会根据实际节点类型调用带 `_` 的子版本。

### 2.5.3 组合工具：`IRMutatorWithAnalyzer`

**位置**：`3rdparty/tvm/include/tvm/tir/analysis.h` + `arith::Analyzer`。

很多 pass 需要在改写的同时**做常量化简 / 区间推理**。TVM 提供了 `IRMutatorWithAnalyzer`：
它继承 `StmtExprMutator`，内部持有一个 `arith::Analyzer`，会自动帮你：

- 进入 `For` 时把循环变量的范围（`min <= var < min + extent`）绑进 analyzer
- 进入 `IfThenElse` 时把条件当成 `bool_constraint` 绑进 analyzer
- 你可以在任意时刻 `analyzer_.CanProve(expr)` 判断某个断言是否恒真

**看一段真实代码**——`src/transform/simplify.cc` 里的 `StmtSimplifier`（继承 `IRMutatorWithAnalyzer`）处理 `For` 时就把上面三点全用上了：

```cpp
// 摘自 src/transform/simplify.cc（StmtSimplifier::VisitStmt_(const ForNode*)）
Stmt VisitStmt_(const ForNode *op) final {
  // ① 用 analyzer 证明"循环次数 <= 0"，能证明就说明这循环压根不会执行 → 直接删掉
  if (analyzer_->CanProve(op->extent <= 0)) {
    return Evaluate(0);                       // 空语句，等于把这个 for 删了
  }
  // ② 进入循环体前，把"循环变量的范围"作为约束绑进 analyzer：
  //    loop_var >= min  且  loop_var < min + extent
  //    With<ConstraintContext> 是 RAII——出了这个作用域约束自动解除
  With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
  With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
  // ③ 递归改写循环体（Parent = IRMutatorWithAnalyzer，即基类默认递归）
  //    因为上面绑了约束，body 里的化简就能用上"loop_var 在这个范围内"这个信息
  return Parent::VisitStmt_(op);
}
```

体会一下这段为什么强：正是因为进 body 前绑了 `loop_var ∈ [min, min+extent)`，body 里像 `if (i < min+extent)` 这种恒真条件才能被 analyzer 直接化简掉。**这就是"边遍历边带着上下文推理"的价值**，也是为什么这么多 pass 选择继承 `IRMutatorWithAnalyzer` 而不是裸的 `StmtExprMutator`。（另一个典型用例见 `src/transform/loop_partition.cc`。）

## 2.6 引用计数 & `ObjectRef` 家族

看 TVM 源码你会经常见到：

```cpp
class ForNode : public StmtNode { ... };
class For : public Stmt { ... };
```

**约定**：类名带 `Node` 后缀的是**底层可变对象**（几乎没人直接用），
类名不带 `Node` 的是**智能指针 wrapper**（大家用的都是它）。

```cpp
For f = ...;               // 指针
const ForNode* op = f.get();  // 拿到底层节点
op->extent;                // 访问字段
```

`For` 继承自 `Stmt` 继承自 `ObjectRef`。`ObjectRef` 就是"带引用计数的智能指针"，
所有 TIR 节点都是不可变（immutable）的——你改一个 For 就意味着造一个新的 For 对象，
原来的 For 引用计数减 1、可能被回收。

**为什么要不可变**？因为 pass 之间常常会共享子树，如果一个 pass 就地修改了一个节点，
其他持有它引用的 pass 就崩了。不可变 + 引用计数 = 天然的 "只写不改"，
副产物就是你的 `same_as` 检查非常便宜。

## 2.7 从"文本 IR"回到"C++ AST"：找到节点定义

看 pass 源码时，你经常需要问："这个 `AttrStmtNode` 里的 `attr_key` 都能是啥？"
答案永远在**头文件**。

导航技巧：

```
你看到 XxxNode → 去 3rdparty/tvm/include/tvm/tir/{stmt,expr,buffer,function}.h 搜 "class XxxNode"
你看到 tir::attr::yyy → 去 3rdparty/tvm/include/tvm/tir/stmt.h 底部搜 "constexpr const char* yyy"
你看到 tl::builtin::zzz() → 去 src/op/builtin.h 搜 "zzz"
```

第 3 章开始我们就会大量用到这些常量（比如 `tir::attr::software_pipeline_stage`、
`tl::builtin::mbarrier_wait_parity`）。

## 2.8 一个练习：手写一个玩具 pass

到目前你已经能读懂下面这个玩具 pass 了。它的功能是：**把所有 `For` 的 `extent` 打印出来**。

```cpp
// pseudo-file: my_toy_pass.cc
#include <tvm/tir/stmt_functor.h>
#include <tvm/ffi/reflection/registry.h>
#include <iostream>

namespace tvm { namespace tir {

class PrintForExtents : public StmtVisitor {
 public:
  void VisitStmt_(const ForNode* op) final {
    std::cout << "For over " << op->loop_var->name_hint
              << ", extent=" << op->extent << "\n";
    StmtVisitor::VisitStmt_(op);  // 记得递归
  }
};

// 包装成一个可从 Python 调用的 pass
Pass PrintForExtentsPass() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    PrintForExtents v;
    v(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.tilelang.PrintForExtents", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PrintForExtents", PrintForExtentsPass);
});

}}  // namespace
```

（真实 pass 注册模板见第 4 章。）

> ⚠️ **常见误解**
> 
> - **"`Buffer` 就是那块内存"** —— 错。`Buffer` 只是**"如何访问一块内存"的描述符**（shape / dtype / strides / scope），真正的内存由 `Allocate` 语句或外部传入的指针提供。同一块内存可以有多个 `Buffer` 视图（不同 stride / 不同 dtype 重解释），这正是很多 layout/量化技巧的基础。
> - **"改 IR 就是原地把某个字段改掉"** —— 错。TIR 节点**不可变**：任何"修改"都是**造一个新节点**、把旧节点替换掉。所以 pass 里你看到的永远是 `return NewNode(...)`，而不是 `node.field = ...`。
> - **"`Stmt` 和 `Expr` 差不多"** —— 别混。`Expr` **有值、无副作用**（能出现在 `=` 右边）；`Stmt` **无值、通常有副作用**（`Store` / `For` / `IfThenElse`）。判断一个节点该继承谁，就看它"能不能被求值成一个数"。

## 2.9 本章要带走的三件事

1. **TIR 是一棵树**，节点分 `Stmt`（无值有副作用）和 `Expr`（有值无副作用），根是 `PrimFunc`。
2. **所有 pass 的通用套路** = `StmtExprMutator`/`StmtVisitor` 递归下降 + 选择性替换节点。
3. **不可变 + 引用计数**：改一个节点=造一个新节点，`same_as` 判断"要不要真的重构"是常用技巧。

---

下一章 [第 3 章 · TileLang DSL 层次](./03_tilelang_dsl.md)：
我们把 `T.Kernel`、`T.Pipelined`、`T.copy`、`T.gemm` 每一个都对应到具体源码，
搞清楚它们在 IR 里到底长什么样。
