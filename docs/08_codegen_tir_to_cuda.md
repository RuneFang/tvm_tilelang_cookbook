# 第 8 章 · Codegen：TIR → CUDA 源码 → cubin

> **TL;DR**：codegen 就是一个**深度优先遍历 TIR 树、逐节点打印 CUDA C++ 字符串**的过程（`CodeGenTileLangCUDA`）——它**不做任何优化**，看到什么打印什么。所以"生成的 CUDA 里有奇怪的东西"几乎总是**上游 pass 的锅**，不是 codegen 的锅。
>
> **本章目标**
> 前面 7 章我们一直在 **TIR 层** 折腾——不管是 pass pipeline、software pipeline，还是 layout / fragment，最终"改的东西"都是 `PrimFunc` 里的语句树。可是 GPU 并不认识 TIR，它只吃 **PTX/cubin**。这中间的桥叫 **codegen**（代码生成器）。
> 本章会带你把这座桥拆开：一步步看 TileLang 是如何把一个 `PrimFunc` 变成一段可读的 CUDA C++ 源码字符串，再交给 NVCC / NVRTC 编成 `cubin`，最后打包进 TVM runtime module。读完之后，你能自己在 CI 日志里定位是"IR 写坏了"还是"codegen 输出有 bug"，也能在需要时给 codegen 新增一个 intrinsic。

---

## 8.1 三段式：先看整个流程

用一张最粗的图先把整章串起来：

```
   PrimFunc (device 部分，已经 lowering 完)
             │
             ▼
   ┌───────────────────────────────┐
   │ (1) BuildTileLangCUDA          │  C++
   │     CodeGenTileLangCUDA cg;    │
   │     cg.AddFunction(...)        │  遍历 IR，把每个节点转成字符串
   │     std::string code = cg.Finish();
   └───────────────────────────────┘
             │  code 是可以复制到 .cu 文件里编译的 CUDA C++ 源码
             ▼
   ┌───────────────────────────────┐
   │ (2) tilelang_callback_cuda_compile │  Python，通过 FFI 回调
   │     → nvcc.compile_cuda(code)      │  用 NVCC 或 NVRTC 编成 PTX/cubin
   └───────────────────────────────┘
             │  ptx 是 bytes
             ▼
   ┌───────────────────────────────┐
   │ (3) CUDAModuleCreateWithFallback │  C++
   │     打包成 tvm::runtime::Module   │  可 GetFunction("gemm") 拿到句柄
   └───────────────────────────────┘
```

三段的分界点非常干净：**IR ↔ 字符串**是第 (1) 段，**字符串 ↔ 二进制**是第 (2) 段，**二进制 ↔ runtime**是第 (3) 段。下面按顺序拆。

---

## 8.2 入口注册：谁把 IRModule 交给 codegen？

第 5 章讲过 `engine/lower.py` 里最后一句 `codegen.build_module(...)`。那句话背后靠的是一个"target 名字 → codegen 函数"的映射表，注册器就是[device_codegen.py](../../tilelang/backend/device_codegen.py)。

关键片段（真实源码，第 34–46 行）：

```python
@dataclass(frozen=True, slots=True)
class DeviceCodegen:
    """Device codegen entry points for one backend target variant."""

    name: str
    build: DeviceCodegenFunc | None = None
    build_without_compile: DeviceCodegenFunc | None = None
    supports_target: TargetPredicate | None = None

    def lower(self, mod: IRModule, target: Target, *, compile_device: bool) -> IRModule:
        build_func = self.build if compile_device else self.build_without_compile
        ...
        return build_func(mod, target)
```

它做了两件事：

1. **`build`** —— 走完整通路：TIR → CUDA 源码 → cubin，运行时可直接 launch。
2. **`build_without_compile`** —— 只走前两步的 TIR → CUDA 源码，把 cubin 那步略过。用来 dump 中间源码看，或者在 CI 上把 target 是 sm_120 的 kernel 也"lower 过一遍看看会不会崩"（因为机器上可能没有 sm_120 的编译器）。

CUDA 后端在 [cuda/codegen.py](../../tilelang/cuda/codegen.py) 里做注册：

```python
register_device_codegen(
    "cuda",
    DeviceCodegen(
        "cuda",
        build=global_func_device_codegen("target.build.tilelang_cuda"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_cuda_without_compile"),
        supports_target=_is_plain_cuda_target,
    ),
    override=True,
)
```

`global_func_device_codegen` 里发生了一件对小白很关键的事：

```python
def global_func_device_codegen(global_func_name: str) -> DeviceCodegenFunc:
    def build(mod, target):
        return tvm.ffi.get_global_func(global_func_name)(mod, target)
    return build
```

也就是说，Python 只是**通过 FFI 名字**去查一个 C++ 函数。真正的 codegen 是 C++ 实现。名字 `"target.build.tilelang_cuda"` 在 C++ 侧由这个宏注册：

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}
```

（[src/cuda/codegen/rt_mod_cuda.cc](../../src/cuda/codegen/rt_mod_cuda.cc) 末尾）

> 🧠 **理解 FFI**：TVM/TileLang 大量用"字符串名字"跨语言绑定函数。你在 Python 里写的 `tvm.ffi.get_global_func("xxx")` 拿到的是一个能被当函数调用的 handle，实际执行时进入的是 C++ 那一行 `.def("xxx", CppFunc)`。这是 TVM 生态最基础的架构约定，看到 `.def(...)` 你就知道"这是 C++ 侧对 Python 开的一扇窗"。

---

## 8.3 (1) BuildTileLangCUDA：从 IRModule 到 CUDA 源码字符串

这就是**整个第 (1) 段的主干**，就在 [rt_mod_cuda.cc](../../src/cuda/codegen/rt_mod_cuda.cc) 的 `BuildTileLangCUDA`。我把它拆成 4 步逐行讲。

### 8.3.1 构造 code generator

```cpp
Module BuildTileLangCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);
  ...
```

`CodeGenTileLangCUDA` 是主角——一个"访问器 + 字符串拼接器"。它继承自 TVM 上游的 `CodeGenC`（见 [codegen_cuda.h](../../src/cuda/codegen/codegen_cuda.h) 第 23 行）。

`Init(output_ssa=false)` 告诉它"生成普通 C++ 风格代码，不需要 SSA 单赋值形式"。SSA 是编译器内部常用的一种 IR 形式（每个变量只被赋值一次），但对我们要打印的最终 CUDA 源码来说没必要。

### 8.3.2 遍历每个 PrimFunc

```cpp
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }
```

对 IRModule 里的每个 device 函数（第 5 章讲过的 `split_host_device` 之后剩下的那一半）：

- 先断言 calling_conv 必须是 `kDeviceKernelLaunch`——host 函数一定要在前面的 pass 里被剥离掉，不能在这一步还混进来。
- 调 `cg.AddFunction(gvar, f)` 把这个函数追加到 codegen 内部的 `std::ostringstream stream` 里。

`AddFunction` 会做的事，简单说是三步：**打印函数签名 → 打印参数声明 → 递归遍历 body**。递归遍历 body 时，就会调到 `codegen_cuda.h` 里覆盖的那一大堆 `VisitStmt_` / `VisitExpr_`。

### 8.3.3 拿到最终字符串

```cpp
  std::string code = cg.Finish();
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
```

`Finish()` 会把这些拼接完的 body 和所需 `#include`、外部 helper 声明加进去，返回**一整段可以塞到 `.cu` 文件里的字符串**。

紧跟着有一个 `tilelang_callback_cuda_postproc` 回调 hook——一个可选钩子：如果用户注册了这个函数（比如你想把生成的 kernel 名字加个后缀，或者往里插几行 pragma），可以在这个位置改字符串。默认情况下没有注册，直接跳过。

### 8.3.4 交给编译回调，拿到 PTX/cubin

```cpp
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    ptx = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "cubin";
  } else {
    ICHECK(0);
  }
```

这就是**从 C++ 反向调 Python** 的一步（第 (1) 段和第 (2) 段的过渡）。它靠 `tilelang_callback_cuda_compile` 这个 FFI 名字回到 Python。下节我们进 Python 侧看这个函数。

> ⚠️ 那个 `ptx[0] != '/'` 是个小 trick：如果 Python 编译回调返回的是**PTX 文本**（一个大字符串，第一个字符往往是 `/`——`.version` 那些注释起始），那么 `fmt="ptx"`；如果是 **cubin 二进制**（首字节几乎不可能是 `/`），那么改成 `fmt="cubin"`。这是 TileLang 让一个回调既能返回 ptx 又能返回 cubin 的一种约定。

### 8.3.5 打包成 runtime Module

```cpp
  Map<String, String> source_map;
  source_map.Set("cuda", code);
  return target::CUDAModuleCreateWithFallback(
      Bytes(ptx.data(), ptx.size()), String(fmt),
      ExtractFuncInfo(mod), source_map);
}
```

- **`Bytes(ptx.data(), ptx.size())`**：把 ptx / cubin 打成 `Bytes`。
- **`ExtractFuncInfo(mod)`**：从 PrimFunc 属性中提取"每个 kernel 的参数 dtype 列表 / launch params / cluster dims"等元信息（`ExtractFuncInfo` 函数就在同文件上方）——这些信息 runtime 侧 launch kernel 时要用。
- **`source_map`**：把原始 CUDA 源码也一并塞进去，这样以后 `mod.get_source("cuda")` 可以在运行时反查生成的源码。第 6 章讲的"对生成源码扫黑名单字符串"这类硬签名测试，就是靠 `mod.get_source("cuda")` 拿到源码后 `assert "<bug marker>" not in source` 实现的。

---

## 8.4 CodeGenTileLangCUDA：一个"访问器"到底是什么？

第 (1) 段最核心的类。这是本章对小白最重要的一节，因为**"AST 访问器 + `std::ostringstream`"是所有编译器 codegen 的通用套路**。搞懂它，你能读懂任何 codegen（Clang、GCC、LLVM 前端等等都是这个套路）。

### 8.4.1 类的样子

来自 [codegen_cuda.h](../../src/cuda/codegen/codegen_cuda.h)：

```cpp
class CodeGenTileLangCUDA final : public CodeGenC {
public:
  CodeGenTileLangCUDA();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  ...
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  ...
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;
```

看这一堆方法名，你会发现**每种 IR 节点都有一个对应的 `VisitStmt_` 或 `VisitExpr_`**。比如：

- 遇到 `ForNode`（一个 for 循环）→ 调 `VisitStmt_(const ForNode*)`。
- 遇到 `CallNode`（一次函数调用，比如 `T.gemm(A, B, C)`）→ 调 `VisitExpr_(const CallNode*, os)`。
- 遇到 `BufferStoreNode`（`A[i] = 3`）→ 调 `VisitStmt_(const BufferStoreNode*)`。

这就是**访问器模式（Visitor Pattern）**。设计意图是：**"节点的定义"和"节点的处理逻辑"解耦**。同一份 TIR，可以既被 codegen 访问（打印成 CUDA 源码），也被优化器访问（改成另一段 TIR），也被检查器访问（做正确性 lint），互不干扰。

### 8.4.2 举个具体例子：ForNode 是怎么变成 `for (int i = ...)` 字符串的？

`CodeGenC` 基类里默认已经能处理 for，但 TileLang 有自己的 `#pragma unroll` 等 CUDA 特化行为，就 override 掉了 `VisitStmt_(const ForNode*)`。逻辑大致是（伪代码）：

```cpp
void CodeGenTileLangCUDA::VisitStmt_(const ForNode* op) {
  std::string loop_var = AllocVarID(op->loop_var.get());   // "i"
  // 打印 pragma（如果有 unroll_factor 属性）
  if (unroll_factor.count(op->loop_var.get())) {
    stream << "#pragma unroll " << unroll_factor[op->loop_var.get()] << "\n";
  }
  // 打印 for header
  PrintIndent();
  stream << "for (int " << loop_var << " = " << op->min << "; "
         << loop_var << " < " << op->min + op->extent << "; ++"
         << loop_var << ") {\n";
  // 递归 visit 循环体
  BeginScope();
  this->VisitStmt(op->body);
  EndScope();
  PrintIndent();
  stream << "}\n";
}
```

关键动作只有三个：

1. **`stream << "for (int i = ...) {"`** —— 往内部字符串流写文字。这个 `stream` 就是 `CodeGenC` 里那个 `std::ostringstream`。
2. **`this->VisitStmt(op->body)`** —— 递归下去，让 body 里的每个语句自己再打印自己。这就是 visitor 递归的核心：**"我知道该怎么打印我自己，但我不知道我 body 里长啥样——它自己会打印它自己"**。
3. **`stream << "}"`** —— 收尾。

Finish() 就是把内部 `stream.str()` 加上 `#include` 之类的 header 拼一起。

### 8.4.3 codegen 也会带"决定要不要 `#include`"的副作用

再看 `codegen_cuda.h` 私有成员：

```cpp
bool enable_fp16_{false};
bool enable_bf16_{false};
...
bool need_barrier_h_{false};
bool need_mma_h_{false};
```

当 codegen 访问到 `float16` 相关的东西时，`enable_fp16_ = true`；访问到 mbarrier 相关的 intrinsic 时，`need_barrier_h_ = true`。这些 flag 在 `Finish()` 的时候会决定"最终字符串开头要不要多一句 `#include \"tl_templates/cuda/mbarrier.h\"`"。

这是**codegen 一个很常见的模式：一边走一边做需求探测**。因为等到走完了才知道"这段代码到底用到了什么"。

---

## 8.5 (2) tilelang_callback_cuda_compile：从源码到 cubin

上面第 8.3.4 节说到 C++ 回调 Python 的那个 `tilelang_callback_cuda_compile`。它就在 [engine/lower.py](../../tilelang/engine/lower.py) 第 101 行：

```python
@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target, pass_config=None):
    target_arch, target_code = nvcc.get_target_arch_and_code(target)
    target_code_list = nvcc.get_target_code_list(target_code)
    gencode_code = nvcc.format_target_code_for_gencode(target_code)
    if gencode_code is None:
        arch = [f"-arch=sm_{target_arch}"]
    else:
        arch = ["-gencode", f"arch=compute_{target_arch},code={gencode_code}"]
    compile_format = "fatbin" if len(target_code_list) > 1 else "cubin"
    ...
```

逐行拆：

- **`nvcc.get_target_arch_and_code(target)`** —— 从 `Target` 里提取 arch，比如 `target_arch="90a"`。sm_90a 是 Hopper 的 GPU 架构。
- **`arch = ["-gencode", "arch=compute_90a,code=sm_90a"]`** —— 这就是等下要塞给 `nvcc` 的 `-gencode` 参数。等价于命令行里的 `nvcc -gencode arch=compute_90a,code=sm_90a ...`。
- **`compile_format = "fatbin"`** —— 如果编译目标包含多个 sm（比如同时打 sm_80 和 sm_90），就出 fatbin（多 GPU 二进制包）；否则出 cubin（单一 GPU 二进制）。

接下来：

```python
    options = [
        "-std=c++20",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
    ]
    ...
    if enable_fast_math:
        options.append("--use_fast_math")
```

这几行组装 NVCC 命令行选项。重点：

- **`-std=c++20`** —— 因为 `tl_templates/cuda/reduce.h` 用到了 C++20 的显式 lambda 模板参数语法。
- **`TILELANG_TEMPLATE_PATH`** —— 就是仓库里 [src/tl_templates/cuda/](../../src/tl_templates/cuda) 那些手写的 helper。你 codegen 出来的 CUDA 源码里那些 `tl::gemm_ss<...>(...)`、`tl::mbarrier_wait(...)`，就都在这个目录里定义。

再往下：

```python
    cache_key = CUDABinaryCache.make_key(
        code=code, target_kind=target.kind.name, target_arch=target_arch, ...
    )
    cached_binary = CUDABinaryCache.load(cache_key, compile_format)
    if cached_binary is not None:
        return bytearray(cached_binary)

    ptx = nvcc.compile_cuda(
        code,
        compile_format,
        arch,
        options=options,
        verbose=verbose,
    )
    CUDABinaryCache.save(cache_key, compile_format, ptx)

    return ptx
```

**这是一个磁盘缓存的短路**：如果这段源码 + 这套选项之前编译过，就直接返回缓存里的 cubin，省下几十秒的 NVCC 时间。这也是为什么 TileLang 的 dev loop 只有第一次慢，后面重复跑非常快。

最后 `nvcc.compile_cuda(...)` 才是真正的调 NVCC 子进程（`contrib/nvcc.py` 里，它 fork 出 `nvcc` 命令行、把源码写到临时文件、拿回 cubin）。

> 💡 **两条并存的编译路径：NVCC vs NVRTC**
>
> 上面走的是 NVCC（一个独立的可执行程序，会读 `.cu` 文件、fork 子进程）。TileLang 还支持 NVRTC——把 CUDA 源码作为字符串**在进程内**直接编成 cubin，不需要额外的可执行文件。执行后端选择在 [cuda/execution_backend.py](../../tilelang/cuda/execution_backend.py)：`tvm_ffi` 走 NVCC，`nvrtc` 走 NVRTC。选择由环境变量或 JIT 配置决定。原理上产物一致，只是入口不同。

---

## 8.6 (3) 打包成 runtime Module

回到 C++。`BuildTileLangCUDA` 末尾那句：

```cpp
return target::CUDAModuleCreateWithFallback(
    Bytes(ptx.data(), ptx.size()), String(fmt),
    ExtractFuncInfo(mod), source_map);
```

`CUDAModuleCreateWithFallback` 是 TVM 上游的 runtime 工厂。它做的事：

1. 把 cubin bytes 加载进 CUDA driver（`cuModuleLoadData`）；
2. 拿到一个 `CUmodule` 句柄；
3. 用 `ExtractFuncInfo` 里的 param dtype、launch params 信息，为每个 kernel 建立一个 `PackedFunc`；
4. 把这些 PackedFunc 塞进 `tvm::runtime::Module`；
5. 把源码字符串挂在 module 上（后面 `mod.get_source("cuda")` 就能拿到）。

`ExtractFuncInfo` 我们已经在 8.3 节看到过，它做的关键几件事：

```cpp
if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
  launch_param_tags.push_back(
      runtime::launch_param::kUseProgramaticDependentLaunch);
}
if (f->HasNonzeroAttr("use_cooperative_groups")) {
  launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
}
if (f->GetAttr<Array<Integer>>("cluster_dims").defined()) {
  launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
  launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
  launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
}
```

把 PrimFunc 上的属性翻译成"下次 launch kernel 时该传什么额外参数"：

- `kHasGridSync` → 用 PDL（Programmatic Dependent Launch，Hopper 上一种 kernel 间流水化的机制）。
- `use_cooperative_groups` → 用 cooperative launch，让 grid 内所有 block 可以同步。
- `cluster_dims` → 传 cluster 维度参数（sm_90 的 thread block cluster 特性）。

这些 tag 在下一章（第 9 章 runtime）会再出现。

---

## 8.7 亲手做一次：把生成的 CUDA 源码 dump 出来看

**这才是这一章的实践意义**——只要你写一个小 kernel，就能自己打印出上面所有理论对应的**真实源码字符串**：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main

kernel = matmul(1024, 1024, 1024)

# ---- 就是这一行 ----
print(kernel.get_kernel_source())
```

你会看到一段大约 100–200 行的 CUDA C++ 源码，里面能识别出：

- **函数签名**：`extern "C" __global__ void __launch_bounds__(128) main_kernel(...)`
- **共享内存声明**：`__shared__ __align__(16) half A_shared[128 * 32];`
- **mbarrier 数组**：`__shared__ __align__(16) tl::Barrier _mbarrier[...];`（第 6 章讲的 Barrier 类型）
- **producer/consumer 分支**：`if (threadIdx.x >= 128) { ... } else { ... }`（第 6 章的 warp specialization）
- **phase counter**：`int producer_phase_cnt[1];` 或 `int consumer_phase_cnt[1];`（如果 kernel 命中 phase counter 分支）

**把 phase counter 相关的字符串搜索一下**，就能非常直观地看到第 6 章讨论的那些概念是怎么变成源码的。这也是"对生成源码断言某条字符串是否出现"这类硬签名测试的实操依据。

> 💡 **调试小技巧**：如果你在开发一个新的 codegen intrinsic，最快的调试方式就是"改一行 → `print(kernel.get_kernel_source())` → 肉眼看输出对不对"。不用等 nvcc 编、不用跑 GPU，几秒钟一个 cycle。

---

## 8.8 常见踩坑

几条常见问题作为章末的实用清单：

**1. 生成的源码里出现了裸的 intrinsic 名（比如 `tl.mvb_stage_index(...)`）**
说明前面的某个 pass 应该剥离掉这个 intrinsic 但没剥（第 6 章讲过这个陷阱）——codegen 从上到下遇到没见过的 CallNode，就会原样打印。**codegen 之前一定要保证 IR 里没有它不认识的 op**。

**2. `Cannot find PackedFunc target.build.tilelang_cuda`**
说明 `import tilelang` 时 `cuda/codegen.py` 那句 `register_device_codegen` 没跑到——通常是 lazy import 出了问题。检查 `_LAZY_DEVICE_CODEGENS` 里有没有注册 `"cuda"` 到某个 module。

**3. `ICHECK failed: calling_conv == CallingConv::kDeviceKernelLaunch`**
说明有一个 host 侧的 PrimFunc 混进了 device codegen。往前找 `SplitHostDevice` 有没有正确执行，或者你的 PrimFunc 是不是漏了 attach `T.attr("target", ...)`。

**4. 编译出来的 cubin 能跑，但结果错**
不要怀疑 codegen（它只是字符串拼接），先怀疑上游的 pass 生成的 IR 本身是不是错的。**用 `kernel.get_kernel_source()` 打印源码 + 手工肉眼审 CUDA，比 debug pass 快得多**。第 6 章讲的那类"索引 / phase 被误改写"的 bug，通常就是这样抓到的。

**5. 生成的源码里缺少 `#include`**
说明你新加的 intrinsic 在 codegen 里没 flip 对应的 `need_xxx_h_ = true`。回头看 8.4.3 那一节。

---

## 8.9 小结

三段式复习一遍：

| 阶段 | 从什么到什么 | 关键代码 | 层 |
|---|---|---|---|
| (1) TIR → CUDA 源码 | IRModule → std::string | `CodeGenTileLangCUDA::AddFunction/Finish` | C++ |
| (2) CUDA 源码 → PTX/cubin | std::string → bytes | `tilelang_callback_cuda_compile` → NVCC/NVRTC | Python |
| (3) 二进制 → runtime | bytes → `tvm.runtime.Module` | `CUDAModuleCreateWithFallback` | C++ |

**记住这三条最有价值的经验：**

1. **codegen 就是 AST 访问器 + 字符串拼接**，别把它想复杂——它不会做优化，不会做安全检查，来什么打什么。
2. **`kernel.get_kernel_source()` 是你最好的朋友**，遇到玄学问题先 dump 源码。
3. **codegen 之前 IR 必须干净**——所有内部 intrinsic 都要被上游 pass 剥离掉。这是"provenance vs syntax"原则在 codegen 侧的实际后果。

下一章我们进 [第 9 章 · Runtime / JIT / Kernel Cache](./09_runtime_jit_kernel_cache.md)：这个打包好的 `tvm.runtime.Module` 是怎么被 Python 拿去 launch 的，`kernel(A, B)` 这一句背后又发生了什么。
