# 第 11 章 · 调试与可视化工具链

> **TL;DR**：调 TileLang 的核心心法是**"让编译器把中间产物吐出来给你看"**——从 `mod.script()`（看 IR）到 `get_kernel_source()`（看生成的 CUDA）到 layout 可视化。本章把这些内置工具按"编译失败 / 结果不对 / 性能不达标"三类问题排成一张对照矩阵，出问题时照着挑。
>
> **本章目标**
> 把「写 TileLang → 编译失败 / 结果不对 / 性能不达标」这三种日常问题，映射到 TileLang 内置的 **8 种官方工具**，从最简单的到最重量级。学完你就有一条**日常闭环**：出问题 → 挑对工具 → 看到最直接的证据 → 定位到某一个 pass 或某一行代码。

> **本章不虚构 API**。每一个函数名、环境变量、命令行、字段名，都对应 `tilelang/` 或 `docs/` 里的真实源码。我在每一节末尾都会附上「事实出处」，你可以自己 grep 复核。

---

## 11.0 先建立心智模型：三类问题 vs 工具矩阵

TileLang 编译一个 kernel，会经过大约这样三层：

```
你的 Python DSL
      │
      ▼  ①lower（一串 pass，每个 pass 改写 TIR）
  一堆 TIR IR 版本
      │
      ▼  ②codegen（把最后一版 TIR 翻译成 CUDA C 源码）
  CUDA C 源码字符串
      │
      ▼  ③nvcc / hipcc 编译
  可执行 kernel（.so / cubin）
      │
      ▼  ④在 GPU 上跑
  数值结果 + 性能
```

大部分「莫名其妙的错」都能对号入座到某一层：

| 症状 | 最可能出在 | 首选工具 |
|---|---|---|
| lower 阶段抛异常 | ① | **打印中间 IR** + **Pass Diff** |
| lower 成功但 nvcc 编译报错 | ② | **post-processing callback** dump CUDA 源码 |
| 编译成功但数值不对 | ①/②/④ | **`T.print`** + **layout visualization** + **Pass Diff** |
| 结果对但慢 | 硬件维度 | **Analyzer** 做静态 roofline；然后再上 Nsight |
| bug 复现代码几百行、想缩小 | 元问题 | **AutoDD** 自动 delta-debug |

> 术语提醒（前面章节讲过，这里只作一个 anchor）：
> - **pass**：一个把 TIR 改写成另一版 TIR 的函数。见 [第 4 章](./04_pass_system.md)。
> - **fragment**：register file 里的分块布局。见 [第 7 章](./07_layout_and_fragment.md)。
> - **PrimFunc / IRModule**：TIR 的两级容器。见 [第 2 章](./02_tvm_tir_basics.md)。

---

## 11.1 最轻量的调试：`T.print`

`T.print` 是**运行时**的调试原语——就像在 CPU 代码里 `print(x)` 一样，只不过它在 GPU 内核里执行，会把值发回 host 端 stdout。

真实签名（在 `tilelang/language/print_op.py`）：

```python
def print(obj=None, msg="", warp_group_id=0, warp_id=0) -> None: ...
```

用法示例（这段是从 `docs/tutorials/debug_tools_for_tilelang.md` 的官方示例改写的**能跑**的最小版本，只删了跟本章无关的 kernel body）：

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def kernel():
    with T.Kernel(1, threads=8) as (bx,):
        tid = T.get_thread_binding()
        # 只让 thread 0 打，避免 8 份同样输出
        if tid == 0:
            T.print(tid, msg="hello world")

kernel()  # 会在 host 端看到：msg='hello world' BlockIdx=(0,0,0), ThreadIdx=(0,0,0): 0
```

**⚠️ 三个必须注意的点**（这些是官方教程里明确写了的坑）：

1. **`T.print` 是 GPU 内并发执行的**——如果不加 `if tid==0:`，你会看到 `blockDim × gridDim` 份重复输出，且乱序。
2. **打 buffer 时不要打全部**，用 `elems=N` 限制打前 N 个元素。`print_shared_buffer_with_condition` 等更细分的形式就是给这个用的。
3. **打 fragment (register) 里的值**要用 fragment 专用形式；因为 fragment 在寄存器里，跟 shared / global 内存打印路径不一样。见 `print_fragment_buffer_with_condition`。

> **事实出处**：`tilelang/language/print_op.py` 的 8 个函数 —— `print_var` / `print_var_with_condition` / `print_global_buffer_with_condition` / `print_shared_buffer_with_condition` / `print_fragment_buffer_with_condition` / `print_msg` / `print_local_buffer_with_condition` / `print`（顶层入口，会分派到上述之一）。

---

## 11.2 拦截生成的 CUDA 源码：post-processing callback

这是**排查 codegen bug** 或者**手工微调最终 CUDA** 的杀手锏。

TileLang 在 codegen 完成后，会去查一个叫 `tilelang_callback_cuda_postproc` 的全局注册函数。如果你注册了，它就把最终 CUDA 源码字符串交给你，你想 print 就 print，想改就改（改完 return 的字符串会被真正拿去 nvcc）。

真实注册方式（`tilelang/engine/callback.py` 里公开了装饰器）：

```python
import tilelang
import tilelang.language as T
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, target):
    print("=" * 40, "GENERATED CUDA", "=" * 40)
    print(code)
    # 也可以在这里改字符串，例如加个 #pragma unroll、注入 debug printf
    return "// modified by callback\n" + code

# 之后正常 tilelang.compile / @tilelang.jit 都会触发
kernel = tilelang.compile(my_prim_func, target="cuda")
print(kernel.get_kernel_source())   # 拿到最终字符串
```

**典型使用场景**：

- nvcc 报「第 1257 行第 40 列，某个 intrinsic 不认识」→ 直接 dump 出来看那一行到底长什么样，判断是 codegen 出的还是 lower 阶段就错的。
- 想验证「我的 pass 有没有生成预期的指令」→ 在字符串里 grep 一下 `ptx_mma` / `cp.async` / `tma_load`。
- **关键的回归测试**：正确性 diff 有时候对不出 corner-case，但 kernel 源码里有没有 bug 特征字符串是**硬签名**。这就是第 6 章 6.9.5 讲的思路：不仅比数值，还要 grep 生成源码里的 bug marker。

---

## 11.3 中量级：Pass Diff——看每一个 pass 改了什么

上面两招是"事后看结果"。真正的**编译器 bug 定位**通常需要"事中看 IR 是怎么一步步走坏的"，这就是 Pass Diff。

Pass Diff 会在**每个 pass 前后**都 dump 一份 TIR，然后算 unified diff。

### 用法 A：环境变量（推荐日常用）

```bash
# 三种模式
TILELANG_PASS_DIFF=terminal python my_script.py   # 终端彩色 diff
TILELANG_PASS_DIFF=html     python my_script.py   # 生成 HTML 报告
TILELANG_PASS_DIFF=both     python my_script.py   # 两者都要

# HTML 报告默认输出到 tmp/pass_diff_output/pass_diff_<timestamp>.html
# 可以用另一个环境变量重定向：
TILELANG_PASS_DIFF_OUTPUT=/tmp/mydebug TILELANG_PASS_DIFF=html python my_script.py
```

**关键实现细节**（决定你能不能用对）：

- 这个 hook 是在 `import tilelang` 时安装的。**先 import tilelang 再 os.environ 设变量是无效的**——必须在启动 Python 之前设。
- 它 hook 的是 `tvm.ir.transform.Pass.__call__`，所以不止 TileLang 自己的 pass，**上游 TVM 的 pass 也会被记录**。
- 有性能开销，跑 benchmark 之前记得关。

### 用法 B：Python API（推荐做单元测试）

```python
import tilelang
from tilelang import tvm
from tilelang.utils.pass_diff import pass_diff

steps = pass_diff(
    func,
    [
        ("AnnotateDeviceRegions", tilelang.transform.AnnotateDeviceRegions()),
        ("SplitHostDevice",       tilelang.transform.SplitHostDevice()),
        ("ThreadSync",            tilelang.transform.ThreadSync("shared")),
    ],
    mode="both",
    context=5,
    html_path="tmp/selected_passes.html",
)

# 每一步 step 是一个 dict:
# name / before_script / after_script / diff_lines / insertions / deletions / changed
assert steps[-1]["changed"]
assert "tvm_storage_sync" in steps[-1]["after_script"]
```

这个 API 就是给写「pass 回归测试」用的——用它可以**断言某个 pass 恰好加上了某个 intrinsic 字符串**，跟 11.2 的 codegen 字符串检查是同一套思路，只是在不同层级。

> **事实出处**：`tilelang/utils/pass_diff.py`（35KB），文档 `docs/tools/pass_diff.md`。

---

## 11.4 重量级：Pass Visualizer——结构树 diff

Pass Diff 是**文本行级 diff**。它有个显著缺点：TIR 里一个 tile op（比如 `T.gemm(A, B, C, ...)`）在文本里就是一行，你看不到里面 `M=`, `K=`, `policy=`, `transpose_B=` 等**字段**变化。

Pass Visualizer 补齐这一层。它把 TIR 渲成一棵 **结构树**（`SBlock` 嵌套 + `reads`/`writes`/`alloc_buffers`/`annotations`），tile op 按字段名展开，还把 tile op、同步原语、硬件 intrinsic 各上一种高亮色。

### 命令行

```bash
python -m tilelang.tools.pass_visualizer.viewer \
    tilelang/tools/pass_visualizer/examples/gemm_relu.py \
    --set M=1024 --set N=1024 --set K=1024 \
    --set block_M=128 --set block_N=128 --set block_K=32 \
    --out gemm_relu_passes.html
```

它会生成一个自包含 HTML（左侧 pass 列表 + 右侧结构树 + 上下键切换 pass），外加一份对应的 `.txt` 方便 grep。

### 什么时候用哪个

| 场景 | 选 |
|---|---|
| lower 中间某处 IR 结构不对 | Pass Visualizer（能看到字段级变化） |
| 想大致过一下整条 pipeline 有没有奇怪变化 | Pass Diff（更轻，环境变量一开就行） |
| 写测试断言「某 pass 应该 rewrite 了某个字符串」 | Pass Diff Python API |
| 演示或者 review PR | Pass Visualizer 出 HTML |

> **事实出处**：`tilelang/tools/pass_visualizer/`，文档 `docs/tutorials/debug_tools_for_tilelang.md` 的 "Pass Visualizer" 一节。

---

## 11.5 Analyzer——不用跑就能估性能上限

`tilelang.tools.Analyzer` 是一个**静态 roofline** 估算器：它把你的 TIR 里的 `T.gemm` 计数为 FLOPs，把 `T.copy` 对 global 缓冲的部分算成 bytes，然后除以理论 TFLOPS / 带宽，给你一个"这个 kernel 大概能跑到多少秒"。

真实入口：

```python
from tilelang.carver.arch import CUDA
from tilelang.tools import Analyzer

tir = my_matmul.get_tir(block_M=128, block_N=128, block_K=32)
device = CUDA("cuda")          # 会查 CUDA device 0 的架构和带宽
result = Analyzer.analysis(tir, device)

print(result.total_flops)          # 只统计了 T.gemm
print(result.total_global_bytes)   # 只统计了 T.copy 跨越 global 边界的
print(result.estimated_time)       # max(compute_time, memory_time)
print(result.expected_tflops)      # 内置表，只覆盖 SM 8.0/8.6/8.9
print(result.expected_bandwidth_GBps)
```

**Analyzer 的边界**（这些是 `docs/tools/analyzer.md` 明说的，别踩）：

- 只识别 `T.gemm` 和 `T.copy`。你手写的 elementwise、reduce、atomic **完全不算进去**。
- 只算 `blockIdx.x` 和 `blockIdx.y` 的 grid extent，`blockIdx.z` **不算**。
- 内置 peak-TFLOPS 表是硬编码的，只有 sm80 / sm86 / sm89。Hopper 和 Blackwell 目前会返回 `None`，`estimated_time` 就只是"memory 部分的时间"。
- 不看 occupancy、bank conflict、cache、pipeline overlap——**它就是一个 upper bound sanity check，不是替代 Nsight**。

推荐用法：**写完 kernel 就跑一次 Analyzer**，如果它说"理论最快 100us"，你 benchmark 出来 3ms，那 99% 是编译或调度问题；反过来如果它说"最快 3ms"，那你 benchmark 出来 4ms 就已经离最优很近了，不必再纠结微调。

> **事实出处**：`tilelang/tools/Analyzer.py`（8.87KB），文档 `docs/tools/analyzer.md`。

---

## 11.6 布局可视化：`plot_layout` 与 `TL_LAYOUT_VISUALIZATION_*`

layout / fragment 是 TileLang 最容易出错、又最难仅凭 IR 文本看懂的部分（第 7 章讲过）。所以官方专门做了两条可视化路径：

### 路径 A：手动画一个已知的 layout / fragment

```python
import tilelang.language as T
from tilelang.tools import plot_layout

transpose = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(
    transpose,
    save_directory="./tmp",
    name="transpose_4x4",
    formats="png",              # "pdf" / "png" / "svg" / "all" / 组合
    view="input",               # 或 "output"
)
```

- 对 `T.Layout`，网格里默认写"这个输入坐标映射到哪个输出坐标（展平后）"。
- 对 `T.Fragment`，网格按 **thread ID** 上色，每格标 `T=<线程>`、`L=<该线程的第几个寄存器 slot>`——直接看出哪个 warp 拿哪一块。

需要 `pip install "tilelang[vis]"` 装 Matplotlib。

### 路径 B：让编译器把它推断出来的 fragment 打出来

在 `@tilelang.jit` 里打开 pass config：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
    }
)
def kernel(A, block_M, block_N):
    ...
```

每一个被 LayoutInference pass 推断出来的 2D fragment，都会在编译时输出：

```
C_local inferred layout:
  Shape: [32, 32] -> [8]
  Thread: <thread-index expression>
  Index:  [<local-index expression>]
  Replicate:  1
```

同时在 `./tmp/` 生成 `C_local_layout.svg`。**当你怀疑一个数值 bug 是"数据被切错了片"而不是"算错了"时**，这就是最快的证据。

> **事实出处**：`tilelang/tools/plot_layout.py`（26KB），文档 `docs/tools/layout_visualization.md`。

---

## 11.7 C++ 侧调试：TVM logging 体系

前面 6 招都是 Python / IR 层。当你需要**改 C++ pass**（第 4 章讲过的场景）时，就用 TVM logging 系统。TileLang 沿用了它，没有自己再写一套。

**三档 log**（都在 `include/tvm/runtime/logging.h`）：

```cpp
LOG(INFO)  << "always compiled in";
DLOG(INFO) << "only in Debug build";
VLOG(1)    << "verbose, level-controlled";
```

**四档 check**：

```cpp
CHECK(cond)  << "release 也会检查";
ICHECK(cond) << "内部不变量，release 也保留";
DCHECK(cond) << "仅 Debug build";
```

**运行时打开 DLOG**：

```bash
# 全局所有文件开到 DEBUG(0)：
TVM_LOG_DEBUG=1 python my_script.py

# 只给某个文件开：
TVM_LOG_DEBUG="DEFAULT=-1,transform/inject_pipeline.cc=1" python my_script.py
```

> **⚠️ 一个非常容易踩的坑**：`TVM_LOG_DEBUG` 是**两个东西共用一个名字**：
> - 编译期宏（CMake `-DCMAKE_BUILD_TYPE=Debug` 会自动 define）——决定 `DLOG` 代码有没有被编进 .so。
> - 运行期环境变量——决定哪些 DLOG 会真的打出来。
>
> 只设运行期变量、但 .so 是 Release build 的，什么都不会输出。这在 `docs/tutorials/logging.md` 里作为 note 单独提示过。

> **事实出处**：`docs/tutorials/logging.md`，`src/runtime/logging.cc`。

---

## 11.8 AutoDD——自动最小化 bug 复现

前 7 招假设你已经有一个"最小可复现"。现实里更多情况是：**500 行 model 代码里某处会挂，我不想手工二分**。

AutoDD（automatic delta debugging）就是干这个的：给它一个「跑起来必然失败」的脚本 + 一个失败信息里必然出现的子串，它会不断做「删一段代码 → 跑一遍 → 如果还出同样错就保留，否则回滚」，直到收敛。

```bash
python -m tilelang.autodd examples/autodd/tilelang_buggy.py \
  --err-msg "T.gemm K shape check failed" \
  -o minimized.py
```

它会输出一个（通常几十行的）`minimized.py`，直接可以贴到 issue 或者 PR 里。

**使用前提**（很关键）：

1. 失败必须是**确定性**的。有随机性的话它会误判"这一步删对了"。
2. `--err-msg` 是**大小写敏感的子串匹配**，选一个足够特征、不容易在无关错误里也出现的字串。

> **事实出处**：`tilelang/autodd.py`（48KB），文档 `docs/tools/autodd.md`。

---

## 11.9 一个真实的调试剧本：把 8 招串起来

假设你写的一个 GEMM kernel，编译成功但数值有 5% 的元素对不上参考实现。真实的排查顺序应该是：

```
Step 1  Analyzer.analysis()：先看理论时间 vs 实际时间是否合理
        └── 如果差 100 倍，先不管正确性，估计整体调度就错了

Step 2  register_cuda_postproc_callback：dump 出最终 CUDA 源码
        └── 搜 mma / cp.async / tma，看有没有明显缺失或多余的指令

Step 3  TL_LAYOUT_VISUALIZATION_ENABLE=True：出 C_local 的 fragment 图
        └── 5% 的元素错，很可能就是某个 warp 拿错了 tile

Step 4  T.print 打前 8 个元素、只让 tid==0 打：确认哪一步 K 迭代开始出错

Step 5  TILELANG_PASS_DIFF=html：过一遍完整 pipeline
        └── 找到「上一 pass IR 是对的、这个 pass 后就错了」的那一个 pass

Step 6  pass_visualizer：对着这个可疑 pass 看字段级 diff，定位到具体是哪个 op 参数变了

Step 7  改 C++ pass 后，DLOG(INFO) + TVM_LOG_DEBUG=<file>=1 验证

Step 8  改完写测试：用 pass_diff Python API 断言这个 pass 的输出里
        不再包含 bug 特征字符串（呼应第 6 章 6.9.5 的正确性回归思路）
```

这条链路是**递进的**：每一步都比上一步"更贵、更精细"。绝大部分正确性 bug 在 Step 3~5 之间就抓到了。

---

## 11.10 小结

一句话记住每个工具：

| 工具 | 一句话 |
|---|---|
| `T.print` | GPU 内运行时打印 |
| `register_cuda_postproc_callback` | 拿到最终 CUDA 字符串 |
| Pass Diff | 每个 pass 前后 TIR 文本 diff |
| Pass Visualizer | 结构树、字段级 diff、HTML |
| Analyzer | 静态 roofline sanity check |
| `plot_layout` / `TL_LAYOUT_VISUALIZATION_*` | 可视化 layout/fragment |
| TVM logging (`DLOG`, `TVM_LOG_DEBUG=…`) | C++ pass 内部调试 |
| AutoDD | 自动最小化 bug 复现 |

下一章 [第 12 章](./12_control_flow_dynamic_reduce_atomic.md) 补齐 TileLang 语言层面另一片你在正文里没见过的 API：控制流、动态形状、以及 reduce/scan/atomic 三类原语。
