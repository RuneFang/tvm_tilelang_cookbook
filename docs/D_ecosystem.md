# 附录 D · 生态延伸：Carver 与 AutoTune、多 GPU 现状

> **这个附录写给谁**：已经读完正文 10 章、知道 TileLang 是怎么把 Python DSL 编译成 CUDA
> 的读者。现在想问两个"外围但常问"的问题：
>
> 1. **`block_M / block_N / num_stages` 这些配置我怎么才能不用瞎猜？**
> 2. **TileLang 能写多 GPU 的 kernel 吗？**
>
> 本附录用最短的篇幅，给出**真实可运行**的答案 + 权威文档指针，避免你被过时博客误导。
>
> **本附录会引用到的真实源码 / 官方文档**：
> - [`tilelang/autotuner/tuner.py`](../../tilelang/autotuner/tuner.py) —— `autotune` / `AutoTuner`
> - [`tilelang/autotuner/capture.py`](../../tilelang/autotuner/capture.py) —— `set_autotune_inputs`
> - [`tilelang/carver/`](../../tilelang/carver/) —— tile shape 候选生成
> - [`tilelang/carver/README.md`](../../tilelang/carver/README.md) —— Carver 的官方定位
> - [`docs/programming_guides/autotuning.md`](../../docs/programming_guides/autotuning.md) —— autotune 官方指南
> - [`examples/gemm/example_gemm_autotune.py`](../../examples/gemm/example_gemm_autotune.py) —— 官方示例
> - [`testing/python/autotune/test_tilelang_autotune_scalar_inputs.py`](../../testing/python/autotune/test_tilelang_autotune_scalar_inputs.py) —— 最短能跑的示例

---

## D.1 TileLangCarver + AutoTune：让配置自己搜出来

### D.1.1 到底是"一个东西"还是"两个东西"？

先澄清一个非常容易混淆的点——**Carver 和 AutoTuner 是两个独立子系统**：

| 子系统 | 目录 | 干的活 | 是否直接给用户用 |
|---|---|---|---|
| **Carver** | [`tilelang/carver/`](../../tilelang/carver/) | 根据"操作类型 + 硬件规格"**推荐**一批 tile 候选（block/warp/rstep/use_tc…） | ✅ 也可以独立用，见 [`tilelang/carver/README.md`](../../tilelang/carver/README.md) |
| **AutoTuner** | [`tilelang/autotuner/`](../../tilelang/autotuner/) | 拿到一批配置后**并行编译 + benchmark + 挑最优 + 落盘缓存** | ✅ 通过 `@tilelang.autotune` 装饰器 |

它们**可以配合，也可以分开用**：

- **只用 AutoTuner**：你手写一个 `configs` 列表，让 tuner 一个个试——**这是 99% 用户实际的用法**，官方 examples 里几乎全是这种。
- **只用 Carver**：你在写别的 compiler，想借 Carver 的启发式生成 tile hint。
- **Carver → AutoTuner**：让 Carver 生成候选、AutoTuner 挑最好——[`examples/gemm/example_gemm_autotune.py`](../../examples/gemm/example_gemm_autotune.py) 里有 `with_roller` 开关做这件事。

> 🧠 **一句话记忆**：
> - **Carver** ≈ 一位"看过一眼硬件规格就能猜出好 tile 尺寸"的老工程师，产出 **hints**（推荐列表）
> - **AutoTuner** ≈ 一台自动化跑机器，把每个 hint 编成 kernel、实测、留下最快那个
>
> 只要你能自己列出候选，**完全不用 Carver 也能享受 autotune**。

### D.1.2 最短能跑的 AutoTune 例子（改自官方测试）

以下代码几乎原封不动来自 [`testing/python/autotune/test_tilelang_autotune_scalar_inputs.py`](../../testing/python/autotune/test_tilelang_autotune_scalar_inputs.py)（可直接 `python xxx.py` 跑）：

```python
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


# ⬇️ 关键 1：autotune 装饰器必须"套在 @tilelang.jit 上面"
@tilelang.autotune(
    configs=[{"threads": 128}, {"threads": 256}],   # 待搜索的配置列表
    warmup=1, rep=1, timeout=60,
)
@tilelang.jit
def add_scalar(N: int = 4096, BLOCK_N: int = 512, threads: int = 128):

    @T.prim_func
    def kernel(A: T.Tensor((N,), T.float32), s: T.float32):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=threads) as pid_n:
            A_local = T.alloc_fragment((BLOCK_N,), T.float32)
            T.copy(A[pid_n * BLOCK_N], A_local)
            for i in T.Parallel(BLOCK_N):
                A_local[i] += s
            T.copy(A_local, A[pid_n * BLOCK_N])

    return kernel


# ⬇️ 关键 2：调用前用 set_autotune_inputs 提供一份 "tuning 用" 的输入张量
tune_a = torch.randn((4096,), device="cuda", dtype=torch.float32)
tune_s = 0.1
with set_autotune_inputs(tune_a, tune_s):
    kernel = add_scalar()           # 触发：编译 2 份候选 → 各自 benchmark → 挑最快

# 之后 kernel(...) 是那个最优候选，直接跑就行
a = torch.randn((4096,), device="cuda", dtype=torch.float32)
before = a.clone()
kernel(a, tune_s)                   # 用户的真实推理输入
torch.testing.assert_close(a, before + tune_s, rtol=1e-4, atol=1e-4)
```

> 📌 **常见坑**（这份测试文件的原作者就是被这个坑到才写了它，见文件里 `test_autotune_scalar_inputs_require_explicit_supply`）：
> **如果 kernel 参数里有非 Tensor 的标量**（例子里的 `s: T.float32`），你**必须**用 `set_autotune_inputs`
> 显式提供输入——否则 AutoTuner 的默认 tensor supplier 不知道该给这个标量传什么，会直接
> `raise ValueError("...set_autotune_inputs...")`。这是 [`tilelang/autotuner/tuner.py`](../../tilelang/autotuner/tuner.py) 里
> 硬编码的检查，不是 bug。

### D.1.3 `@tilelang.autotune` 的完整参数

来自 [`tilelang/autotuner/tuner.py:1406`](../../tilelang/autotuner/tuner.py)（真实签名，行号截至写作时）：

```python
@tilelang.autotune(
    configs,                          # dict 列表 或 (args...) -> list[dict] 的可调用对象
    warmup=25, rep=100, timeout=100,  # 单个配置的 benchmark 参数（各 100 次求平均、100 秒超时）
    supply_type=TensorSupplyType.Auto,# 默认 tensor 生成器（形状必须静态）
    ref_prog=None,                    # 参考程序：给它一份输入让它算真值，用于校验精度
    supply_prog=None,                 # 自定义 tensor 供应函数
    rtol=1e-2, atol=1e-2,             # 精度容忍度
    max_mismatched_ratio=0.01,        # 允许多少比例的元素不匹配
    skip_check=False,                 # 关掉精度校验（求快时用，慎用）
    manual_check_prog=None,           # 完全自定义的校验回调
    cache_input_tensors=False,        # 是否把 tuning 输入落盘用于复现
    do_not_specialize=None,           # 哪些参数不参与"特化"
)
```

再往下的 **编程式接口** `AutoTuner.from_kernel(...)`（[`tuner.py:277`](../../tilelang/autotuner/tuner.py)）
以及**输入供应 / 缓存 / 环境变量**都由官方文档 [`docs/programming_guides/autotuning.md`](../../docs/programming_guides/autotuning.md)
详细覆盖，本书不再重复。真正需要的时候把那份文档从头到尾读一次即可（约 300 行、11 KB）。

### D.1.4 缓存：搜完一次以后再也不用等

AutoTuner 在两个层次做缓存：

| 层次 | 位置 | 由谁生成 |
|---|---|---|
| 进程内 | `AutoTuner._memory_cache`（[`tuner.py:245` 附近](../../tilelang/autotuner/tuner.py)）| 一次进程内多次调用同一 `matmul(M, N, K)` 时命中 |
| 磁盘 | `$TILELANG_CACHE_DIR/autotuner/<key>/`（`_get_cache_dir()`，[`tuner.py:270`](../../tilelang/autotuner/tuner.py)）| 跨进程/跨机器复用，落盘 `best_config.json` + `latency.json` + `device_kernel.cu` + `kernel_lib.so` 等 |

缓存 key 由 (TileLang 版本 + 函数源码 + 闭包自由变量 + configs + compile args + profile args) 一起哈希——
只要**任一位**变了，key 就变，会重新搜。

要强制不用磁盘缓存（例如你在做 CI 里 ablation）：

```bash
export TILELANG_AUTO_TUNING_DISABLE_CACHE=1
```

要**彻底**清掉 kernel + autotune 全部缓存：

```bash
rm -rf ~/.tilelang/cache          # 默认 TILELANG_CACHE_DIR 路径
```

### D.1.5 Carver 单独用：给别的编译器/别的地方生成 tile hint

Carver 本身**不依赖 AutoTuner**——它是一个独立的"看着硬件规格挑 tile"的启发式库，
甚至可以给 Triton / TVM 生成 hint（这是它自己 README 明确写的用途）。

最短能跑的 Carver 例子（来自 [`tilelang/carver/README.md`](../../tilelang/carver/README.md)）：

```python
from tilelang import carver
from tilelang.carver.arch import CUDA

arch = CUDA("nvidia/geforce-rtx-4090")           # 描述硬件
tpl = carver.MatmulTemplate(                     # 描述算子
    M=1024, N=1024, K=1024,
    in_dtype="float16", accum_dtype="float16", out_dtype="float16",
).with_arch(arch)

hints = tpl.recommend_hints(topk=20)             # 生成 20 个候选
for h in hints:
    print(h)
# 输出（截取）：
# {'block': [32, 64], 'warp': [16, 32], 'rstep': [128], 'use_tc': True, ...}
# {'block': [64, 32], 'warp': [32, 16], 'rstep': [128], 'use_tc': True, ...}
# ...
```

内置模板（见 [`tilelang/carver/template/`](../../tilelang/carver/template/)）：
`MatmulTemplate`、`GEMVTemplate`、`ElementwiseTemplate`、`GeneralReductionTemplate`、`FlashAttentionTemplate`。

**Carver + AutoTuner 联动**的写法在 [`examples/gemm/example_gemm_autotune.py`](../../examples/gemm/example_gemm_autotune.py) 里，
核心逻辑就是把 `carver.MatmulTemplate(...).recommend_hints(topk=N)` 的输出**翻译成 autotune 的 configs 列表**——
一个是产 hint，一个是挑最快，各司其职。

### D.1.6 什么时候用 / 什么时候别用

**用 autotune 的信号**：
- Kernel 的性能敏感于 tile 尺寸 / stage 数 / thread 数
- 你有多个 shape / dtype 组合要覆盖，一个个手调工作量太大
- 你能给出"标准输入 + 参考实现"用于精度校验

**别用（或先别用）的信号**：
- Kernel 还没写对——先用固定 config 跑通、`ref_prog` 校验对之后再上 autotune
- Kernel 参数里有 `T.dyn` 动态形状——`supply_type=Auto` 不能工作，必须走 `set_autotune_inputs` 或 `supply_prog`
- 搜索空间 > 10⁴ 量级——编译加 benchmark 都会非常慢，先用 Carver 或人工过滤缩小 configs

---

## D.2 Distributed / Multi-GPU：现状与务实做法

### D.2.1 一句话现状：**TileLang 目前不提供多 GPU 通信原语**

在写这本书的时候（对应仓库 `HEAD`），我 grep 了整个 `tilelang/`：

- **零** `nccl` 引用（除 Apache License 样板注释里的 "distributed with this work"）
- **零** `nvshmem` 引用
- **零** `all_reduce` / `all_gather` / `broadcast` 之类的 collective 原语

也就是说，**你不能在 `@T.prim_func` 里调用 `T.all_reduce(...)`**——它不存在。
TileLang 当前的抽象层是 **单 GPU、单 kernel** 的：你写的 `T.Kernel(...)` 只描述一个 CTA grid，
跨设备的通信不在它的表达能力里。

任何声称 "TileLang 内置 NCCL/NVSHMEM" 的博客都可能是过时或误导性的——请以本附录列出的
真实文件路径 grep 结果为准。

### D.2.2 那 tuner.py 里的 `benchmark_multi_gpu=True` 是什么？

这是一个**特别容易被误解**的参数。看 [`tilelang/autotuner/tuner.py:933`](../../tilelang/autotuner/tuner.py) 的真实用法：

```python
def run(
    self,
    ...,
    benchmark_multi_gpu: bool = False,
    benchmark_devices: list[int] | None = None,
    ...
):
    """
    ...
    benchmark_multi_gpu:
        Whether to benchmark configurations across multiple CUDA GPUs.
    benchmark_devices:
        CUDA device ordinals used for benchmark workers when benchmark_multi_gpu=True.
    """
```

**它的含义是"把不同的 tuning 候选分派到不同 GPU 上并行 benchmark"**——纯粹的搜索加速手段，
和"生成的 kernel 是否跨 GPU 运行"**完全无关**。

- 你有 configs = [c1, c2, c3, c4]、机器上有 4 张 GPU
- 打开 `benchmark_multi_gpu=True, benchmark_devices=[0, 1, 2, 3]`
- Tuner 把 4 个候选 kernel 分别在 4 张卡上 benchmark，搜索时间约 ÷ 4
- 最后挑出的**那一个** kernel 仍然是**单 GPU** kernel

不要把这个当成"多 GPU kernel 支持"来用。

### D.2.3 那我真的要跑多 GPU 怎么办？

**当前生态里的务实做法（不需要修改 TileLang）**：

```
┌────────────────────────────────────────────────────────┐
│  Python 层（PyTorch）                                    │
│                                                         │
│   torch.distributed.init_process_group("nccl")          │
│   torch.distributed.all_reduce(x)   ← 通信在这里       │
│                                                         │
│   my_tl_kernel(x, w)  ← TileLang 编出来的单卡 kernel   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

也就是：**通信在 Python 侧（PyTorch NCCL / `torch.distributed`）做，计算 kernel 用 TileLang**。
TileLang JIT 出来的 kernel 对 PyTorch 而言就是个普通算子，塞进任何 rank 的 forward pass
都可以正常调用。这套组合已经能覆盖张量并行 / 流水线并行 / 数据并行大部分场景。

具体做法：

1. 用 `torchrun --nproc_per_node=8 script.py` 起 8 rank
2. 每个 rank 各自 `@tilelang.jit` 出自己的 kernel（会命中同一个磁盘缓存 key，只有 rank 0 会真编译，
   其它 rank 直接读缓存——见 [`tilelang/cache/kernel_cache.py`](../../tilelang/cache/kernel_cache.py) 第 9 章讲过）
3. Forward/backward 之间在 Python 侧调 `dist.all_reduce` / `dist.all_gather`

### D.2.4 什么时候这个方案不够？

如果你需要的**不是** "rank 之间数据同步"，而是**在 kernel 内部**做设备间通信
（例如 fused all-reduce-matmul、DeepEP 那种细粒度 NVLink 通信、Symmetric Memory），
那 **TileLang 当前抽象层做不了**。可选路径：

- 上游 PR：给 TileLang 加 `T.symmetric_alloc(...)` / `T.nvshmem_put(...)` 之类的原语
  （社区目前有需求，但截至写作时未合入 main）
- 直接手写 CUDA + NVSHMEM/NVLink SM，绕开 TileLang
- 用 Triton + `triton_dist`（有实验性支持），或 CUDA 的 `torch.distributed.symmetric_memory`

对绝大部分用户来说，**D.2.3 描述的"PyTorch NCCL + TileLang 单卡 kernel"组合已经完全够用**——
只有当你要做的就是"在 kernel 内部融合通信"这个具体研究方向时，才需要考虑上面几个替代路径。

### D.2.5 附：写作时的 grep 证据

如果后续版本这条现状变了（例如社区加了 NCCL 原语），可以用同样的方法重新核实：

```bash
# 在仓库根跑
rg -n --hidden -g '!*.md' -g '!LICENSE*' -w -e 'nccl|nvshmem|all_reduce|all_gather' tilelang/
# 写作时：全为 Apache License 注释里的 "distributed with this work"，无一命中真实 API
```

---

## D.3 三件事带走

1. **配置搜索**：日常直接用 `@tilelang.autotune(configs=[...])` + `set_autotune_inputs(...)` 就够了；
   想更省事让 Carver 帮你生成 candidates；权威文档在 [`docs/programming_guides/autotuning.md`](../../docs/programming_guides/autotuning.md)。
2. **`benchmark_multi_gpu` ≠ 多 GPU kernel**——它只是把 tuning 分派到多卡加速搜索。
3. **多卡训练**当前的推荐做法是 PyTorch NCCL 做通信 + TileLang 做单卡计算 kernel；
   如需 kernel 内融合通信，目前只能绕开 TileLang。

---

回到 [README.md](./README.md) 目录。
