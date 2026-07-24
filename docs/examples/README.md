# Cookbook 示例代码

> 本目录把 [cookbook 正文](../README.md) 里出现过的所有代码片段抽出来，做成**可直接 `python xxx.py` 运行**的独立文件。每个文件顶部有一段 docstring 指明它来自哪一章的哪一节。

---

## 目录索引

### 第 1 章 · Hello TileLang

| 文件 | 一句话 |
|---|---|
| [ch01_hello/01_matmul_minimal.py](./ch01_hello/01_matmul_minimal.py) | 最小可跑的 fp16 matmul + relu，验证 TileLang 装好了 |
| [ch01_hello/02_look_around_pipeline.py](./ch01_hello/02_look_around_pipeline.py) | dump 编译流水线各阶段的中间产物（TIR / device_mod / CUDA / host） |

### 第 2 章 · TVM/TIR 基础

| 文件 | 一句话 |
|---|---|
| [ch02_tir_basics/01_primfunc_add_one.py](./ch02_tir_basics/01_primfunc_add_one.py) | 手写一个 `add_one` PrimFunc，用玩具 walker 打印 IR 树 |

### 第 3 章 · DSL 练习

| 文件 | 一句话 |
|---|---|
| [ch03_dsl/01_pipeline_annotations.py](./ch03_dsl/01_pipeline_annotations.py) | diff `num_stages=1` vs `3` 的 pipeline 注解 |
| [ch03_dsl/02_shared_vs_shared_dyn.py](./ch03_dsl/02_shared_vs_shared_dyn.py) | diff `shared` vs `shared.dyn` 的 codegen |
| [ch03_dsl/03_elementwise_add.py](./ch03_dsl/03_elementwise_add.py) | 只用 `T.copy` 写一个 elementwise-add |

### 第 4 章 · Pass 系统

| 文件 | 一句话 |
|---|---|
| [ch04_pass/01_try_visitor.py](./ch04_pass/01_try_visitor.py) | 跑一遍官方的 `NestedLoopChecker` visitor pass |
| [ch04_pass/02_count_copies_pass.py](./ch04_pass/02_count_copies_pass.py) | 自己写一个数 `T.copy` 的 Python pass |

### 第 5 章 · Lowering Pipeline

| 文件 | 一句话 |
|---|---|
| [ch05_lowering/01_prologue_output.py](./ch05_lowering/01_prologue_output.py) | 单独调 `CUDAPassPipelineBodyPrologue`，看段 A 出口 IR |
| [ch05_lowering/02_step_by_step_before_after.py](./ch05_lowering/02_step_by_step_before_after.py) | 手动串 pass 到 `InjectSoftwarePipeline` 前后 diff |

### 第 6 章 · 软件流水 + Warp Specialization

| 文件 | 一句话 |
|---|---|
| [ch06_pipeline_ws/01_ws_on_off_diff.py](./ch06_pipeline_ws/01_ws_on_off_diff.py) | WS on/off 两种 pass config 下的 device_mod 对比 |
| [ch06_pipeline_ws/02_wsoff_vs_ws_regression.py](./ch06_pipeline_ws/02_wsoff_vs_ws_regression.py) | K-trip 不对齐场景：WS-off vs WS-on 硬签名 + bit-exact 回归测试 |

### 第 7 章 · Layout / Fragment

| 文件 | 一句话 |
|---|---|
| [ch07_layout/01_layout_basics.py](./ch07_layout/01_layout_basics.py) | 手写转置 / row-major / XOR-swizzle 三种 Layout |
| [ch07_layout/02_annotate_layout.py](./ch07_layout/02_annotate_layout.py) | 用 `T.annotate_layout` + `make_swizzled_layout` 手动挂 layout |
| [ch07_layout/03_before_after_layout_inference.py](./ch07_layout/03_before_after_layout_inference.py) | `LayoutInference` 前后 IR 对比 |

### 第 8 章 · Codegen

| 文件 | 一句话 |
|---|---|
| [ch08_codegen/01_dump_kernel_source.py](./ch08_codegen/01_dump_kernel_source.py) | dump 生成的 CUDA C++ 源码 |

### 第 9 章 · JIT & Cache

| 文件 | 一句话 |
|---|---|
| [ch09_jit/01_lazy_jit_pipeline.py](./ch09_jit/01_lazy_jit_pipeline.py) | 完整 JIT 调用链，展示第二次调用命中内存缓存 |
| [ch09_jit/02_dump_sources_and_ir.py](./ch09_jit/02_dump_sources_and_ir.py) | 一次性 dump 源码 / PTX / SASS / 每一步 IR |

### 第 10 章 · 写 Pass 与贡献

| 文件 | 一句话 |
|---|---|
| [ch10_contribute/01_tag_buffers_pass.py](./ch10_contribute/01_tag_buffers_pass.py) | 玩具 Python pass：在 PrimFunc body 外包一层 `AttrStmt` 标记 |
| [ch10_contribute/02_pytest_pass_template.py](./ch10_contribute/02_pytest_pass_template.py) | 给自定义 pass 写 pytest 单元测试的模板 |

### 第 11 章 · 调试与可视化

| 文件 | 一句话 |
|---|---|
| [ch11_debugging/01_t_print_hello.py](./ch11_debugging/01_t_print_hello.py) | GPU 内 `T.print` 打印最小 demo |
| [ch11_debugging/02_postproc_callback.py](./ch11_debugging/02_postproc_callback.py) | `register_cuda_postproc_callback` 拦截 codegen 源码 |
| [ch11_debugging/03_pass_diff_api.py](./ch11_debugging/03_pass_diff_api.py) | `pass_diff` Python API 生成 HTML 报告 |
| [ch11_debugging/04_analyzer_and_plot_layout.py](./ch11_debugging/04_analyzer_and_plot_layout.py) | `Analyzer` 静态 roofline + `plot_layout` 布局可视化 |

### 第 12 章 · 控制流 / 动态形状 / Reduce·Scan·Atomic

| 文件 | 一句话 |
|---|---|
| [ch12_control_flow/01_dynamic_shape_matmul.py](./ch12_control_flow/01_dynamic_shape_matmul.py) | `T.dynamic("m")` 动态形状 GEMM，多种形状复用同一份 kernel |
| [ch12_control_flow/02_reduce_sum_vector.py](./ch12_control_flow/02_reduce_sum_vector.py) | `T.reduce_sum` 单 block 向量求和 |
| [ch12_control_flow/03_cumsum_prefix.py](./ch12_control_flow/03_cumsum_prefix.py) | `T.cumsum` in-place 前缀和 |
| [ch12_control_flow/04_atomic_histogram.py](./ch12_control_flow/04_atomic_histogram.py) | `T.atomic_add` 多 CTA 更新同一份 histogram |

### 第 13 章 · Cluster / TMA / Hopper

| 文件 | 一句话 |
|---|---|
| [ch13_hopper/01_cluster_sm_to_sm.py](./ch13_hopper/01_cluster_sm_to_sm.py) | Cluster 内两个 rank 用 shared + mbarrier 互推数据 |
| [ch13_hopper/02_warp_specialize_manual.py](./ch13_hopper/02_warp_specialize_manual.py) | 手写 `T.ws(0) / T.ws(1)` producer-consumer 骨架 |

### 第 14 章 · Quantization

| 文件 | 一句话 |
|---|---|
| [ch14_quantize/01_w4a16_dequant_gemm.py](./ch14_quantize/01_w4a16_dequant_gemm.py) | 完整可跑的 W4A16 dequant GEMM 骨架 |
| [ch14_quantize/02_fp8_dtype_selection.py](./ch14_quantize/02_fp8_dtype_selection.py) | `determine_fp8_type` 跨硬件选对 FP8 dtype |

### 附录 F · Eager JIT + CuTeDSL

| 文件 | 一句话 |
|---|---|
| [appF_eager_cutedsl/01_lazy_vs_eager.py](./appF_eager_cutedsl/01_lazy_vs_eager.py) | Lazy JIT vs Eager JIT 两种写法对比 |
| [appF_eager_cutedsl/02_cutedsl_backend.py](./appF_eager_cutedsl/02_cutedsl_backend.py) | 同一 kernel 分别用默认后端和 CuTeDSL 后端跑 |

---

## 运行前提

1. **已经装好 TileLang**：`pip install tilelang`（或从源码 build）。
2. **有一张 CUDA GPU**：多数示例默认 target=`cuda`，如果你在 CPU-only 机器上跑，多数示例会在 `tilelang.compile` 阶段直接失败。
3. **PyTorch**：所有需要跑数值的示例都用 `torch` 拿输入 / 参考实现。
4. **Hopper 相关示例（第 13 章）**：需要 SM90+（H100 / RTX 4090-Hopper 变种以上）。
5. **量化示例（第 14 章）**：至少 SM70+，MXFP4 需要 Hopper 或 gfx950。
6. **可视化示例（第 11 章 `plot_layout`）**：需要 `pip install "tilelang[vis]"`。
7. **附录 F CuTeDSL 后端**：需要额外 `pip install cutlass`。

---

## 快速上手

```bash
cd docs/examples/ch01_hello
python 01_matmul_minimal.py
```

如果它输出：
```
[hello] max abs diff = 0.00xxx   PASS
```
就说明你的 TileLang 装好了。

---

## 命名约定

- 每个文件是**自包含**的：单文件运行、单文件解释。
- 文件名前缀 `NN_` 是章节内的顺序号；对应正文里的小节序号。
- 每份代码顶部都有 docstring：**来源、目的、跑法、期望输出、坑**。
- **凡是 tilelang 真实源码里也有的示例**（比如 `examples/dynamic_shape/example_dynamic.py`），docstring 里会注明 `upstream:` 字段，避免重复维护同样的代码。

---

## 已知运行问题

| 问题 | 说明 |
|---|---|
| `AttributeError: T.something` | TileLang 版本差异——某些 API 在旧版本叫不同名字。文件里会尽量注明最低版本。 |
| `nvcc fatal error: unsupported gpu architecture 'compute_90'` | 你的 CUDA toolkit 太旧，升级到 12.3+ 才能编 Hopper。 |
| `Import error: cutlass` | 附录 F 的 CuTeDSL 示例需要额外安装 `cutlass` Python 包。 |
| 数值 rtol 微小差异 | fp16 / bf16 累加顺序不同带来的 ULP 级差异，写脚本时都开了 `atol=1e-2` 左右。 |

---

## 反馈

发现哪份代码跑不通、或者跟正文对不上，欢迎在对应文件顶部加：

```python
# TODO(reader): <问题描述>
```

方便下一次迭代收敛。
