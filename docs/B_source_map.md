# 附录 B · 本仓库源码地图

> 目的：调 bug 或者读某一章时，"我这个问题应该改哪个文件 / 相关正文在第几章"的快速定位表。
>
> 每个子表的最后一列 **章** 是全书正文对该文件的重点讲解位置——不是"仅在这一章出现过"，而是"想看深度解释先翻这里"。

## B.1 顶层目录

| 目录 | 职责 |
|---|---|
| `3rdparty/tvm/` | TVM 上游代码（作为 submodule），提供 TIR / Pass 基础设施 |
| `3rdparty/cutlass/` | NVIDIA CuTe / CUTLASS，用于 gemm 底层 |
| `3rdparty/composable_kernel/` | AMD 版 gemm 底层 |
| `src/` | TileLang 自己的 C++ 侧（IR op / pass / codegen） |
| `tilelang/` | TileLang 自己的 Python 侧（DSL / JIT / cache） |
| `examples/` | 覆盖所有主流场景的示例 |
| `benchmark/` | 性能基准 |
| `testing/` | 单元测试 |
| `docs/` | 官方 Sphinx 文档 |
| `tmp/` | 你自己的临时资料（含本 cookbook） |

## B.2 `src/` 详解

| 子目录 | 职责 |
|---|---|
| `src/op/` | Tile-level op 的通用定义（copy / gemm / reduce / scan / fill / …） |
| `src/transform/` | 与后端无关的 pass（大多数） |
| `src/layout/` | Layout / Swizzle / Fragment / CuTe 兼容层 |
| `src/cuda/` | NVIDIA CUDA 后端 |
| `src/rocm/` | AMD ROCm 后端 |
| `src/cpu/` | CPU 后端 |
| `src/metal/` | Apple Metal 后端 |
| `src/webgpu/` | WebGPU 后端 |
| `src/backend/common/` | 各后端共享的辅助 |
| `src/tl_templates/` | 生成 CUDA / HIP 代码时会 include 的 C++ 模板头文件 |
| `src/runtime/` | 运行时辅助（错误处理、日志） |

### B.2.1 `src/op/` 的关键文件

TileLang DSL 里那些 `T.copy` / `T.gemm` / `T.reduce` 的 C++ 内部 op 定义：

| 文件 | 作用 | 章 |
|---|---|---|
| `builtin.h` / `builtin.cc` | 所有 TileLang 内部 intrinsic 的登记表（`mvb_stage_index` / `mbarrier_*` / `create_barriers`） | 6 |
| `copy.h` / `copy.cc` | tile-level copy 的属性、lowering、layout 推断 | 3 |
| `gemm.h` / `gemm.cc` | tile-level gemm（sm_80 / Hopper WGMMA / Blackwell UMMA 的分派） | 3 |
| `gemm_sp.cc` | 结构化稀疏 gemm | 3 |
| `reduce.h` / `reduce.cc` | tile-level reduce（sum / max / min / and / or） | 3 |
| `parallel.h` / `parallel.cc` | `T.Parallel` 循环的 layout 推断入口 | 7 |
| `region.h` / `region.cc` | tile 上的"区域"抽象 | 7 |
| `fill.cc` / `finalize_reducer.cc` | 填充、reducer 收尾 | 3 |

### B.2.2 `src/transform/` 常见 pass（后端无关）

| 文件 | 作用 | 章 |
|---|---|---|
| `frontend_legalize.cc` | 前端 legalize | 5 |
| `layout_inference.cc` | 层布局推断 | 5/7 |
| `lower_tile_op.cc` | 把 tile-level op 展开成 TIR | 5 |
| `pipeline_planning.cc` | 计划软件流水（决定哪些语句进哪个 stage） | 5/6 |
| `inject_pipeline.cc` | 真正把 pipeline 展开成显式双缓冲 | 5/6 |
| `storage_rewrite.cc` | 存储重写（共享内存分配等） | 5 |
| `thread_storage_sync.cc` | 插入 `__syncthreads` 等同步 | 5 |
| `merge_shared_memory_allocations.cc` | 合并 shared memory 分配以省空间 | 5 |
| `flatten_buffer.cc` | 多维 buffer 展平成一维 | 5 |
| `loop_partition.cc` | 循环切分 | 5 |
| `loop_vectorize.cc` / `vectorize_loop.cc` | 向量化 | 5 |
| `unroll_loop.cc` | 展开循环 | 5 |
| `simplify.cc` | 常量化简 | 5 |
| `split_host_device.cc` | 把 IRModule 拆成 host 部分和 device 部分 | 5/8 |
| `make_packed_api.cc` | 生成打包 API（Python 侧 ↔ C++ 侧调用桥） | 5/9 |

### B.2.3 `src/cuda/transform/` CUDA 专属 pass

| 文件 | 作用 | 章 |
|---|---|---|
| `multi_version_buffer_rewriter.cc` | 生成多版本 buffer（配合 pipeline） | 6 |
| `producer_consumer_ws.cc` | Warp specialization（producer/consumer warp 分工） | 6 |
| `annotate_warp_group_reg_alloc.cc` | 给 warp group 分配寄存器预算 | 6 |
| `fuse_mbarrier_arrive_expect_tx.cc` | 融合 mbarrier arrive + expect_tx | 6 |
| `inject_fence_proxy.cc` | 插入 fence proxy | 6 |
| `inject_tcgen05_fence.cc` | SM100 的 tcgen05 fence | 6 |
| `lower_hopper_intrin.cc` | Hopper 特有 intrinsic 展开 | 6 |
| `lower_blackwell_2sm.cc` | Blackwell 2-SM cluster | 6 |
| `lower_shared_barrier.cc` | shared memory barrier 展开 | 6 |
| `lower_shared_tmem.cc` | shared / tensor memory | 6 |
| `lower_ldg_stg.cc` | 显式 ldg/stg 指令 | 5 |
| `lower_l2_persistent_annotation.cc` | L2 persistent 访问注解 | 5 |
| `lower_pdl.cc` | Programmatic Dependent Launch | 8 |
| `ptx_async_copy_injector.cc` | 注入 `cp.async` PTX | 6 |
| `persist_threadblock.cc` | Persistent kernel（外层 tile 循环） | 6 |

### B.2.4 `src/cuda/codegen/`

| 文件 | 作用 | 章 |
|---|---|---|
| `codegen_cuda.cc` / `.h` | 主 codegen（TIR → CUDA C++ 源码），继承 `CodeGenC` | 8 |
| `codegen_cutedsl.cc` | 生成 CuTe DSL 代码 | — |
| `codegen_py.cc` | 生成 Python 侧代码（工具用途） | — |
| `ptx.cc` / `.h` | PTX 层封装 | 8 |
| `intrin_rule_cuda.cc` | intrinsic 打印规则（`__expf` 等如何变字符串） | 8 |
| `rt_mod_cuda.cc` | 运行时模块入口 + `BuildTileLangCUDA` + `ExtractFuncInfo` | 8 |

### B.2.5 `src/layout/`

| 文件 | 作用 | 章 |
|---|---|---|
| `layout.h` / `layout.cc` | `Layout` / `Fragment` 的 C++ 定义 | 7 |
| `swizzle.h` / `swizzle.cc` | Swizzle 模式定义（避免 bank conflict） | 7 |
| `gemm_layouts.cc` | Gemm 常见 A/B/C layout 的预置 | 7 |
| `layout_helpers.cc` | 各种维度重排小工具 | 7 |
| `tcgen05_layouts.cc` | Blackwell tcgen05 相关 layout | 7 |
| `cute_layout.cc` | 与 CuTe layout 的互转 | 7 |

## B.3 `tilelang/` 详解

| 子目录 | 职责 | 章 |
|---|---|---|
| `tilelang/language/` | DSL 前端（`T.Kernel`, `T.copy`, `T.gemm`, …） | 3 |
| `tilelang/engine/` | Lowering 主入口 `lower.py`（编译回调、host/device 拆分编排） | 4/5 |
| `tilelang/backend/pass_pipeline/` | `PassPipeline` 容器与 `resolve_pipeline`（按 target 分派 pass 顺序） | 4/5 |
| `tilelang/backend/` | 后端 dispatch（选 CUDA / ROCm / Metal / CPU / WebGPU），含 `device_codegen.py` / `host_codegen.py` | 5/8 |
| `tilelang/transform/` | Python 侧 pass 接口 | 4 |
| `tilelang/jit/` | JIT 装饰器、kernel 包装、adapter | 9 |
| `tilelang/cache/` | Kernel & cubin cache（`kernel_cache.py` / `cuda_binary_cache.py`） | 9 |
| `tilelang/cuda/` | CUDA 后端 Python 侧入口（`pipeline.py` / `codegen.py` / `execution_backend.py`） | 5/8/9 |
| `tilelang/layout/` | Python 侧 layout API | 7 |
| `tilelang/carver/` | Tile-shape hint 生成（可独立用，也可喂给 autotuner） | D |
| `tilelang/autotuner/` | 自动 tune：并行编译 + benchmark + 精度校验 + 缓存 | D |
| `tilelang/tools/` | 分析工具（`plot_layout`、`pass_visualizer`） | 10 |
| `tilelang/utils/` | 通用工具（`pass_diff` 等） | 10 |
| `tilelang/testing/` | 测试辅助 | 10 |
| `tilelang/contrib/` | 编译器包装（`nvcc.py`, `nvrtc.py`, `hipcc.py`） | 8/9 |
| `tilelang/profiler/` | 性能分析 | 10 |
| `tilelang/quantize/` | 量化相关 | — |
| `tilelang/tileop/` | Tile op 的 Python 定义（Gemm 等） | 3 |
| `tilelang/intrinsics/` | intrinsic 定义 | 3 |
| `tilelang/analysis/` | IR 分析工具（ast printer 等） | 10 |

### B.3.1 高频文件速查

调 bug 时 90% 都会跳到这几个 Python 文件：

| 文件 | 作用 | 章 |
|---|---|---|
| `tilelang/engine/lower.py` | Lowering 主入口 + `tilelang_callback_cuda_compile` | 5/8 |
| `tilelang/backend/pass_pipeline/pipeline.py` | `PassPipeline` 容器 + `resolve_pipeline` | 4 |
| `tilelang/cuda/pipeline.py` | CUDA 后端的 `CUDAPassPipelineBody`（约 60 个 pass 的顺序表） | 5 |
| `tilelang/cuda/codegen.py` | CUDA codegen 注册项（`target.build.tilelang_cuda`） | 8 |
| `tilelang/cuda/execution_backend.py` | 选 NVCC 还是 NVRTC | 8/9 |
| `tilelang/backend/device_codegen.py` | 各后端 codegen 注册中心 | 8 |
| `tilelang/backend/host_codegen.py` | Host stub 生成 | 8/9 |
| `tilelang/jit/kernel.py` | `JITKernel` 类（`kernel(A, B)` 的实现） | 9 |
| `tilelang/cache/kernel_cache.py` | Kernel 缓存 key 的构造 | 9 |
| `tilelang/transform/pass_config.py` | `PassConfigKey`（`tl.*` pass 配置项的键名枚举） | 4 |
| `tilelang/env.py` | 所有 `TILELANG_*` 环境变量与缓存开关 | 4/9/10 |

## B.4 测试目录

| 目录 | 内容 | 章 |
|---|---|---|
| `testing/python/transform/` | 单个 pass 的正确性测试 | 6/10 |
| `testing/python/kernel/` | 端到端 kernel 测试 | 10 |
| `testing/python/language/` | DSL API 测试 | 3 |
| `testing/python/tilelang/` | JIT / cache / 编译流程测试 | 9 |

（若你在源码里遇到本表未列出的关键文件，欢迎在此追加。）
