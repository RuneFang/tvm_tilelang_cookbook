# TVM / TileLang Cookbook —— 从零到能改 Pass

<p align="center">
  <a href="https://github.com/RuneFang/tvm_tilelang_cookbook/stargazers">
    <img src="https://img.shields.io/github/stars/RuneFang/tvm_tilelang_cookbook?style=flat-square&logo=github&color=yellow" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/RuneFang/tvm_tilelang_cookbook/network/members">
    <img src="https://img.shields.io/github/forks/RuneFang/tvm_tilelang_cookbook?style=flat-square&logo=github&color=blue" alt="GitHub forks"/>
  </a>
  <a href="https://github.com/RuneFang/tvm_tilelang_cookbook/issues">
    <img src="https://img.shields.io/github/issues/RuneFang/tvm_tilelang_cookbook?style=flat-square&logo=github" alt="GitHub issues"/>
  </a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=RuneFang.tvm_tilelang_cookbook" alt="visitors"/>
</p>

> 一本面向"能写点 Python、听过 CUDA 但没读过编译器代码"的小白，
> 手把手带你从 `@tilelang.jit` 一路挖到 `.cubin` 的书。

---

## 👋 先聊两句

这本 Cookbook 说白了就是我自己**学 TileLang 的过程中，和 Claude 反复对话、追代码、踩坑、再回头整理**攒出来的一份中文笔记。

- 网上讲 TVM 的资料一大堆，但要么停在"介绍一下 TVM"，要么直接甩你满脸 C++；
- TileLang 又比较新，中文资料几乎为零；
- 我自己被 IR / Pass / Layout / Warp Specialization / TMA 这些词轮番暴打，只能抓着 Claude 一段一段啃。

于是就把那些"终于搞懂了！"的瞬间沉淀下来，配上**能跳到行号的源码引用**和**能直接跑起来的示例**，就是你现在看到的这些章节。语气会尽量轻松点，不装大佬 😅

---

## ⭐ 顺手点个 Star 呗

**创作真的不易，如果这里的内容帮到你了，回 GitHub 给个 ⭐ 再走呗，那真的是我继续写下去最大的动力！**

👉 [github.com/RuneFang/tvm_tilelang_cookbook](https://github.com/RuneFang/tvm_tilelang_cookbook)

### Star 历史

<img src="https://raw.githubusercontent.com/RuneFang/tvm_tilelang_cookbook/main/assets/star-history.svg" alt="Star History Chart"/>

---

## 📈 谁在看这本笔记

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=RuneFang.tvm_tilelang_cookbook.site&left_text=Total%20Visits&left_color=555&right_color=blue" alt="Total Visits"/>
  &nbsp;
  <img src="https://komarev.com/ghpvc/?username=runefang-cookbook&label=Page%20Views&color=orange&style=flat-square" alt="Page Views"/>
</p>

---

## 这本书解决什么问题

你可能有以下几种困惑，本书按顺序全部回答：

1. **"我 `pip install tilelang` 之后写了 30 行 Python，它凭什么变成 GPU 上跑的 kernel？"** → 第 1 章
2. **"IR、TIR、Stmt、PrimFunc、IRModule 都是啥？为啥所有 pass 都在改一棵树？"** → 第 2 章
3. **"`T.Kernel`、`T.copy`、`T.gemm` 这些 DSL 关键字在源码里到底长什么样？"** → 第 3 章
4. **"Pass 是什么？我想自己加一个 pass 该怎么写？`StmtExprMutator`/`StmtVisitor` 怎么用？"** → 第 4 章
5. **"从 Python 到 CUDA 源码字符串，中间跑了几十个 pass，每个 pass 都在干啥？"** → 第 5 章
6. **"Warp Specialization、软件流水、mbarrier phase counter 是什么？为什么我改一行会让 K=3 时结果错？"** → 第 6 章
7. **"Layout / Fragment / Swizzle 到底在描述什么物理事实？"** → 第 7 章
8. **"TIR 最后是怎么被打印成 CUDA 源码、再被 nvrtc 编成 cubin 的？"** → 第 8 章
9. **"JIT 一次到底缓存了什么、下次调用又跳过了哪些步骤？"** → 第 9 章
10. **"我想加个 pass、修个 bug、提 PR，工作流是什么？怎么 diff IR、怎么找 review？"** → 第 10 章
11. **"kernel 编译报错 / 结果不对，我该按什么顺序抓 bug？TileLang 有哪些内置调试工具？"** → 第 11 章
12. **"if / while / T.dynamic 动态形状 / T.reduce_sum / T.cumsum / T.atomic_add 都在哪、怎么用？"** → 第 12 章
13. **"Hopper 的 cluster / TMA / warp specialization 到底怎么在 TileLang 里落地？"** → 第 13 章
14. **"W4A16、FP8、MXFP4 这些量化 GEMM 在 TileLang 里是怎么写的？lop3.py 那 51KB 干啥？"** → 第 14 章

---

## 全书目录

| # | 章节 | 状态 |
|---|---|---|
| 00 | [前言与阅读指南](./00_preface.md) | ✅ |
| 01 | [一个最小的 TileLang 例子跑起来了发生什么](./01_hello_tilelang.md) | ✅ |
| 02 | [TVM / TIR 基础概念（PrimFunc / Buffer / Stmt / Expr / IRModule）](./02_tvm_tir_basics.md) | ✅ |
| 03 | [TileLang DSL 层次（Kernel / Pipelined / Persistent / copy / gemm）](./03_tilelang_dsl.md) | ✅ |
| 04 | [Pass 系统与 Pass Pipeline](./04_pass_system.md) | ✅ |
| 05 | [Lowering Pipeline 逐 pass 巡礼](./05_lowering_pipeline.md) | ✅ |
| 06 | [软件流水 + Warp Specialization 深挖](./06_pipeline_and_warp_specialize.md) | ✅ |
| 07 | [Layout 系统与 Fragment](./07_layout_and_fragment.md) | ✅ |
| 08 | [Codegen：TIR → CUDA 源码 → cubin](./08_codegen_tir_to_cuda.md) | ✅ |
| 09 | [Runtime、JIT、Kernel Cache](./09_runtime_jit_kernel_cache.md) | ✅ |
| 10 | [写 Pass / 调试 / 贡献工作流](./10_contribute.md) | ✅ |
| 11 | [调试与可视化工具链](./11_debugging_and_visualization.md) | ✅ |
| 12 | [控制流 / 动态形状 / Reduce · Scan · Atomic](./12_control_flow_dynamic_reduce_atomic.md) | ✅ |
| 13 | [Cluster / TMA / Hopper 深挖](./13_cluster_tma_hopper.md) | ✅ |
| 14 | [Quantization / FP8 / MXFP / INT4 Dequant](./14_quantization_fp8_mxfp_int4.md) | ✅ |
| A | [附录 A：TVM/TIR 关键类速查表](./A_tvm_tir_cheatsheet.md) | ✅ |
| B | [附录 B：本仓库源码地图](./B_source_map.md) | ✅ |
| C | [附录 C：术语表](./C_glossary.md) | ✅ |
| D | [附录 D：生态延伸 —— Carver / AutoTune / 多 GPU 现状](./D_ecosystem.md) | ✅ |
| E | [附录 E：编译器背景速成（AST / IR / Pass / SSA / JIT / FFI / …）](./E_compiler_background.md) | ✅ |
| F | [附录 F：Eager JIT 与 CuTe DSL 分支](./F_eager_and_cutedsl.md) | ✅ |

图例：✅ 已完稿 · 🟡 部分完成 · ⏳ 待撰写

---

## 示例代码

正文里所有代码片段都被抽出来做成了**可独立 `python xxx.py` 运行的示例文件**，按章节归档：

📁 **[examples/](./examples/)** — 全 30+ 份示例文件，含 docstring 注明来源章节、跑法和已知坑

推荐用法：读完正文一章 → 打开对应 `examples/chNN_*/` 目录 → 挑一个跑起来 → 改参数看输出。

---

## 怎么读这本书

- **完全小白**：从 00 → 10 顺序读，前 3 章会花你一天，剩下每章 1~2 小时；11-14 章按兴趣挑。**碰到 IR/AST/pass/SSA/FFI 之类的编译器词汇卡壳？先翻附录 E 建心智模型再回来读。**
- **写过一点 CUDA、想理解 TileLang**：跳到 3 → 5 → 6，然后 13（Hopper 特性）。
- **想给 TileLang 提 PR**：读 4 → 10 → 5，配合仓库里的 `docs/developer_guide/` 官方文档一起看。
- **调 bug 中**：**先翻第 11 章挑对工具**，然后按附录 B 找源码目录、回对应章节。
- **写量化 / LLM 推理 kernel**：跳到 13 → 14。
- **看到不认识的 `@tilelang.jit` 写法或 `TILELANG_TARGET=cutedsl`**：翻附录 F。

每章开头都有一个 "TL;DR" 卡片和 "你会读到的真实源码文件" 列表，
方便你边读边用 IDE 跳到源码验证。

---

## 关于代码引用格式

书中所有代码引用都指向**本地仓库路径**，例如：

- `src/transform/inject_pipeline.cc:1234` 表示 `inject_pipeline.cc` 的第 1234 行左右
- `tilelang/engine/lower.py::get_pass_context()` 表示该 Python 文件里的函数
- 版本对应写作时的 `HEAD`，日后源码演进可能会有偏移，但类名 / 函数名一般稳定

---

## 反馈

发现哪一章讲得像天书、哪个术语没解释清楚，或者哪段源码引用对不上，
直接在对应文件下加 `> TODO(reader):` 注释，或者 [来仓库开个 Issue](https://github.com/RuneFang/tvm_tilelang_cookbook/issues) 就行，下一次更新会一并处理。

---

<p align="center">
  <b>如果这份笔记帮你省下了几个小时啃源码的时间，就给它一颗 ⭐ 吧！</b><br>
  <sub>—— 一个也在苦哈哈自学 TileLang 的普通开发者</sub>
</p>
