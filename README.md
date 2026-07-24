# TVM / TileLang Cookbook

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
  <a href="https://github.com/RuneFang/tvm_tilelang_cookbook/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/RuneFang/tvm_tilelang_cookbook?style=flat-square" alt="License"/>
  </a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=RuneFang.tvm_tilelang_cookbook" alt="visitors"/>
</p>

> 一本面向"能写点 Python、听过 CUDA 但没读过编译器代码"的小白，
> 手把手带你从 `@tilelang.jit` 一路挖到 `.cubin` 的书。

---

## 👋 先打个招呼

嗨，这个仓库其实就是我自己在**学 TileLang 的过程中和 Claude 反复聊天、追代码、踩坑、再回头整理**攒出来的一份中文笔记。

写它的初心特别朴素：

- 网上讲 TVM 的资料一大堆，但大多数要么停在"介绍 TVM 是什么"，要么直接甩你一脸 C++ 源码；
- TileLang 又是个相对新的项目，中文资料几乎为零；
- 我自己看的时候被 IR / Pass / Layout / Warp Specialization / TMA 这些词轮番暴打，只好抓着 Claude 一段一段啃，一点一点问。

于是我就把这些对话里"终于搞懂了！"的瞬间沉淀下来，配上**能跳到具体行号的源码引用**和**能直接 `python xxx.py` 跑起来的示例**，变成了你现在看到的这本 Cookbook。

所以它的画风大概是这样的：

- 不装大佬，遇到不懂的词就先解释一遍再往下讲；
- 每个结论都尽量能对到 `src/xxx.cc:1234` 或 `tilelang/yyy.py::func` 这种具体位置；
- 会告诉你"我踩过哪个坑、Claude 一开始给了个错答案后来是怎么纠正的"这种真实过程。

如果你也在自学 TileLang / TVM，希望这本笔记能帮你少走一点弯路 🙌

---

## ⭐ 求个 Star（真的！）

**创作不易，白嫖也别悄悄溜走呀 —— 顺手点个 ⭐ 再走呗！**

你的一个 star 对我来说：

- 是继续写下去的最大动力（真的，每次看到 star 涨一颗都会开心半天）；
- 能让更多同样在啃 TileLang 的小伙伴通过 GitHub 搜到这份笔记；
- 也让我更有底气继续追新版本、补新章节。

> 👉 点右上角那颗 ⭐ **Star**，就一秒钟，谢谢你！

### Star 历史曲线

<img src="./assets/star-history.svg" alt="Star History Chart"/>

---

## 📖 在线阅读

👉 **<https://runefang.github.io/tvm_tilelang_cookbook/>**

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
| 00 | [前言与阅读指南](./docs/00_preface.md) | ✅ |
| 01 | [一个最小的 TileLang 例子跑起来了发生什么](./docs/01_hello_tilelang.md) | ✅ |
| 02 | [TVM / TIR 基础概念（PrimFunc / Buffer / Stmt / Expr / IRModule）](./docs/02_tvm_tir_basics.md) | ✅ |
| 03 | [TileLang DSL 层次（Kernel / Pipelined / Persistent / copy / gemm）](./docs/03_tilelang_dsl.md) | ✅ |
| 04 | [Pass 系统与 Pass Pipeline](./docs/04_pass_system.md) | ✅ |
| 05 | [Lowering Pipeline 逐 pass 巡礼](./docs/05_lowering_pipeline.md) | ✅ |
| 06 | [软件流水 + Warp Specialization 深挖](./docs/06_pipeline_and_warp_specialize.md) | ✅ |
| 07 | [Layout 系统与 Fragment](./docs/07_layout_and_fragment.md) | ✅ |
| 08 | [Codegen：TIR → CUDA 源码 → cubin](./docs/08_codegen_tir_to_cuda.md) | ✅ |
| 09 | [Runtime、JIT、Kernel Cache](./docs/09_runtime_jit_kernel_cache.md) | ✅ |
| 10 | [写 Pass / 调试 / 贡献工作流](./docs/10_contribute.md) | ✅ |
| 11 | [调试与可视化工具链](./docs/11_debugging_and_visualization.md) | ✅ |
| 12 | [控制流 / 动态形状 / Reduce · Scan · Atomic](./docs/12_control_flow_dynamic_reduce_atomic.md) | ✅ |
| 13 | [Cluster / TMA / Hopper 深挖](./docs/13_cluster_tma_hopper.md) | ✅ |
| 14 | [Quantization / FP8 / MXFP / INT4 Dequant](./docs/14_quantization_fp8_mxfp_int4.md) | ✅ |
| A | [附录 A：TVM/TIR 关键类速查表](./docs/A_tvm_tir_cheatsheet.md) | ✅ |
| B | [附录 B：本仓库源码地图](./docs/B_source_map.md) | ✅ |
| C | [附录 C：术语表](./docs/C_glossary.md) | ✅ |
| D | [附录 D：生态延伸 —— Carver / AutoTune / 多 GPU 现状](./docs/D_ecosystem.md) | ✅ |
| E | [附录 E：编译器背景速成（AST / IR / Pass / SSA / JIT / FFI / …）](./docs/E_compiler_background.md) | ✅ |
| F | [附录 F：Eager JIT 与 CuTe DSL 分支](./docs/F_eager_and_cutedsl.md) | ✅ |

图例：✅ 已完稿 · 🟡 部分完成 · ⏳ 待撰写

---

## 示例代码

正文里所有代码片段都被抽出来做成了**可独立 `python xxx.py` 运行的示例文件**，按章节归档：

📁 **[docs/examples/](./docs/examples/)** — 全 30+ 份示例文件，含 docstring 注明来源章节、跑法和已知坑

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

## 🤝 反馈 & 贡献

发现哪一章讲得像天书、哪个术语没解释清楚、哪段源码引用对不上？非常欢迎：

- 提 [Issue](https://github.com/RuneFang/tvm_tilelang_cookbook/issues) 直接开喷（温柔一点更好 🙏）；
- 或者直接 PR，也可以在对应文件里加 `> TODO(reader):` 注释，我下次更新一并处理；
- 觉得对你有帮助的话，别忘了回来给个 ⭐ Star 呀！

---

## 📜 License

本仓库为个人学习笔记，内容以 [MIT](./LICENSE) 协议开源，欢迎自由使用与二次创作，转载请注明出处即可。

---

<p align="center">
  <b>如果这份笔记帮你省下了几个小时啃源码的时间，就给它一颗 ⭐ 吧！</b><br>
  <sub>—— 一个也在苦哈哈自学 TileLang 的普通开发者</sub>
</p>
