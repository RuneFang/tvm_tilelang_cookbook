# 第 14 章 · Quantization / FP8 / MXFP / INT4 Dequant

> **TL;DR**：量化 kernel 的核心矛盾是"**权重以低比特存、计算却要用高比特做**"，于是每次都要在 kernel 里做 **dequant**（低比特 → fp16/bf16）。`tilelang/quantize/` 那一大堆代码，本质都在**把这个 dequant 做得尽可能快**——用位运算、用 `lop3.b32`、用 `prmt` 替代查表。本章讲清楚这些手段各自解决什么问题。
>
> **本章目标**
> 讲清楚 TileLang **量化子系统**：为什么 `tilelang/quantize/` 里有 60KB+ 代码、这些代码在 kernel 里怎么用、以及为什么 FP8 / MXFP / INT4 dequant 是 LLM inference 的必修课。
>
> 学完你能：
> - 写一个 W4A16 的 dequant GEMM（4bit 权重 + fp16 激活）
> - 知道 FP8 e4m3 / e5m2 在不同硬件上的正确 dtype 名字
> - 看得懂 `_tir_packed_to_unsigned_convert`、`_tir_u8_to_f4_to_bf16` 这类内部函数为什么长那样
> - 理解 `lop3.py`（51KB）和 `mxfp.py`（10KB）在做什么

> 前置：第 3 章（PrimFunc）、第 7 章（layout/fragment）、第 12 章（reduce/atomic）读过。

---

## 14.0 为什么 LLM 时代必须懂量化

先做一次数量级估算，看看为什么"量化"从可选项变成了必选项：

**Llama-70B 的 weight 总量**：≈ 140 GB（fp16）

| 精度 | 权重大小 | 一张 H100（80GB HBM）能不能装 |
|---|---|---|
| fp32 | 280 GB | ❌ |
| fp16 / bf16 | 140 GB | ❌（还要 2 张） |
| **fp8** | 70 GB | ✅（刚好） |
| **int4** | 35 GB | ✅ 还剩一半装 KV cache |

再加上 **HBM 带宽是瓶颈** 这个事实——decode 阶段每 token 都要把整份权重 load 一遍——**权重精度从 16 bit 降到 4 bit 就等于速度 4×**。

所以问题变成：**怎么在 kernel 里高效地把 4 bit 权重 dequant 成 bf16/fp16，再送进 tensor core**。这就是本章 90% 的内容。

> **先扫清几个数值格式的命名**（本章反复用，不懂会看不下去）：
>
> - **浮点数 = 1 位符号 + 若干位指数（exponent, e）+ 若干位尾数（mantissa, m）**。指数位决定**能表示的范围**，尾数位决定**精度**。
> - **fp16**：1+5+10 位。**bf16**（bfloat16）：1+**8**+**7** 位——和 fp16 同样 16 位，但**指数位更多、尾数位更少**，所以**范围和 fp32 一样大、但精度略低**，训练/推理里很常用。
> - **fp8** 有两种，名字直接告诉你位数分配：**e4m3** = 4 位指数 + 3 位尾数（精度稍高、范围小），**e5m2** = 5 位指数 + 2 位尾数（范围大、精度低）。（14.7 会讲不同硬件上它们的具体 dtype 名字和 OCP/FNUZ 变种。）
> - **int4**：4 位整数。**nibble** = 半个字节 = 4 位，所以"一个 uint8 装 2 个 int4"就是"一个 byte 装 2 个 nibble"。
> - **scale（缩放因子）**：低比特存不下真实数值范围，于是存一个"整数/低精度值 × scale"。按 scale 的粒度分：**per-tensor**（整个张量共享 1 个 scale，最省但最粗）、**per-channel**（每行/列一个）、**per-element**（每个元素一个，最准但最贵）。**MXFP** 走的是折中：每 32 个元素共享一个 scale。

---

## 14.1 TileLang 量化系统的三层结构

```
   layer 1  一堆位运算 helper（Python）
   ────────────────────────────────────────
       tilelang.quantize.quantization  —— TIR-level 位运算
       tilelang.quantize.utils         —— 打包/解包/交织
       tilelang.quantize.mxfp          —— MXFP scale/decode 例程
       tilelang.quantize.lop3          —— LOP3 指令 dispatch

   layer 2  Kernel 里怎么调
   ────────────────────────────────────────
       tilelang.language.fp8           —— fp8 dtype 选择
       T.Tensor((.., .., ), dtype)     —— storage_dtype 就是 uint8/uint32
       在 T.Parallel 里对每个元素调 layer 1 的函数

   layer 3  完整 dequant GEMM 参考
   ────────────────────────────────────────
       examples/dequantize_gemm/*.py   —— 10 个真实 kernel
```

我们按 **layer 3 → layer 2 → layer 1** 的顺序讲，因为**读者最想要的先给**（一个跑得起来的 W4A16 kernel），然后再回过头解释每层。

---

## 14.2 完整例子：W4A16 Dequant GEMM

先看官方 `examples/dequantize_gemm/README.md` 里给的最小 kernel（我加了逐行注释）：

```python
@T.prim_func
def dequant_matmul(
    A:  T.Tensor(A_shape, in_dtype),         # fp16 激活
    B:  T.Tensor(B_shape, storage_dtype),    # uint8 存 4-bit 权重（2 个/byte）
    Ct: T.Tensor((N, M), out_dtype),
):
    with T.Kernel(T.ceildiv(N, block_N),
                  T.ceildiv(M, block_M),
                  threads=threads) as (bx, by):

        A_shared          = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared          = T.alloc_shared(B_shared_shape, storage_dtype)   # 还是 uint8
        B_local           = T.alloc_fragment(B_shared_shape, storage_dtype)
        B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)  # ★
        Ct_local          = T.alloc_fragment((block_N, block_M), accum_dtype)

        T.clear(Ct_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

            # ① 正常载入 A 和 B（但 B 是 uint8，一 byte 等于 2 个 int4）
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
            T.copy(B_shared, B_local)

            # ② ★ 核心：对每个 int4 位置调 packed→fp16 dequant
            for i, j in T.Parallel(block_N, block_K):
                B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
                    num_bits,                # 4
                    B_local[i, j // 2],      # 2 个 int4 挤在 j//2 那个 byte 里
                    j % 2,                   # 拿第 0 个还是第 1 个 nibble
                    dtype=in_dtype,          # 目标 fp16
                )

            # ③ 用 dequant 完的 fp16 fragment 直接进 tensor core
            T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)

        T.copy(Ct_local, Ct[bx * block_N, by * block_M])
```

> **别被那个"双层括号"`_tir_packed_to_unsigned_convert("int", 8)(...)` 吓到**：它是**"返回函数的函数"（工厂函数）**。
> - **第一层 `("int", 8)`** 只做"配置"：告诉它打包用的存储类型是 `int`、位宽 `8`（即 `int8`），返回一个**专门解这种打包格式的转换函数**。
> - **第二层 `(num_bits, val, pos, dtype=...)`** 才真正干活：把某个 byte（`B_local[i, j//2]`）里第 `pos` 个 nibble 解出来、转成目标 `dtype`。
>
> 所以写成两组括号 = "先按存储格式取到对应的解码器，再用它解一个具体元素"。源码见 [`tilelang/quantize/quantization.py`](../../tilelang/quantize/quantization.py) 的 `_tir_packed_to_unsigned_convert`（函数体里 `return f_convert` 就是这个"返回函数"的证据）。它内部做的位运算，14.3 会展开讲。

**理解这个 kernel 有 5 个 key insight**：

1. **权重的 `storage_dtype` 是 `uint8`，不是 int4**。TileLang 没有 int4 dtype，用 uint8 打包，一个 byte 存 2 个 int4（`num_elems_per_byte=2`）。
2. **索引除法 `k * block_K // num_elems_per_byte`**：因为 B 一 byte 装 2 个 int4，K 维实际内存尺寸是逻辑尺寸 / 2。
3. **`B_dequantize_local` 是关键中间物**：dequant 完的 fp16 数据必须过 fragment 才能给 tensor core 用。
4. **`T.Parallel` 里的 dequant 是逐元素的**——但因为 pass 会把它 vectorize，最终 CUDA 里其实是每个 thread 一次处理 4/8 个元素。
5. **dequant 的位置在 pipeline 里**：跟 copy 和 gemm 在同一个 stage。想省寄存器可以把 dequant 挪到 gemm 前一 stage，让软件流水更宽——这是 `example_dequant_gemm_fine_grained.py` 干的事。

---

## 14.3 Layer 1（一）：`quantization.py` 的 TIR 位运算

现在往下钻。`_tir_packed_to_unsigned_convert` 到底做了什么？看真实源码简化版：

```python
def _tir_u32_to_int_to_float(nbit, val, pos, dtype):
    # val 是 uint32 packed
    # pos 是 nibble 位置
    mask = (1 << nbit) - 1
    unextended = (val >> (pos * nbit)) & mask     # 抠出 nbit 位
    shift = 32 - nbit
    extended = (int32(unextended) << shift) >> shift  # 符号扩展
    return cast(dtype, extended)                    # int→fp16
```

这些 `_tir_*` 函数就是把**位运算表达式**用 TIR 拼出来。这样在 kernel 里调用它们，编译器可以：
- 把连续几个 nibble 的位运算**融合**成一次 `shl/and/or`
- 把 `int→fp16` 转换替换成硬件 fast-path（PTX `cvt` 或 `lop3`）

一份**真实公开的入口清单**（来自 `tilelang/quantize/__init__.py`）：

| 函数 | 输入 → 输出 | 用途 |
|---|---|---|
| `_tir_packed_int_to_int_convert` | packed intN → intM | 通用整数解包 |
| `_tir_packed_to_signed_convert` | packed → signed int/float | INT4/INT8 有符号解包 |
| `_tir_packed_to_unsigned_convert` | packed → unsigned int/float | 大部分 W4A16 用这个 |
| `_tir_packed_to_unsigned_convert_with_zeros` | 同上，多个 zero-point | 对称 vs 非对称量化 |
| `_tir_packed_to_fp4_to_f16` | packed FP4 → f16 | FP4 dequant |
| `_tir_u8_to_f8_e4m3_to_f16` | u8 存 fp8 → f16 | FP8 dequant 到 f16 |
| `_tir_u8_to_f4_to_bf16` | u8 存 fp4 → bf16 | FP4 dequant 到 bf16（带 scale） |

### 举一个：`_tir_u8_to_f4_to_bf16` 在干什么

按源码逐位说明：
1. 从 uint8 里抠出 4 位（FP4 E2M1 格式：符号 1 + 指数 2 + 尾数 1）
2. 把 FP4 的 2 位指数**偏移到 bf16 的 8 位指数**（bias = 126）
3. 加上 `scale`（外部传进来的量化 scale）
4. 用 `min(exp + scale, 255)` clamp
5. 用位运算重组成 bf16 的 16 位并 `reinterpret`

这就是**为什么这些 helper 都很长**——每一步都要用 TIR 表达式手工搓出来，让编译器最终能生成一条 PTX。

---

## 14.4 Layer 1（二）：`lop3.py` 的 LOP3 加速

`lop3.py` **51KB** 的代码干一件事：**用 CUDA `lop3.b32` 指令一次做 3 输入位运算**。

背景：PTX 的 `lop3.b32 %dst, %a, %b, %c, immLut` 一条指令能算出**任意** 3 输入 boolean 函数 `f(a,b,c)`——`immLut` 是一个 8 位真值表。它比"and + or + xor 三条指令"快 3 倍。

在 dequant 里非常有用：**把 4 个 nibble 展开成 4 个 fp16**这个操作，恰好可以写成一条 `lop3` + 一条 `mul`。所以 `lop3.py` 里维护了**几十种 LOP3 pattern 的真值表和查询函数**。

对外只暴露一个入口：

```python
from tilelang.quantize import get_lop3_intrin_group

lop3_desc = get_lop3_intrin_group(
    in_dtype="int4",
    out_dtype="fp16",
    storage_nbit=32,
    with_scale=True,
    with_zeros=False,
    zeros_mode="original",
)
```

返回一个描述器，编译器把它作为一段 inline PTX 注入到最终 CUDA。你通常不直接调这个，是**在写更高级的 dequant template 时用**——例如 BitBLAS。

一般用户看这一节的目的：**知道它存在**，并且**能在生成的源码里认出它**。想亲眼看到"一条 `lop3.b32` 替代多条位运算"这个运行逻辑，跑一个 int4 dequant kernel、dump 出 CUDA 源码 grep 一下即可：

```python
# 跑仓库里现成的 int4 dequant GEMM 例子，把生成源码打印出来
# examples/dequantize_gemm/ 下有多个现成脚本，任选一个 fp16 int4 的
kernel = ...                                  # 你的 int4 dequant kernel（见 examples/dequantize_gemm/）
src = kernel.get_kernel_source()

for line in src.splitlines():
    if "lop3.b32" in line:                    # ← 就是 lop3.py 注入的那条指令
        print(line.strip())
# 你会看到类似：asm volatile("lop3.b32 %0, %1, %2, %3, %4;" ... immLut ...)
# 即：一条指令 + 一个 8 位真值表 immLut，替代了 and/or/xor 三条指令
```

看到这几行，你就把"14.3 里那串手搓的位运算"和"最终一条 PTX 指令"对上了。

---

## 14.5 Layer 1（三）：`mxfp.py` 的 MXFP 支持

**MXFP**（Micro-scaling FP）是 OCP 提出的 FP4/FP6/FP8 标准：**每 32 个元素共享一个 8 位 scale**（比 per-tensor scale 精细，比 per-element scale 便宜）。Blackwell 的 tensor core 原生支持 MXFP4/MXFP6/MXFP8。

`tilelang/quantize/mxfp.py` 提供：

```python
from tilelang.quantize import get_mxfp_intrin_group
```

返回一段 CUDA / HIP inline 源码，负责**从 FP4 packed byte 解出 bf16 + scale**。源码里的核心函数 `decode_f4_to_bf16_twiddling`——用 `prmt` + `mul.bf16x2` 一次搞定 4 个 FP4→bf16 变换。

这里的 `twiddling` 借用了 C/CUDA 圈的口语 "bit-twiddling"（位操作技巧）——**泛指各种用位运算实现的小把戏**，不是一个有严格定义的正式术语。这里具体指的是：**用 PTX 的字节重排指令 `prmt` 代替查表**，从而完全避免 shared memory 访问。同一段代码在 CUDA 和 HIP（gfx950）上有**两份实现**，因为 AMD 没有 `prmt` 指令，只能用 C++ 位运算展开。这就是为什么 `mxfp.py` 里两个模板都存在。

**用户视角**：写 MXFP4 GEMM 时你不用管这些——直接看 [`examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py`](../../examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py)，它把 `get_mxfp_intrin_group` 用 `T.import_source` 或类似机制嵌入到 kernel 里。

想直接看看 `get_mxfp_intrin_group` 生成的那段源码长什么样、`prmt` 到底出现在哪，不必编 kernel，打印它返回的源码字段即可：

```python
from tilelang.quantize import get_mxfp_intrin_group

# use_twiddling=True 才选到用 prmt 的那份实现（source_bit=4 + bf16 输出）
grp = get_mxfp_intrin_group(
    out_dtype="bfloat16",
    source_bit=4,
    storage_dtype="uint8",
    use_twiddling=True,
)
print(grp["func_name"])                    # decode_fp4_to_bf16_twiddling
src = grp["c_source"]                       # 这段就是要 import 进 kernel 的 CUDA 源码
for line in src.splitlines():
    if "prmt" in line:                      # ← 字节重排指令，就是它替代了查表
        print(line.strip())
```

> `get_mxfp_intrin_group` 返回一个 dict，两个键：`func_name`（生成的解码函数名）和 `c_source`（对应的 C/PTX 源码字符串）。传 `use_twiddling=False` 时选到的是不带 `prmt` 的普通实现，可对照着 dump 一次看区别。

---

## 14.6 Layer 1（四）：`utils.py` 的打包工具

数据在 host 端也要打包成 uint8/uint32，`utils.py` 提供三个入口：

```python
from tilelang.quantize import gen_quant4, general_compress, interleave_weight
```

| 函数 | 干啥 |
|---|---|
| `gen_quant4(...)` | 生成 int4 量化权重（含 zero-point / scale） |
| `general_compress(w, storage_nbit, in_nbit, ...)` | 把一个大的 int8/int16 权重按 in_nbit 打包成 storage_nbit 的整数 |
| `interleave_weight(w, ...)` | 按硬件友好的 order 交织权重（LOP3 dequant 要求特定 layout） |

**"interleave"是什么意思**？在 GPU 上 4 个连续 fp16 值来自同一 warp 的 4 个 thread。dequant 时要一条指令解 4 个 int4，必须让**内存里连续的 4 个 int4** 对应**未来 fragment 里同一 lane 的 4 个 slot**。这个映射不是恒等，需要预先交织。

**忘了 `interleave_weight` 是常见坑**：数值算出来"看起来对但每一个都对不上参考实现"，就是它。

---

## 14.7 Layer 2：`tilelang.language.fp8`

FP8 在不同硬件上的 dtype 名字**不一样**，这个模块就是帮你选对：

```python
from tilelang.language.fp8 import determine_fp8_type

dtype = determine_fp8_type("e4m3")
# 在 H100 (CUDA)：      T.float8_e4m3fn
# 在 gfx942 (ROCm):     T.float8_e4m3fnuz    ← 注意 fnuz 后缀
# 在 gfx950 (ROCm新)：  T.float8_e4m3fn      ← OCP 兼容
```

**为什么这么麻烦**：FP8 有两个"变种"标准：
- **OCP** (CUDA, gfx950)：有 NaN / Inf，符合 IEEE
- **FNUZ** (旧 AMD)：无 NaN / Inf，指数偏移不同

数值不一致时先检查这个函数返回的 dtype 是否对得上你 host 端的 `torch.float8_*`。TileLang 还有 `determine_torch_fp8_type()` 帮你在 host 端也拿到匹配的 torch dtype。

---

## 14.8 全景对照：几种量化方案

在真实 example 里都有对应 kernel，直接指路：

| 精度组合 | 应用场景 | 参考 kernel |
|---|---|---|
| **W4A16**（int4 权重 + fp16 激活） | Llama / Qwen inference | `example_dequant_gemv_fp16xint4.py` |
| **W4A8**（int4 权重 + fp8 激活） | 极致省显存的 inference | `example_dequant_gemm_w4a8.py` |
| **W4A(fp8) Hopper** | H100 上 DeepSeek V3 类模型 | `example_dequant_gemm_w4_fp8_ds_v3_hopper.py` |
| **FP4×BF16 Hopper** | Blackwell 前的最佳精度权衡 | `example_dequant_gemm_bf16_fp4_hopper.py` |
| **MXFP4×BF16 Hopper** | OCP MXFP，块 scale | `example_dequant_gemm_bf16_mxfp4_hopper.py` |
| **MXFP4×BF16 CDNA4** | AMD gfx950 的对应实现 | `example_dequant_gemm_bf16_mxfp4_cdna4.py` |
| **Fine-grained pipelining dequant** | 把 dequant 放进 pipeline 独立 stage | `example_dequant_gemm_fine_grained.py` |

**一个通用套路**（所有这些 kernel 共同的骨架）：

```
主循环:
  ① copy A（激活，正常 dtype）        → A_shared
  ② copy B_packed（uint8/uint32）    → B_shared    ← 尺寸缩小 2×/4×/8×
  ③ copy 到 fragment                  B_local
  ④ 位运算 dequant 到高精度 fragment  B_dequant_local
  ⑤ T.gemm(A_shared, B_dequant_local, C_local)
```

真正的性能差异**全在 ④ 的实现**：
- 朴素版：`_tir_packed_to_unsigned_convert` 逐元素解
- 中级版：`lop3.py` 生成 LOP3 pattern，一条指令解 4 个
- 高级版：`mxfp.py` 里的 twiddling，用 `prmt` 完全避免表查
- 顶级版：BitBLAS 那样叠 layout permutation + LOP3 + 软件流水

---

## 14.9 数值精度的 sanity check

量化 kernel 最容易在**数值上**出错。三条 always-do：

### 1. 参考实现要用**同样的量化**做

你的 dequant kernel 是 `int4 → fp16 → matmul`。参考实现**不能**是 `float32 matmul`——那样误差会大到无从判断"是量化本身的误差还是 kernel bug"。参考实现应该是：

```python
def ref(A_fp16, B_int4, scale, zero):
    B_fp16 = (B_int4.float() - zero) * scale     # 完全一样的 dequant 公式
    return (A_fp16.float() @ B_fp16.T).to(torch.float16)
```

这样对比时 `rtol=1e-3, atol=1e-3` 就足够——如果对不上，一定是 kernel bug 而非量化本身。

### 2. 用第 11 章的工具做 layered 排查

- **先 `T.print` 打前 8 个 `B_dequantize_local` 值**：看解出来的 fp16 有没有一开始就是全 0 或者数量级不对
- **`register_cuda_postproc_callback` grep `lop3` / `cvt` / `prmt`**：确认硬件加速有没有生成
- **`TL_LAYOUT_VISUALIZATION_ENABLE`**：如果解出来"某几个元素对某几个不对"，很可能是 interleave 错了，layout 图能立即看出

### 3. 别忘了 `interleave_weight`

前面提过。如果你 CPU 端只 `pack` 没 `interleave`，dequant 后拿到的 fp16 fragment 里 lane 顺序错，`T.gemm` 直接吃到乱序数据。

---

## 14.10 小结

- LLM 时代量化是**性能刚需**：4-bit 权重 = 4× 带宽
- TileLang 量化系统分**三层**：kernel 里的位运算 helper（`quantization.py` / `lop3.py` / `mxfp.py`）→ 语言层入口（`fp8.py` / `T.import_source`）→ 参考 kernel（`examples/dequantize_gemm/`）
- 典型 dequant GEMM 骨架 = 正常 GEMM + 一个 `T.Parallel` 里做 packed→fp16 位变换 + 送进 `T.gemm`
- 性能靠 **LOP3 / MXFP twiddling / interleave** 三招
- FP8 dtype 名字**跨硬件不一致**，一律用 `determine_fp8_type()` 拿
- 数值 debug：参考实现要用**同一个量化公式**，然后按第 11 章的工具链 layered 排查

下一部分进入附录 F，讲两条**TileLang 里存在但正文没提**的重要"分支路径"：**Eager JIT 模式**和 **CuTe DSL codegen**。
