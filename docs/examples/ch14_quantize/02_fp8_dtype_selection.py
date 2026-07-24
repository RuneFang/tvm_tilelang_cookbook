"""
第 14 章 §14.7 · determine_fp8_type — 跨硬件挑对 FP8 dtype

- 来源：cookbook 14_quantization_fp8_mxfp_int4.md §14.7
- 目的：不写 kernel，纯打印在当前 target 下 e4m3 / e5m2 对应的 tvm dtype
- 跑法：`python 02_fp8_dtype_selection.py`
- 期望输出：
    e4m3 → float8_e4m3fn        (CUDA / gfx950)
      OR   float8_e4m3fnuz      (gfx942/gfx940 旧 AMD)
    e5m2 → float8_e5m2         (CUDA / gfx950)
- 坑：
    - determine_fp8_type 的返回值是 tvm 里的 dtype 名字字符串（不是 torch dtype）
    - 想同时对齐 torch 端，用 determine_torch_fp8_type
"""

try:
    from tilelang.language.fp8 import determine_fp8_type, determine_torch_fp8_type
except ImportError:
    print("你的 TileLang 版本没有 tilelang.language.fp8；请升级到 0.1.10+")
    raise


def main():
    for tag in ["e4m3", "e5m2"]:
        tvm_dtype = determine_fp8_type(tag)
        try:
            torch_dtype = determine_torch_fp8_type(tag)
        except Exception as e:
            torch_dtype = f"<n/a: {e}>"
        print(f"{tag} → tvm={tvm_dtype!s:<28} torch={torch_dtype}")


if __name__ == "__main__":
    main()
