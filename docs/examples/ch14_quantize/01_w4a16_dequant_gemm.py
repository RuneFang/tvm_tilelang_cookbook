"""
第 14 章 §14.2 · W4A16 Dequant GEMM 的可运行骨架

- 来源：cookbook 14_quantization_fp8_mxfp_int4.md §14.2
- 目的：单文件演示"int4 权重 + fp16 激活 → fp16 输出"完整流程；host 端 pack、device 端 dequant→gemm
- 跑法：`python 01_w4a16_dequant_gemm.py`
- 期望输出：`PASS  max abs diff = ...`
- 坑：
    - _tir_packed_to_unsigned_convert 是无符号解包；如果你需要有符号 int4 请用 _tir_packed_to_signed_convert
    - "参考实现"必须用同样的 dequant 公式，否则你分不清是"kernel bug"还是"量化本身的误差"
    - 真实生产 kernel 会加 scale / interleave_weight，这里为了 minimal 都省了

upstream 完整参考：examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py
"""

import torch
import tilelang
import tilelang.language as T
from tilelang.quantize import _tir_packed_to_unsigned_convert


def build(M, N, K, block_M=64, block_N=64, block_K=32, num_stages=2, threads=128):
    num_bits = 4
    num_elems_per_byte = 8 // num_bits  # 2
    storage_dtype = "uint8"
    in_dtype = "float16"
    out_dtype = "float16"
    accum_dtype = "float32"

    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)      # uint8, 2 nibbles / byte
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def dequant_matmul(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Ct: T.Tensor((N, M), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N),
                          T.ceildiv(M, block_M),
                          threads=threads) as (bx, by):

                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)

                T.clear(Ct_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # A：正常 load
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    # B：packed uint8 load
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)

                    # 逐元素解 4-bit → fp16
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=in_dtype,
                        )

                    T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)

                T.copy(Ct_local, Ct[bx * block_N, by * block_M])
        return dequant_matmul

    return kernel()


def dequant_ref(B_packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """host 端做同样的解 4-bit 逻辑，作为参考实现。"""
    B_int = torch.zeros(N, K, dtype=torch.int32)
    for j in range(K):
        byte = B_packed[:, j // 2]
        nib = (byte >> ((j % 2) * 4)) & 0xF
        B_int[:, j] = nib.int()
    return B_int.to(torch.float16)


def main():
    M, N, K = 128, 128, 128
    k = build(M, N, K)

    # 造 int4 权重：把每个 nibble 都限制在 [0,15]，两个塞进一个 byte
    torch.manual_seed(0)
    B_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32)
    B_packed = torch.zeros(N, K // 2, dtype=torch.uint8)
    B_packed[:, :] = (B_int4[:, 0::2] | (B_int4[:, 1::2] << 4)).to(torch.uint8)

    A = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    B_packed_gpu = B_packed.cuda()

    Ct = k(A, B_packed_gpu)
    torch.cuda.synchronize()

    # 参考：dequant → matmul（用完全一样的公式，避免"量化本身误差"混入判定）
    B_fp16 = dequant_ref(B_packed, N, K).cuda()
    ref_Ct = (B_fp16.float() @ A.float().T).to(torch.float16)

    diff = (Ct - ref_Ct).abs().max().item()
    ok = diff < 5.0
    print(f"{'PASS' if ok else 'FAIL'}  max abs diff = {diff:.4f}")


if __name__ == "__main__":
    main()
