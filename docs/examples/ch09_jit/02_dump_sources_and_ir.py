"""
第 9 章 §9.8 · 实用工具组合演示：dump source / ptx / sass / dump IR

- 来源：cookbook 09_runtime_jit_kernel_cache.md §9.8
- 目的：把常用的 debugging / introspection 工具串在一起
- 跑法：`python 02_dump_sources_and_ir.py`
- 期望输出：
    [源码]  device_kernel.cu → /tmp/tl_out/kernel.cu
    [源码]  host_kernel.cc   → /tmp/tl_out/host.cc
    [PTX]   → /tmp/tl_out/kernel.ptx
    [SASS]  → /tmp/tl_out/kernel.sass （视 CUDA toolkit 是否装齐）
    [IR]    → /tmp/tl_out/dump/*.py（每个 pass 前后各一份）
- 坑：
    - show_sass 需要 cuobjdump 在 PATH 里；不在则回落到只出 ptx
    - dump_ir_dir 会保留很多文件，用完记得 rm -rf
"""

import os
import shutil
import tilelang
from tilelang import tvm
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 128, bx * 128])
    return kernel


def main():
    out_dir = "/tmp/tl_out"
    dump_dir = os.path.join(out_dir, "dump")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(dump_dir)

    # 打开 dump_ir 配合编译
    with tvm.transform.PassContext(
        opt_level=3,
        config={"tl.enable_dump_ir": True, "tl.dump_ir_dir": dump_dir},
    ):
        mm = matmul(512, 512, 512)

    # 1) 源码
    kernel_path = os.path.join(out_dir, "kernel.cu")
    host_path = os.path.join(out_dir, "host.cc")
    try:
        mm.export_sources(kernel_path=kernel_path, host_path=host_path)
        print(f"[源码] device → {kernel_path}")
        print(f"[源码] host   → {host_path}")
    except Exception as e:
        print(f"[源码] export_sources 失败：{e}；改用 get_kernel_source")
        with open(kernel_path, "w") as f:
            f.write(mm.get_kernel_source())

    # 2) PTX
    try:
        ptx_path = os.path.join(out_dir, "kernel.ptx")
        mm.export_ptx(ptx_path)
        print(f"[PTX] → {ptx_path}")
    except Exception as e:
        print(f"[PTX] 失败：{e}")

    # 3) SASS
    try:
        sass_path = os.path.join(out_dir, "kernel.sass")
        mm.export_sass(sass_path)
        print(f"[SASS] → {sass_path}")
    except Exception as e:
        print(f"[SASS] 失败（需要 cuobjdump）：{e}")

    # 4) dump 出来的 IR
    ir_files = sorted(os.listdir(dump_dir))
    print(f"[IR] dumped {len(ir_files)} snapshots to {dump_dir}")


if __name__ == "__main__":
    main()
