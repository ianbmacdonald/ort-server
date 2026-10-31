# Patches for the prplOS (musl) builds

Each patch is `git format-patch` output against the pinned upstream commit of the third-party source it
modifies; apply with `git am` (or `patch -p1`) before building. ort-server's own source needs no patch.

| Directory | Upstream | Base commit | Patches |
|---|---|---|---|
| `onnxruntime/` | https://github.com/microsoft/onnxruntime | tag `v1.31.1` = `6af1a932e2286d5c58166085d24c68783033edee` | `0001-mlas-keep-the-x86-ISA-runtime-dispatch-in-ORT_MINIMA.patch` |

No other third-party source is modified: cpp-httplib, nlohmann/json, the Rust crates of `third_party/tok_ffi`
and stb_image build unpatched for musl.

## onnxruntime/0001: keep the MLAS x86 ISA dispatch in a minimal build

An ONNX Runtime `--minimal_build` defines `ORT_MINIMAL_BUILD`, and MLAS then compiles out its runtime x86
dispatch for AVX-512F, AVX-512 Core/VNNI, AVX-VNNI and AMX (`onnxruntime/core/mlas/lib/platform.cpp` and
`convsym.cpp`). The AVX-512 kernels are still compiled, but nothing references them, so the linker drops them
and the library runs AVX2 kernels on every CPU. The patch adds `MLAS_MINIMAL_BUILD_KEEP_X86_DISPATCH`
(defined in `mlasi.h`) and keeps both blocks when it is set. It changes nothing in a full build, and its
hunks sit inside `MLAS_TARGET_AMD64`, so it is a no-op on aarch64 (applied there too, so both architectures
build the same source).

Measured with ONNX Runtime 1.30.0 on an AMD Ryzen AI Max+ 395 (Zen 5, AVX-512), DistilBERT, 2 threads,
64 / 512 tokens:

| library | warm median | AVX-512 instructions | size |
|---|---|---|---|
| minimal build, unpatched | 29.7 / 169.3 ms | 0 | 4,565,632 B |
| minimal build, patched | 14.1 / 94.4 ms | 37,263 | 5,143,328 B |
| pip glibc full build | 14.0 / 93.9 ms | 35,512 | - |

Scores are identical across all three. AVX2-only CPUs take the same code path as before. The patch was
carried from 1.30.0 to 1.31.1 unchanged (one hunk applies 14 lines lower).

    git clone --branch v1.31.1 https://github.com/microsoft/onnxruntime
    git -C onnxruntime am ../patches/musl/onnxruntime/*.patch
