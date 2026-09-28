# ONNX Runtime patches for the minimal musl build

## ort-minimal-keep-x86-dispatch.patch (ONNX Runtime 1.30.0)

An ONNX Runtime `--minimal_build` defines `ORT_MINIMAL_BUILD`, and MLAS then compiles out its runtime x86
dispatch for AVX-512F, AVX-512 Core/VNNI, AVX-VNNI and AMX (`onnxruntime/core/mlas/lib/platform.cpp` and
`convsym.cpp`). The AVX-512 kernels are still compiled, but nothing references them, so the linker drops them
and the library runs AVX2 kernels on every CPU.

The patch adds `MLAS_MINIMAL_BUILD_KEEP_X86_DISPATCH` (defined in `mlasi.h`) and keeps both blocks when it is set.

Measured on an AMD Ryzen AI Max+ 395 (Zen 5, AVX-512), DistilBERT, 2 threads, 64 / 512 tokens:

| library | warm median | AVX-512 instructions | size |
|---|---|---|---|
| minimal build, unpatched | 29.7 / 169.3 ms | 0 | 4,565,632 B |
| minimal build, patched | 14.1 / 94.4 ms | 37,263 | 5,143,328 B |
| pip glibc full build | 14.0 / 93.9 ms | 35,512 | - |

Scores are identical across all three. AVX2-only CPUs take the same code path as before.

Apply it to the ONNX Runtime source tree before building:

    git -C onnxruntime apply ../patches/ort-minimal-keep-x86-dispatch.patch
