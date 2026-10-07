#!/bin/bash
# Build ONNX Runtime 1.30.0 (shared library, CPU EP) for prplOS 5.1 musl, on the ai4 build host layout.
#
#   tools/ort-musl-build.sh <x86_64|aarch64> <required_operators.config | full> <build-dir>
#
# <config>: a --minimal_build (ORT-format models only) with exactly the operators in the config, reduced
#   operator type support, no ML ops, as the v0.1.x releases. Make a config from your models with
#   tools/ort-ops-config.sh. patches/ort-minimal-keep-x86-dispatch.patch must be applied to the source.
# full: a standard build (every ONNX and contrib operator, ML ops, loads .onnx and .ort).
#
# Inputs ($R, default ~/build-litert): onnxruntime-src (or $ORT_SRC) at v1.30.0 with the patch applied, the
# prplOS 5.1 staging trees, tflite-venv (python for build.py). The prplOS sysroot's protobuf 3.17.3 must be
# moved out first (ORT would find it and fail against its newer protoc); this script only checks.
# Run it inside a memory guard, e.g. systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0.
set -euo pipefail
ARCH=$1; OPS=$2; B=$(readlink -f "$3")
R=${R:-$HOME/build-litert}; SRC=${ORT_SRC:-$R/onnxruntime-src}; JOBS=${JOBS:-8}
HERE=$(cd "$(dirname "$0")/.." && pwd)
if [ "$ARCH" = aarch64 ]; then
  SD=$R/prplos-5.1-staging-aarch64; S=$SD/target-aarch64_cortex-a53_musl
  T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin/aarch64-openwrt-linux-musl-
  # -march/-mtune, not -mcpu: a global -mcpu overrides the per-file -march MLAS/KleidiAI use for
  # their runtime-dispatched kernels.
  F="-march=armv8-a+crc -mtune=cortex-a53"
  EXTRA=("CMAKE_C_FLAGS=$F" "CMAKE_CXX_FLAGS=$F" "CMAKE_ASM_FLAGS=$F")
elif [ "$ARCH" = x86_64 ]; then
  SD=$R/prplos-5.1-staging; S=$SD/target-x86_64_musl
  T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin/x86_64-openwrt-linux-musl-; EXTRA=()
else
  echo "arch must be x86_64 or aarch64" >&2; exit 2
fi
export STAGING_DIR=$SD
if ls -d "$S"/usr/include/google/protobuf "$S"/usr/lib/libprotobuf.* >/dev/null 2>&1; then
  echo "protobuf is in the sysroot $S; move it out first (see the comment above)" >&2; exit 1
fi
[ "$(cat "$SRC/VERSION_NUMBER")" = 1.30.0 ] || { echo "$SRC is not ONNX Runtime 1.30.0" >&2; exit 1; }
git -C "$SRC" apply --reverse --check "$HERE/patches/ort-minimal-keep-x86-dispatch.patch" ||
  { echo "apply patches/ort-minimal-keep-x86-dispatch.patch to $SRC first" >&2; exit 1; }
if [ "$OPS" = full ]; then
  MODE=()
else
  MODE=(--minimal_build --include_ops_by_config "$(readlink -f "$OPS")" --enable_reduced_operator_type_support --disable_ml_ops)
fi
rm -rf "$B"
cd "$SRC"
"$R/tflite-venv/bin/python" tools/ci_build/build.py --build_dir "$B" --config Release --parallel "$JOBS" \
  --skip_tests --skip_submodule_sync --build_shared_lib "${MODE[@]}" \
  --cmake_generator Ninja --compile_no_warning_as_error \
  --cmake_extra_defines \
    CMAKE_SYSTEM_NAME=Linux CMAKE_SYSTEM_PROCESSOR=$ARCH \
    CMAKE_C_COMPILER=${T}gcc CMAKE_CXX_COMPILER=${T}g++ "${EXTRA[@]}" \
    CMAKE_SYSROOT=$S CMAKE_FIND_ROOT_PATH=$S \
    CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_TELEMETRY=OFF > "$B.log" 2>&1
LIB=$(find "$B/Release" -maxdepth 1 -name "libonnxruntime.so.*.*" -type f -print -quit)
ls -la "$LIB"
"${T}readelf" -d "$LIB" | grep NEEDED
