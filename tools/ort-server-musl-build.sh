#!/bin/bash
# Build ort-server for prplOS 5.1 musl against a musl ONNX Runtime (tools/ort-musl-build.sh), on the ai4
# build host layout.
#
#   tools/ort-server-musl-build.sh <x86_64|aarch64> <ort-build-dir> <build-dir> [<ort-version-label>]
#
# <ort-build-dir> is tools/ort-musl-build.sh's build dir (or an ORT root with lib/); its libonnxruntime.so.<ver> and the source's
# C API headers ($ORT_SRC) become <build-dir>/ort-root (what ORT_SERVER_ORT_ROOT takes). third_party/tok_ffi is
# built by cargo for the Rust musl target (ORT_SERVER_TOK_FFI_TARGET); its C build scripts (onig_sys) take the
# prplOS cross compiler from CC_<triple>/AR_<triple>. Leaves <build-dir>/ort-server and ort-server.stripped.
# Offline: see DEPENDENCIES.md (FETCHCONTENT_SOURCE_DIR_HTTPLIB/_JSON, cargo vendor + CARGO_NET_OFFLINE).
set -euo pipefail
ARCH=$1; ORTB=$(readlink -f "$2"); B=$(readlink -m "$3"); LABEL=${4:-}
R=${R:-$HOME/build-litert}; ORT_SRC=${ORT_SRC:-$R/onnxruntime-src}; JOBS=${JOBS:-8}
SRC=$(cd "$(dirname "$0")/.." && pwd)
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
if [ "$ARCH" = aarch64 ]; then
  SD=$R/prplos-5.1-staging-aarch64; S=$SD/target-aarch64_cortex-a53_musl
  T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin/aarch64-openwrt-linux-musl-
  F="-march=armv8-a+crc -mtune=cortex-a53"; TRIPLE=aarch64-unknown-linux-musl
  export CFLAGS_aarch64_unknown_linux_musl="$F"
  export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C target-cpu=cortex-a53"
elif [ "$ARCH" = x86_64 ]; then
  SD=$R/prplos-5.1-staging; S=$SD/target-x86_64_musl
  T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin/x86_64-openwrt-linux-musl-; F=""; TRIPLE=x86_64-unknown-linux-musl
else
  echo "arch must be x86_64 or aarch64" >&2; exit 2
fi
export STAGING_DIR=$SD
U=${TRIPLE//-/_}
export "CC_$U=${T}gcc" "AR_$U=${T}gcc-ar"
LIB=$(find "$ORTB/Release" "$ORTB/lib" -maxdepth 1 -name "libonnxruntime.so.*.*" -type f -print -quit 2>/dev/null || true)
[ -n "$LIB" ] || { echo "no libonnxruntime.so.<ver> in $ORTB/Release or $ORTB/lib" >&2; exit 1; }
V=${LIB##*/libonnxruntime.so.}
rm -rf "$B"; mkdir -p "$B/ort-root/include" "$B/ort-root/lib"
cp "$ORT_SRC"/include/onnxruntime/core/session/*.h "$B/ort-root/include/"
cp "$LIB" "$B/ort-root/lib/"
ln -s "libonnxruntime.so.$V" "$B/ort-root/lib/libonnxruntime.so.1"
ln -s libonnxruntime.so.1 "$B/ort-root/lib/libonnxruntime.so"
cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=$ARCH \
  -DCMAKE_C_COMPILER=${T}gcc -DCMAKE_CXX_COMPILER=${T}g++ -DCMAKE_CXX_FLAGS="$F" \
  -DCMAKE_SYSROOT=$S -DCMAKE_FIND_ROOT_PATH=$S \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DCARGO_EXECUTABLE="$CARGO" \
  -DONNXRUNTIME_VERSION="${LABEL:-$V}" \
  -DORT_SERVER_ORT_ROOT="$B/ort-root" \
  -DORT_SERVER_TOK_FFI_TARGET=$TRIPLE
cmake --build "$B" -j"$JOBS"
"${T}strip" -o "$B/ort-server.stripped" "$B/ort-server"
ls -la "$B/ort-server" "$B/ort-server.stripped" "$B/ort-root/lib/libonnxruntime.so.$V"
"${T}readelf" -d "$B/ort-server" | grep -E "NEEDED|RUNPATH|RPATH"
