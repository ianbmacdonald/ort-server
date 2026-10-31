#!/bin/bash
# Package an ort-server musl release tarball (prplOS 5.1), on the build host.
# Usage: tools/package-musl.sh <version> <x86_64|aarch64> <flavour>
#   BLD       the ort-server build tree (default build-prplos-<arch>)
#   ORT_ROOT  the ONNX Runtime root it was linked against (default $BLD/ort-root, from tools/ort-server-musl-build.sh)
#   ORT_SRC   the ONNX Runtime source it was built from (licence and third-party notices)
#   OPS       the operator config the library was built with (default ops/default.required_operators.config;
#             "full" for a full build)
#   OUT       output directory (default $R/bundle)
# <flavour> names the operator set in the archive name, e.g. default or full.
set -euo pipefail
VER=$1; ARCH=$2; FLAVOUR=$3
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$0")/.." && pwd)
case "$ARCH" in
x86_64)
  SD=$R/prplos-5.1-staging; T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin/x86_64-openwrt-linux-musl-
  HOST="prplOS 5.1 x86_64 (musl, gcc 13.3.0)" ;;
aarch64)
  SD=$R/prplos-5.1-staging-aarch64; T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin/aarch64-openwrt-linux-musl-
  HOST="prplOS 5.1 aarch64 cortex-a53 (musl, gcc 13.3.0, -march=armv8-a+crc -mtune=cortex-a53)" ;;
*) echo "usage: $0 <version> <x86_64|aarch64> <flavour>" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
BLD=${BLD:-$SRC/build-prplos-$ARCH}
ORT_ROOT=${ORT_ROOT:-$BLD/ort-root}
ORT_SRC=${ORT_SRC:-$R/onnxruntime-src}
OPS=${OPS:-$SRC/ops/default.required_operators.config}
TRIPLE=$ARCH-unknown-linux-musl
TOKLIB=$BLD/tok_ffi/$TRIPLE/release/libtok_ffi.a
ORTLIB=$(cd "$ORT_ROOT/lib" && ls libonnxruntime.so.*.* | head -1)
ORTV=${ORTLIB#libonnxruntime.so.}
NAME=ort-server-musl-$ARCH-$FLAVOUR-$VER
OUT=${OUT:-$R/bundle}
B=$OUT/$NAME
L=$B/licenses
need() {  # need <src> <dest-name>: licence texts are not optional
  [ -f "$1" ] || { echo "missing licence text $1" >&2; exit 1; }
  cp "$1" "$L/$2"
}

rm -rf "$B" && mkdir -p "$B/bin" "$L" "$B/patches"
cp "$BLD/ort-server.stripped" "$B/bin/ort-server"
cp "$ORT_ROOT/lib/$ORTLIB" "$B/bin/libonnxruntime.so.1"
patchelf --set-rpath '$ORIGIN' "$B/bin/ort-server"
need "$SRC/LICENSE" LICENSE.ort-server
need "$ORT_SRC/LICENSE" onnxruntime-LICENSE
need "$ORT_SRC/ThirdPartyNotices.txt" onnxruntime-ThirdPartyNotices.txt
need "$BLD/_deps/httplib-src/LICENSE" cpp-httplib-LICENSE
need "$BLD/_deps/json-src/LICENSE.MIT" nlohmann-json-LICENSE
need "$SRC/third_party/stb/LICENSE" stb-LICENSE
# The Rust crates of third_party/tok_ffi (incl. HuggingFace tokenizers, onig, Oniguruma), per target.
python3 "$SRC/tools/rust_licenses.py" "$SRC/third_party/tok_ffi" "$TRIPLE" "$L" ort-server \
  --cargo "$HOME/.cargo/bin/cargo"
RUSTC=$(strings "$TOKLIB" | grep -o -m1 'rustc version [0-9][^ )]*' || true)
RD=$(ls -d "$HOME"/.rustup/toolchains/stable-*/share/doc/rust/licenses | head -1)
mkdir -p "$L/rust-std"
for f in Apache-2.0 MIT LLVM-exception Unicode-3.0; do need "$RD/$f.txt" "rust-std/$f.txt"; done
echo "The Rust standard library (${RUSTC:-rustc}) linked into third_party/tok_ffi: MIT OR Apache-2.0; its bundled libunwind: Apache-2.0 WITH LLVM-exception; Unicode data: Unicode-3.0." > "$L/rust-std/README"
[ "$OPS" = full ] || cp "$OPS" "$B/required_operators.config"
cp "$SRC"/patches/musl/onnxruntime/*.patch "$SRC/patches/musl/README.md" "$B/patches/"
{
  echo "ort-server $VER for $HOST, $FLAVOUR ONNX Runtime $ORTV."
  echo "Source: https://github.com/ianbmacdonald/ort-server @ $(git -C "$SRC" rev-parse --short HEAD) (branch $(git -C "$SRC" branch --show-current))"
  if [ "$OPS" = full ]; then
    echo "ONNX Runtime: $ORTV ($(git -C "$ORT_SRC" rev-parse --short HEAD)) full build (every ONNX and contrib operator, ML ops;"
    echo "  loads .onnx and .ort), telemetry off, built by tools/ort-musl-build.sh <arch> full."
  else
    echo "ONNX Runtime: $ORTV ($(git -C "$ORT_SRC" rev-parse --short HEAD)) --minimal_build, --enable_reduced_operator_type_support,"
    echo "  --disable_ml_ops, telemetry off, built by tools/ort-musl-build.sh; operators = required_operators.config"
    echo "  ($(basename "$OPS"), $(grep -v '^#' "$OPS" | cut -d';' -f3 | tr ',' '\n' | grep -c .) operator entries)."
    echo "  A model needing another operator fails to load and ONNX Runtime names the missing kernel."
    echo "Minimal ONNX Runtime loads ORT-format models only: put model.ort in the model directory, made with"
    echo "  python -m onnxruntime.tools.convert_onnx_models_to_ort <dir> --optimization_style Runtime (onnxruntime $ORTV)."
  fi
  if [ "$ARCH" = x86_64 ]; then
    echo "Patched (patches/0001-mlas-keep-the-x86-ISA-runtime-dispatch-in-ORT_MINIMA.patch): a minimal build keeps"
    echo "  MLAS runtime x86 dispatch (AVX-512F/Core/VNNI, AVX-VNNI, AMX), which upstream compiles out of ORT_MINIMAL_BUILD."
  else
    echo "The patches in patches/ are applied, as in the x86_64 release; the MLAS one is a no-op on aarch64 (its hunks are"
    echo "  inside MLAS_TARGET_AMD64). The MLAS ARM64 runtime dispatch is built in; Cortex-A53 runs the NEON kernels."
  fi
  echo "ort-server linked with --gc-sections (ORT_SERVER_GC_SECTIONS=$(grep -E '^ORT_SERVER_GC_SECTIONS:' "$BLD/CMakeCache.txt" | cut -d= -f2-))."
  echo "Runtime libraries (not bundled, from the prplOS image):"
  "${T}readelf" -d "$B/bin/ort-server" | grep -o 'NEEDED.*\[.*\]' | grep -o '\[[^]]*\]' | sed 's/^/  /'
} > "$B/BUILD-INFO.txt"
chmod -R go-w "$B"

cd "$OUT"
tar czf "$NAME.tar.gz" "$NAME"
sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
echo "$NAME.tar.gz $(stat -c %s "$NAME.tar.gz") bytes sha256 $(cut -c1-64 "$NAME.tar.gz.sha256")"
cat "$B/BUILD-INFO.txt"
ls "$L"
