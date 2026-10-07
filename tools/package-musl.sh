#!/bin/bash
# Package an ort-server musl release tarball (prplOS 5.1), on the build host.
# Usage: tools/package-musl.sh <version> <x86_64|aarch64> <flavour>
#   BLD       the ort-server build tree (default build-prplos-<arch>)
#   ORT_ROOT  the ONNX Runtime root it was linked against (include/, lib/libonnxruntime.so.1.30.0)
#   ORT_SRC   the ONNX Runtime 1.30.0 source it was built from (licence and third-party notices)
#   OPS       the operator config the library was built with (default ops/curated.required_operators.config)
#   OUT       output directory (default $R/bundle)
# <flavour> names the operator set in the archive name, e.g. curated.
set -euo pipefail
VER=$1; ARCH=$2; FLAVOUR=$3
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$0")/.." && pwd)
case "$ARCH" in
x86_64)
  SD=$R/prplos-5.1-staging; T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin/x86_64-openwrt-linux-musl-
  TOKDIR=$R/tokenizers-cpp/rust/target/x86_64-unknown-linux-musl/release
  HOST="prplOS 5.1 x86_64 (musl, gcc 13.3.0)" ;;
aarch64)
  SD=$R/prplos-5.1-staging-aarch64; T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin/aarch64-openwrt-linux-musl-
  TOKDIR=$R/aarch64/tokenizers-target/aarch64-unknown-linux-musl/release
  HOST="prplOS 5.1 aarch64 cortex-a53 (musl, gcc 13.3.0, -march=armv8-a+crc -mtune=cortex-a53)" ;;
*) echo "usage: $0 <version> <x86_64|aarch64> <flavour>" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
BLD=${BLD:-$SRC/build-prplos-$ARCH}
ORT_ROOT=${ORT_ROOT:?set ORT_ROOT to the ONNX Runtime root ort-server was linked against}
ORT_SRC=${ORT_SRC:-$R/onnxruntime-src}
OPS=${OPS:-$SRC/ops/curated.required_operators.config}
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
cp "$ORT_ROOT/lib/libonnxruntime.so.1.30.0" "$B/bin/libonnxruntime.so.1"
patchelf --set-rpath '$ORIGIN' "$B/bin/ort-server"
need "$SRC/LICENSE" LICENSE.ort-server
need "$ORT_SRC/LICENSE" onnxruntime-LICENSE
need "$ORT_SRC/ThirdPartyNotices.txt" onnxruntime-ThirdPartyNotices.txt
need "$R/tokenizers-cpp/LICENSE" tokenizers-cpp-LICENSE
need "$BLD/_deps/httplib-src/LICENSE" cpp-httplib-LICENSE
need "$BLD/_deps/json-src/LICENSE.MIT" nlohmann-json-LICENSE
need "$SRC/third_party/stb/LICENSE" stb-LICENSE
# The Rust crates of the tokenizer shim (incl. HuggingFace tokenizers, onig, Oniguruma), per target.
python3 "$SRC/tools/rust_licenses.py" "$R/tokenizers-cpp/rust" "$ARCH-unknown-linux-musl" "$L" ort-server \
  --cargo "$HOME/.cargo/bin/cargo"
RUSTC=$(strings "$TOKDIR/libtokenizers_c.a" | grep -o -m1 'rustc version [0-9][^ )]*' || true)
RD=$(ls -d "$HOME"/.rustup/toolchains/stable-*/share/doc/rust/licenses | head -1)
mkdir -p "$L/rust-std"
for f in Apache-2.0 MIT LLVM-exception Unicode-3.0; do need "$RD/$f.txt" "rust-std/$f.txt"; done
echo "The Rust standard library (${RUSTC:-rustc}) linked into the tokenizer shim: MIT OR Apache-2.0; its bundled libunwind: Apache-2.0 WITH LLVM-exception; Unicode data: Unicode-3.0." > "$L/rust-std/README"
cp "$OPS" "$B/required_operators.config"
cp "$SRC/patches/ort-minimal-keep-x86-dispatch.patch" "$B/patches/"
{
  echo "ort-server $VER for $HOST, $FLAVOUR minimal ONNX Runtime."
  echo "Source: https://github.com/ianbmacdonald/ort-server @ $(git -C "$SRC" rev-parse --short HEAD) (branch $(git -C "$SRC" branch --show-current))"
  echo "ONNX Runtime: 1.30.0 ($(git -C "$ORT_SRC" rev-parse --short HEAD)) --minimal_build, --enable_reduced_operator_type_support,"
  echo "  --disable_ml_ops, telemetry off, built by tools/ort-musl-build.sh; operators = required_operators.config"
  echo "  ($(basename "$OPS"), $(grep -v '^#' "$OPS" | cut -d';' -f3 | tr ',' '\n' | grep -c .) operator entries)."
  if [ "$FLAVOUR" = curated ]; then
    echo "  Curated: the union over the six use cases of the gateway study (text classification, sentence"
    echo "  embeddings, image classification incl. MobileNetV2, object detection, audio, time series; 17 models)"
    echo "  and the v0.1.x BERT-family set (BERT, DistilBERT, RoBERTa [covers XLM-RoBERTa, CamemBERT], DeBERTa v1/v2,"
    echo "  ELECTRA, ALBERT, ModernBERT; sequence and token classification). A model needing another operator"
    echo "  fails to load. tools/ort-musl-build.sh <arch> full <dir> is the general build."
  fi
  if [ "$ARCH" = x86_64 ]; then
    echo "Patched (patches/ort-minimal-keep-x86-dispatch.patch): the minimal build keeps MLAS runtime x86 dispatch"
    echo "  (AVX-512F/Core/VNNI, AVX-VNNI, AMX), which upstream compiles out of ORT_MINIMAL_BUILD."
  else
    echo "patches/ort-minimal-keep-x86-dispatch.patch is applied, as in the x86_64 release, and is a no-op on aarch64"
    echo "  (its hunks are inside MLAS_TARGET_AMD64). The MLAS ARM64 runtime dispatch is built in; Cortex-A53 runs the"
    echo "  NEON kernels."
  fi
  echo "ort-server linked with --gc-sections (ORT_SERVER_GC_SECTIONS=$(grep -E '^ORT_SERVER_GC_SECTIONS:' "$BLD/CMakeCache.txt" | cut -d= -f2-))."
  echo "Minimal ONNX Runtime loads ORT-format models only: put model.ort in the model directory, made with"
  echo "  python -m onnxruntime.tools.convert_onnx_models_to_ort <dir-with-model.onnx> (onnxruntime 1.30)."
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
