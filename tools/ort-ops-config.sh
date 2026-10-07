#!/bin/bash
# tools/ort-ops-config.sh <dir-with-model.onnx-files> : convert every .onnx in the directory to ORT format with
# runtime optimisations (what ort-server loads as model.ort) and write the union operator config,
# <dir>/required_operators.with_runtime_opt.config, for tools/ort-musl-build.sh. Needs onnxruntime 1.30 in python.
set -euo pipefail
PY=${PY:-python3}
"$PY" -m onnxruntime.tools.convert_onnx_models_to_ort "$1" --optimization_style Runtime
cat "$1/required_operators.with_runtime_opt.config"
