"""Generate tiny-image: an image-classification model dir for /classify/image.

The model takes float32 NCHW [1, 3, 4, 4] (as a PyTorch export does) and returns
softmax over the three per-channel means, so its scores are computable by hand
from the input pixels: test/smoke.py checks them exactly, which also proves the
HWC -> CHW transpose (a wrong one mixes the channels' means).

    python test/fixtures/make_tiny_image.py   # needs the onnx package
"""

import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).parent
dst = HERE / "tiny-image"
dst.mkdir(exist_ok=True)

graph = helper.make_graph(
    [
        helper.make_node("ReduceMean", ["input"], ["means"], axes=[2, 3], keepdims=0),
        helper.make_node("Softmax", ["means"], ["probs"], axis=1),
    ],
    "tiny-image",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
    [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, dst / "model.onnx")
(dst / "labels.txt").write_text("red\ngreen\nblue\n")
manifest = {
    "task": "image-classification",
    "labels_file": "labels.txt",
    "score_normalization": "none",
    "top_k_default": 2,
    "preprocess": {"resize": "stretch", "mean": 127.5, "std": 127.5, "layout": "NCHW"},
}
(dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("tiny-image written (NCHW [1,3,4,4] -> softmax of channel means)")
