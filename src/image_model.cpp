#include "image_model.h"

#include "counting_semaphore.h"
#include "errors.h"
#include "image_manifest.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr auto kSlotWait = std::chrono::seconds(30);

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string shape_string(const std::vector<int64_t>& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
}

std::vector<float> hwc_to_chw(const std::vector<float>& hwc, int h, int w) {
    const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
    std::vector<float> chw(hwc.size());
    for (size_t p = 0; p < plane; ++p) {
        for (size_t c = 0; c < 3; ++c) chw[c * plane + p] = hwc[p * 3 + c];
    }
    return chw;
}

}  // namespace

struct ImageModel::Impl {
    ImageManifest manifest;
    std::vector<std::string> labels;
    ImageServeOptions options;
    Ort::Env env;
    Ort::Session session{nullptr};
    std::string input_name;
    std::string output_name;
    std::array<int64_t, 4> input_shape{};
    int in_h = 0;
    int in_w = 0;
    CountingSemaphore decode_slots;
    std::mutex run_mutex;

    Impl(const fs::path& dir, int threads, const ImageServeOptions& opts, bool verbose)
        : manifest(load_image_manifest(dir)),
          labels(load_labels(dir / manifest.labels_file)),
          options(opts),
          env(ORT_LOGGING_LEVEL_WARNING, "ort-server"),
          decode_slots(opts.max_concurrent_decodes) {
        Ort::SessionOptions so;
        if (threads > 0) {
            so.SetIntraOpNumThreads(threads);
            so.SetInterOpNumThreads(1);
        } else {
            so.SetIntraOpNumThreads(0);
        }
        fs::path model_file = dir / "model.onnx";
        if (fs::exists(dir / "model.ort")) {
            model_file = dir / "model.ort";
            so.AddConfigEntry("session.load_model_format", "ORT");
        }
        session = Ort::Session(env, model_file.c_str(), so);

        if (session.GetInputCount() != 1) {
            throw std::runtime_error("image models must have exactly one input; this one has " +
                                     std::to_string(session.GetInputCount()));
        }
        if (session.GetOutputCount() != 1) {
            throw std::runtime_error("image models must have exactly one output; this one has " +
                                     std::to_string(session.GetOutputCount()));
        }
        Ort::AllocatorWithDefaultOptions alloc;
        input_name = session.GetInputNameAllocated(0, alloc).get();
        output_name = session.GetOutputNameAllocated(0, alloc).get();

        // The shape info is a view into its TypeInfo, which must outlive it.
        const Ort::TypeInfo in_info = session.GetInputTypeInfo(0);
        const auto it = in_info.GetTensorTypeAndShapeInfo();
        const auto et = it.GetElementType();
        if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
            et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
            throw std::runtime_error("quantized-input models not supported yet");
        }
        if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("model input must be float32");
        const std::vector<int64_t> dims = it.GetShape();
        const bool nchw = manifest.layout == TensorLayout::NCHW;
        const bool rank4 = dims.size() == 4 && dims[0] == 1 &&
                           std::all_of(dims.begin(), dims.end(), [](int64_t d) { return d >= 1; });
        if (!nchw && rank4 && dims[1] == 3 && dims[3] != 3) {
            throw std::runtime_error("model input is " + shape_string(dims) +
                                     " (NCHW); set preprocess.layout to \"NCHW\" in manifest.json");
        }
        if (nchw && rank4 && dims[3] == 3 && dims[1] != 3) {
            throw std::runtime_error("model input is " + shape_string(dims) +
                                     " (NHWC) but manifest.json sets preprocess.layout \"NCHW\"");
        }
        if (!rank4 || (nchw ? dims[1] : dims[3]) != 3) {
            throw std::runtime_error(std::string("model input must be ") +
                                     (nchw ? "[1, 3, height, width]" : "[1, height, width, 3]") + "; it is " +
                                     shape_string(dims));
        }
        in_h = static_cast<int>(nchw ? dims[2] : dims[1]);
        in_w = static_cast<int>(nchw ? dims[3] : dims[2]);
        std::copy(dims.begin(), dims.end(), input_shape.begin());

        const Ort::TypeInfo out_info = session.GetOutputTypeInfo(0);
        const auto ot = out_info.GetTensorTypeAndShapeInfo();
        if (ot.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("model output must be float32");
        }
        size_t n = 1;
        for (auto d : ot.GetShape()) n *= static_cast<size_t>(std::max<int64_t>(d, 0));
        if (n != labels.size()) {
            throw std::runtime_error("model output has " + std::to_string(n) + " scores but " +
                                     manifest.labels_file + " has " + std::to_string(labels.size()) +
                                     " labels");
        }
        if (verbose) {
            std::fprintf(stderr,
                         "ort-server: image-classification, input %dx%d %s, %zu labels, %d thread(s), "
                         "%d decode slot(s)\n",
                         in_w, in_h, nchw ? "NCHW" : "NHWC", labels.size(), threads,
                         opts.max_concurrent_decodes);
        }
    }

    json classify(std::string_view bytes, int top_k) {
        if (bytes.size() > options.max_image_bytes) {
            throw PayloadTooLarge("image is " + std::to_string(bytes.size()) + " bytes; the limit is " +
                                  std::to_string(options.max_image_bytes));
        }
        if (bytes.empty()) throw InvalidInput("image is empty");
        if (imgproc::sniff_image_format(bytes) == imgproc::ImageFormat::Unknown) {
            throw InvalidInput("unsupported image format (JPEG or PNG)");
        }

        double decode_ms = 0, preprocess_ms = 0, inference_ms = 0;
        int src_w = 0, src_h = 0;
        std::vector<float> tensor;
        {
            if (!decode_slots.acquire_for(kSlotWait)) {
                throw Busy("busy: no image decode slot became free within 30 s");
            }
            SlotGuard slot(decode_slots);
            auto t0 = std::chrono::steady_clock::now();
            imgproc::DecodeResult d = imgproc::decode_rgb8(bytes, options.limits);
            decode_ms = ms_since(t0);
            if (d.error != imgproc::DecodeError::None) throw InvalidInput(d.message);
            src_w = d.image.width;
            src_h = d.image.height;
            t0 = std::chrono::steady_clock::now();
            tensor = imgproc::resample_triangle(d.image.pixels.get(), src_w, src_h, 3, in_w, in_h);
            d.image.pixels.reset();
            imgproc::normalize_in_place(tensor, manifest.mean, manifest.std);
            if (manifest.layout == TensorLayout::NCHW) tensor = hwc_to_chw(tensor, in_h, in_w);
            preprocess_ms = ms_since(t0);
        }

        std::vector<float> scores(labels.size());
        {
            std::lock_guard<std::mutex> lock(run_mutex);
            const auto t0 = std::chrono::steady_clock::now();
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(),
                                                               input_shape.data(), input_shape.size());
            const char* in_names[] = {input_name.c_str()};
            const char* out_names[] = {output_name.c_str()};
            auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
            const auto info = outputs[0].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                info.GetElementCount() != scores.size()) {
                throw std::runtime_error("cannot read the model output");
            }
            const float* out = outputs[0].GetTensorData<float>();
            std::copy(out, out + scores.size(), scores.begin());
            inference_ms = ms_since(t0);
        }
        std::vector<float>().swap(tensor);

        // score_normalization "none": the model must already emit probabilities.
        for (float s : scores) {
            if (!(s >= -1e-6f && s <= 1.0f + 1e-6f)) {
                throw std::runtime_error("model output is not probabilities");
            }
        }

        const size_t n = labels.size();
        size_t k = static_cast<size_t>(top_k > 0 ? top_k : manifest.top_k_default);
        k = std::min(k, n);
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                          [&](size_t a, size_t b) {
                              return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
                          });

        json predictions = json::array();
        json by_label = json::object();
        for (size_t i = 0; i < k; ++i) {
            const size_t idx = order[i];
            predictions.push_back({{"index", idx}, {"label", labels[idx]}, {"score", scores[idx]}});
            // ImageNet repeats some names ("crane", "maillot"); keep the higher score.
            if (!by_label.contains(labels[idx])) by_label[labels[idx]] = scores[idx];
        }
        return json{{"predictions", predictions},
                    {"labels", by_label},
                    {"input", {{"width", src_w}, {"height", src_h}}},
                    {"timings",
                     {{"decode_ms", decode_ms}, {"preprocess_ms", preprocess_ms}, {"inference_ms", inference_ms}}}};
    }
};

ImageModel::ImageModel(const fs::path& dir, int threads, const ImageServeOptions& options, bool verbose)
    : impl_(std::make_unique<Impl>(dir, threads, options, verbose)) {}

ImageModel::~ImageModel() = default;

json ImageModel::classify(std::string_view image_bytes, int top_k) {
    return impl_->classify(image_bytes, top_k);
}
