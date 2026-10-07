// ort-server — generic ONNX Runtime model server for Lemonade (see README).
//
// A manifest.json with "task": "image-classification" selects the image path
// (POST /classify/image, image_model.cpp, the tflite-server contract); any
// other model directory is the text path below.
//
// v1: CPU EP, text classification. The model is a plain exported ONNX graph
// (input_ids / attention_mask / token_type_ids -> logits). This process loads
// the model + its HF tokenizer (via tokenizers-cpp), derives the output
// contract from an optional manifest.json (or infers it from the export's own
// config.json), tokenizes the request at /classify, runs the session, and
// shapes the output (normalize for sequence-classification, per-token
// aggregation for token-classification).

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

// tokenizers-cpp loads the model's own HF tokenizer.json and tokenizes in
// process. The C API is used directly: the C++ wrapper's base interface
// hardcodes add_special_tokens=false, but encoder classifiers need [CLS]/[SEP]
// to match the HuggingFace reference they were validated against.
#include "tokenizers_c.h"

#include "counting_semaphore.h"
#include "errors.h"
#include "flat_json.h"
#include "image_model.h"
#include "image_preprocess.h"

#include <mutex>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Manifest {
    std::string task;                   // "text-classification" | "token-classification"
    std::vector<std::string> id2label;  // index -> label
    std::string score_normalization = "softmax";  // "softmax" | "sigmoid"
    std::string token_aggregation = "max";        // token-cls only; "max" | "mean"
    int max_length = 512;               // token budget; longer inputs are truncated
};

constexpr long long kMaxTopK = 1000000;
constexpr auto kAdmissionWait = std::chrono::seconds(30);
constexpr auto kSocketTimeout = std::chrono::seconds(5);

struct Args {
    std::string model_path;
    int port = 0;
    int threads = 0;  // 0: ONNX Runtime sizes (and pins) its own pool
    bool verbose = false;
    uint64_t max_image_bytes = 16u << 20;
    uint64_t max_image_pixels = 4000000;
    uint64_t decode_budget_factor = 16;
    uint64_t max_decode_bytes = 256u << 20;
    int max_concurrent_decodes = 1;
    int http_threads = 4;
};

const char* kUsage =
    "usage: ort-server --model-path <dir> --port <n> [--threads N] [--verbose]\n"
    "  image models: [--max-image-bytes N] [--max-image-pixels N] [--max-concurrent-decodes 1..2]\n"
    "                [--decode-budget-factor N] [--max-decode-bytes N] [--http-threads 2..16]";

// The range is checked on the parsed long long, before any narrowing cast.
long long parse_int(const std::string& flag, const char* v, long long lo, long long hi) {
    size_t pos = 0;
    long long n = 0;
    try {
        n = std::stoll(v, &pos);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos == 0 || v[pos] != '\0') throw std::runtime_error(flag + " needs an integer\n" + kUsage);
    if (n < lo || n > hi) {
        throw std::runtime_error(flag + " must be from " + std::to_string(lo) + " to " + std::to_string(hi));
    }
    return n;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        const bool has_value = i + 1 < argc;
        if (f == "--model-path" && has_value) a.model_path = argv[++i];
        else if (f == "--port" && has_value) a.port = static_cast<int>(parse_int(f, argv[++i], 1, 65535));
        else if (f == "--threads" && has_value) a.threads = static_cast<int>(parse_int(f, argv[++i], 0, 1024));
        else if (f == "--verbose") a.verbose = true;
        else if (f == "--max-image-bytes" && has_value) a.max_image_bytes = parse_int(f, argv[++i], 1024, 256ll << 20);
        else if (f == "--max-image-pixels" && has_value) {
            a.max_image_pixels = parse_int(f, argv[++i], 1, 16384ll * 16384);
        } else if (f == "--decode-budget-factor" && has_value) {
            a.decode_budget_factor = parse_int(f, argv[++i], 4, 64);
        } else if (f == "--max-decode-bytes" && has_value) {
            a.max_decode_bytes = parse_int(f, argv[++i], 16ll << 20, 4ll << 30);
        } else if (f == "--max-concurrent-decodes" && has_value) {
            a.max_concurrent_decodes = static_cast<int>(parse_int(f, argv[++i], 1, 2));
        } else if (f == "--http-threads" && has_value) {
            a.http_threads = static_cast<int>(parse_int(f, argv[++i], 2, 16));
        }
    }
    if (a.model_path.empty() || a.port == 0) throw std::runtime_error(kUsage);
    return a;
}

void parse_id2label(const json& id2label, Manifest& m, const std::string& origin) {
    if (!id2label.is_object() || id2label.empty()) {
        throw std::runtime_error("id2label is missing or empty in " + origin);
    }
    m.id2label.resize(id2label.size());
    std::vector<bool> seen(id2label.size(), false);
    for (auto it = id2label.begin(); it != id2label.end(); ++it) {
        size_t pos = 0;
        unsigned long idx = 0;
        try {
            idx = std::stoul(it.key(), &pos);
        } catch (const std::exception&) {
            pos = 0;
        }
        if (pos != it.key().size()) {
            throw std::runtime_error("id2label key '" + it.key() + "' is not an index in " + origin);
        }
        if (idx >= m.id2label.size() || seen[idx]) {
            throw std::runtime_error("id2label keys must be unique and contiguous 0..n-1 in " + origin);
        }
        if (!it.value().is_string()) {
            throw std::runtime_error("id2label values must be strings in " + origin);
        }
        seen[idx] = true;
        m.id2label[idx] = it.value().get<std::string>();
    }
}

json read_json_if_present(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return json::object();
    try {
        json j; f >> j;
        return j.is_object() ? j : json::object();
    } catch (const std::exception&) {
        return json::object();
    }
}

// This server feeds the model a single sequence with a fabricated all-ones
// attention mask and all-zero token_type_ids, and truncates by keeping the
// trailing token. That is exactly right for BERT-family single-sequence
// encoders and wrong for architectures with different segment/special-token
// conventions (XLNet puts its classifier token last with a distinct segment
// id), so the supported set is an explicit allowlist rather than a claim.
const std::set<std::string>& supported_model_types() {
    static const std::set<std::string> kSupported = {
        "albert", "bert",     "camembert",  "deberta", "deberta-v2",
        "distilbert", "electra", "roberta", "xlm-roberta", "modernbert",
        "openai_privacy_filter", "pii_masking"
    };
    return kSupported;
}

void validate_model_family(const json& config, const fs::path& dir) {
    // A manifest describes the OUTPUT contract (labels, normalization). It says
    // nothing about the INPUT convention — attention mask, segment ids, special
    // tokens — which is what this server hardcodes. So a manifest cannot excuse
    // a missing config.json: without it we cannot know the architecture, and an
    // unchecked one would be served with a fabricated mask that may not fit.
    if (config.empty()) {
        throw std::runtime_error(
            "config.json is missing or unreadable in " + dir.string() +
            ". It is required (even alongside a manifest.json) to confirm the "
            "model uses the single-sequence encoder convention this server "
            "implements.");
    }
    std::string model_type;
    if (config.contains("model_type") && config["model_type"].is_string()) {
        model_type = config["model_type"].get<std::string>();
    }
    if (model_type.empty()) {
        throw std::runtime_error("config.json in " + dir.string() +
                                 " declares no model_type; cannot verify that this "
                                 "architecture uses the single-sequence encoder "
                                 "convention ort-server implements");
    }
    if (!supported_model_types().count(model_type)) {
        std::string supported;
        for (const auto& t : supported_model_types()) {
            supported += (supported.empty() ? "" : ", ") + t;
        }
        throw std::runtime_error(
            "unsupported model_type '" + model_type +
            "'. ort-server implements the single-sequence encoder convention "
            "(all-ones attention mask, all-zero token_type_ids, trailing-token "
            "truncation), which is valid for: " + supported +
            ". Other architectures need their own mask/segment handling.");
    }
}

// max_length precedence mirrors the exporter: the tokenizer's declared budget,
// then the model's position table (less 2 — RoBERTa-family configs declare
// max_position_embeddings larger than the usable budget), then 512.
void apply_inferred_max_length(const json& tokenizer_config, const json& config, Manifest& m) {
    auto valid = [](const json& j, const char* key) -> int {
        // HF writes a huge sentinel (1e30) when the tokenizer has no real limit;
        // that parses as a double and is skipped by the integer check.
        if (!j.contains(key) || !j[key].is_number_integer()) return 0;
        auto n = j[key].get<long long>();
        return (n >= 2 && n <= 1000000) ? static_cast<int>(n) : 0;
    };
    if (int n = valid(tokenizer_config, "model_max_length")) {
        m.max_length = n;
        return;
    }
    if (int n = valid(config, "max_position_embeddings")) {
        m.max_length = n > 4 ? n - 2 : n;
        return;
    }
    m.max_length = 512;
}

Manifest manifest_from_json(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json");
    if (!f) throw std::runtime_error("cannot open manifest.json in " + dir.string());
    json j; f >> j;
    Manifest m;
    // Even with an explicit manifest, the tokenization/mask conventions below
    // still have to hold for this architecture.
    validate_model_family(read_json_if_present(dir / "config.json"), dir);
    m.task = j.at("task").get<std::string>();
    if (m.task != "text-classification" && m.task != "token-classification") {
        throw std::runtime_error("unsupported task in manifest.json: '" + m.task +
                                 "' (expected text-classification or token-classification)");
    }
    // Wrong-typed values are errors, not silent fallbacks to defaults.
    if (j.contains("score_normalization")) {
        if (!j["score_normalization"].is_string()) {
            throw std::runtime_error("score_normalization must be a string");
        }
        m.score_normalization = j["score_normalization"].get<std::string>();
    }
    // "none" is rejected: the /classify contract promises label scores in [0,1].
    if (m.score_normalization != "softmax" && m.score_normalization != "sigmoid") {
        throw std::runtime_error("unsupported score_normalization: " + m.score_normalization);
    }
    // token_aggregation is null for sequence-classification; tolerate null/absent,
    // but reject unknown values regardless of task.
    if (j.contains("token_aggregation") && !j["token_aggregation"].is_null()) {
        if (!j["token_aggregation"].is_string()) {
            throw std::runtime_error("token_aggregation must be a string or null");
        }
        m.token_aggregation = j["token_aggregation"].get<std::string>();
        if (m.token_aggregation != "max" && m.token_aggregation != "mean") {
            throw std::runtime_error("unsupported token_aggregation: " + m.token_aggregation);
        }
    }
    if (j.contains("max_length")) {
        if (!j["max_length"].is_number_integer()) {
            throw std::runtime_error("max_length must be an integer");
        }
        m.max_length = j["max_length"].get<int>();
        if (m.max_length < 2) throw std::runtime_error("max_length must be >= 2");
    } else {
        // Manifest omits the budget: fall back to the model's own metadata
        // rather than a blanket 512, which can overflow a smaller position table.
        apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"),
                                  read_json_if_present(dir / "config.json"), m);
    }
    parse_id2label(j.at("id2label"), m, "manifest.json");
    return m;
}

// Fallback for a stock HF/Optimum export (no manifest.json): infer the contract
// from config.json (+ tokenizer_config.json), applying HF problem_type semantics.
Manifest manifest_from_hf_config(const fs::path& dir) {
    std::ifstream f(dir / "config.json");
    if (!f) {
        throw std::runtime_error("neither manifest.json nor config.json found in " +
                                 dir.string());
    }
    json j; f >> j;
    Manifest m;
    validate_model_family(j, dir);

    std::string arch;
    if (j.contains("architectures") && j["architectures"].is_array() &&
        !j["architectures"].empty() && j["architectures"][0].is_string()) {
        arch = j["architectures"][0].get<std::string>();
    }
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with(arch, "ForTokenClassification")) {
        m.task = "token-classification";
    } else if (ends_with(arch, "ForSequenceClassification")) {
        m.task = "text-classification";
    } else {
        throw std::runtime_error("cannot infer task from config.json architecture '" +
                                 arch + "'; provide a manifest.json");
    }

    // problem_type is only a training-time hint: models trained with BCE outside
    // the HF Trainer routinely leave it null, so an absent value CANNOT be read
    // as "single-label". Manifest-less inference therefore ASSUMES single-label
    // softmax and says so; a multi-label model must declare it — either via
    // problem_type in its config, or with an explicit manifest.json.
    std::string problem_type;
    if (j.contains("problem_type") && j["problem_type"].is_string()) {
        problem_type = j["problem_type"].get<std::string>();
    }
    if (problem_type == "regression") {
        throw std::runtime_error("regression heads have no label scores in [0,1]");
    }
    if (m.task == "text-classification" && problem_type == "multi_label_classification") {
        m.score_normalization = "sigmoid";
    } else {
        m.score_normalization = "softmax";
        if (m.task == "text-classification" && problem_type.empty()) {
            fprintf(stderr,
                    "ort-server: config.json declares no problem_type; assuming "
                    "SINGLE-LABEL softmax. If this is a multi-label (BCE-trained) "
                    "model, supply a manifest.json with "
                    "\"score_normalization\": \"sigmoid\" — otherwise the scores "
                    "will be wrong.\n");
        }
    }

    parse_id2label(j.at("id2label"), m, "config.json");
    if (m.id2label.size() < 2) {
        throw std::runtime_error("single-output heads have no label scores in [0,1]");
    }
    apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"), j, m);
    return m;
}

// manifest.json (explicit contract, validated strictly) wins; a bare Optimum
// export runs via config.json inference so users need no lemonade tooling.
Manifest load_manifest(const fs::path& dir) {
    if (fs::exists(dir / "manifest.json")) return manifest_from_json(dir);
    return manifest_from_hf_config(dir);
}

std::vector<float> softmax(const float* v, size_t n) {
    float mx = *std::max_element(v, v + n);
    std::vector<float> out(n);
    double sum = 0;
    for (size_t i = 0; i < n; ++i) { out[i] = std::exp(v[i] - mx); sum += out[i]; }
    for (auto& x : out) x = static_cast<float>(x / sum);
    return out;
}

std::vector<float> normalize(const float* v, size_t n, const std::string& mode) {
    if (mode == "softmax") return softmax(v, n);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-v[i]));
    return out;
}

std::string load_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Collect the token ids the post-processor INSERTS ([CLS]/[SEP], <s>/</s>, …).
// The HF tokenizer serializes these three ways, and a "Sequence" can nest them:
//   TemplateProcessing  -> special_tokens[*].ids   (modern BERT/DistilBERT)
//   BertProcessing      -> cls / sep = [token, id]
//   RobertaProcessing   -> cls / sep = [token, id]  (stock RoBERTa; different shape)
// Reading only TemplateProcessing would silently miss RoBERTa's <s>/</s>, leaving
// them in token-classification aggregation that HF's pipeline drops.
void collect_inserted_special_ids(const json& pp, std::set<int64_t>& out) {
    if (!pp.is_object()) return;
    const std::string type = pp.value("type", "");
    if (type == "Sequence" && pp.contains("processors") && pp["processors"].is_array()) {
        for (const auto& child : pp["processors"]) collect_inserted_special_ids(child, out);
        return;
    }
    if (pp.contains("special_tokens") && pp["special_tokens"].is_object()) {
        for (const auto& entry : pp["special_tokens"]) {
            if (!entry.is_object() || !entry.contains("ids")) continue;
            for (const auto& id : entry["ids"]) {
                if (id.is_number_integer()) out.insert(id.get<int64_t>());
            }
        }
    }
    for (const char* field : {"cls", "sep"}) {
        if (pp.contains(field) && pp[field].is_array() && pp[field].size() == 2 &&
            pp[field][1].is_number_integer()) {
            out.insert(pp[field][1].get<int64_t>());
        }
    }
}

class Model {
public:
    Model(const fs::path& dir, int threads, bool verbose)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ort-server"), manifest_(load_manifest(dir)) {
        (void)verbose;
        std::string blob = load_bytes(dir / "tokenizer.json");

        // Parse BEFORE handing the blob to the Rust tokenizer: it unwraps its
        // parse Result, so a truncated or corrupt tokenizer.json (a partial
        // download, say) panics across the FFI and aborts the process with no
        // usable message. Fail cleanly instead.
        json tj;
        try {
            tj = json::parse(blob);
        } catch (const std::exception& e) {
            throw std::runtime_error("tokenizer.json in " + dir.string() +
                                     " is not valid JSON (truncated or corrupt "
                                     "download?): " + e.what());
        }

        tokenizer_ = tokenizers_new_from_str(blob.data(), blob.size());
        if (!tokenizer_) throw std::runtime_error("failed to load tokenizer.json from " + dir.string());

        // Positions the tokenizer INSERTS ([CLS]/[SEP] from the post-processor
        // template, plus padding). They belong in the model's INPUT — sequence
        // classifiers pool [CLS] — but HuggingFace's token-classification
        // pipeline drops them from its OUTPUT, so aggregating them would report
        // entities the reference never does.
        //
        // Deliberately NOT every `added_token` marked special: [UNK] and [MASK]
        // are special *tokens* but appear as ordinary content positions (an
        // out-of-vocabulary word becomes [UNK] and HF still scores it), and
        // HF's special_tokens_mask marks them 0.
        if (tj.contains("post_processor")) {
            collect_inserted_special_ids(tj["post_processor"], special_ids_);
        }

        // A tokenizer.json that carries a `padding` section pads every encoding
        // out to a fixed width — the HuggingFace reference does NOT pad by
        // default, so those trailing [PAD] ids must be dropped. Feeding them to
        // the model (under our all-ones attention mask) makes it attend to
        // padding as if it were text and silently corrupts the scores.
        if (tj.contains("padding") && tj["padding"].is_object() &&
            tj["padding"].contains("pad_id") &&
            tj["padding"]["pad_id"].is_number_integer()) {
            pad_id_ = tj["padding"]["pad_id"].get<int64_t>();
            special_ids_.insert(pad_id_);
        }

        Ort::SessionOptions opts;
        if (threads > 0) {
            // ORT pins intra-op workers to cores only when it sizes the pool
            // itself (intra-op threads == 0); an explicit size keeps the
            // workers inside the caller's taskset/cpuset.
            opts.SetIntraOpNumThreads(threads);
            opts.SetInterOpNumThreads(1);
        } else {
            opts.SetIntraOpNumThreads(0);
        }
        // A minimal ONNX Runtime build (--minimal_build, e.g. for a small musl
        // gateway) loads only the ORT flatbuffer format; a full build loads either.
        // model.ort wins when both are present.
        fs::path model_file = dir / "model.onnx";
        if (fs::exists(dir / "model.ort")) {
            model_file = dir / "model.ort";
            opts.AddConfigEntry("session.load_model_format", "ORT");
        }
        session_ = Ort::Session(env_, model_file.c_str(), opts);

        size_t n_in = session_.GetInputCount();
        for (size_t i = 0; i < n_in; ++i) {
            input_names_.push_back(session_.GetInputNameAllocated(i, alloc_).get());
        }
        size_t n_out = session_.GetOutputCount();
        for (size_t i = 0; i < n_out; ++i) {
            output_names_.push_back(session_.GetOutputNameAllocated(i, alloc_).get());
        }
    }

    ~Model() {
        if (tokenizer_) tokenizers_free(tokenizer_);
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    json classify(const std::string& text, int top_k) {
        // Special tokens ON: encoder classifiers pool [CLS]/use [SEP], and the
        // catalog's parity validation runs the HF tokenizer with them enabled —
        // serving without them would silently diverge from the validated scores.
        // The mutex covers the Rust FFI's &mut self contract; tokenization is
        // microseconds next to the session run, which stays concurrent.
        std::vector<int64_t> input_ids;
        {
            std::lock_guard<std::mutex> lock(tokenizer_mutex_);
            TokenizerEncodeResult result;
            tokenizers_encode(tokenizer_, text.data(), text.size(),
                              /*add_special_token=*/1, &result);
            input_ids.assign(result.token_ids, result.token_ids + result.len);
            tokenizers_free_encode_results(&result, 1);
        }
        // Drop the tokenizer's own padding so the model sees exactly what the
        // HuggingFace reference sees (see pad_id_ in the constructor).
        if (pad_id_ >= 0) {
            while (input_ids.size() > 1 && input_ids.back() == pad_id_) {
                input_ids.pop_back();
            }
        }
        if (input_ids.empty()) throw std::runtime_error("empty tokenization");

        // Truncate to the manifest's token budget, keeping the trailing token
        // (usually [SEP] / </s>) so the sequence stays well-formed.
        const size_t max_len = static_cast<size_t>(manifest_.max_length);
        if (input_ids.size() > max_len) {
            int64_t last = input_ids.back();
            input_ids.resize(max_len - 1);
            input_ids.push_back(last);
        }
        const int64_t seq_len = static_cast<int64_t>(input_ids.size());

        // Standard encoder inputs: attention_mask all-ones, token_type_ids all-zeros.
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> token_type_ids(input_ids.size(), 0);
        const std::array<int64_t, 2> shape{1, seq_len};

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto make = [&](std::vector<int64_t>& data) {
            return Ort::Value::CreateTensor<int64_t>(mem, data.data(), data.size(),
                                                     shape.data(), shape.size());
        };

        // Feed each declared input by name (models differ: DistilBERT/RoBERTa
        // have no token_type_ids; BERT/DeBERTa do).
        std::vector<Ort::Value> inputs;
        std::vector<const char*> in_names;
        for (const auto& name : input_names_) {
            if (name == "input_ids") inputs.push_back(make(input_ids));
            else if (name == "attention_mask") inputs.push_back(make(attention_mask));
            else if (name == "token_type_ids") inputs.push_back(make(token_type_ids));
            else throw std::runtime_error("unexpected model input: " + name);
            in_names.push_back(name.c_str());
        }

        std::vector<const char*> out_names;
        for (const auto& n : output_names_) out_names.push_back(n.c_str());

        auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                    inputs.size(), out_names.data(), out_names.size());

        auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
        if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "model output must be float32 logits (fp16/quantized-output exports are not supported)");
        }
        const float* logits = outputs[0].GetTensorData<float>();
        auto out_shape = type_info.GetShape();
        const size_t num_labels = manifest_.id2label.size();

        // Guard against a model/manifest mismatch before indexing the buffer.
        if (out_shape.empty() || out_shape.back() < 0 ||
            static_cast<size_t>(out_shape.back()) != num_labels) {
            throw std::runtime_error(
                "model output last dimension (" +
                std::to_string(out_shape.empty() ? -1 : out_shape.back()) +
                ") does not match manifest id2label size (" + std::to_string(num_labels) + ")");
        }

        std::map<std::string, float> scores;
        if (manifest_.task == "token-classification") {
            // out_shape = [1, tokens, labels]; aggregate per-label across
            // tokens per the manifest (a routing-friendly presence signal).
            if (out_shape.size() < 3) {
                throw std::runtime_error("token-classification model must output [batch, tokens, labels]");
            }
            const size_t tokens = static_cast<size_t>(out_shape[out_shape.size() - 2]);
            const bool mean = manifest_.token_aggregation == "mean";
            std::vector<double> agg(num_labels, 0.0);
            size_t counted = 0;
            for (size_t t = 0; t < tokens; ++t) {
                // Skip [CLS]/[SEP]/… : the HuggingFace pipeline filters those
                // positions out of its output, so scoring them would invent
                // entities the reference never reports.
                if (t < input_ids.size() && special_ids_.count(input_ids[t])) continue;
                ++counted;
                auto p = normalize(logits + t * num_labels, num_labels, manifest_.score_normalization);
                for (size_t l = 0; l < num_labels; ++l) {
                    if (mean) agg[l] += p[l];
                    else agg[l] = std::max(agg[l], static_cast<double>(p[l]));
                }
            }
            if (counted == 0) {
                throw InvalidInput("input has no content tokens to classify "
                                   "(it tokenizes to special tokens only)");
            }
            for (size_t l = 0; l < num_labels; ++l) {
                scores[manifest_.id2label[l]] =
                    static_cast<float>(mean ? agg[l] / counted : agg[l]);
            }
        } else {
            // sequence-classification: normalize the label logits.
            if (out_shape.size() > 2) {
                throw std::runtime_error(
                    "text-classification model must output [batch, labels]; got a rank-" +
                    std::to_string(out_shape.size()) + " tensor (token-classification model?)");
            }
            auto p = normalize(logits, num_labels, manifest_.score_normalization);
            for (size_t l = 0; l < num_labels; ++l) scores[manifest_.id2label[l]] = p[l];
        }

        std::vector<std::pair<std::string, float>> ranked(scores.begin(), scores.end());
        std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if (top_k > 0 && static_cast<size_t>(top_k) < ranked.size()) ranked.resize(top_k);

        json labels = json::object();
        for (auto& [label, score] : ranked) labels[label] = score;
        return json{{"labels", labels}};
    }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions alloc_;
    TokenizerHandle tokenizer_ = nullptr;
    int64_t pad_id_ = -1;  // >= 0 when tokenizer.json enables padding
    std::set<int64_t> special_ids_;
    std::mutex tokenizer_mutex_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    Manifest manifest_;
};

// True when <dir>/manifest.json is a JSON object with "task":
// "image-classification". Anything else, including an unreadable manifest, is
// left to the text path, which reports its own errors.
bool is_image_model(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json", std::ios::binary);
    if (!f) return false;
    json j;
    try {
        f >> j;
    } catch (const std::exception&) {
        return false;
    }
    return j.is_object() && j.contains("task") && j["task"] == "image-classification";
}

// Error text can echo raw request bytes (nlohmann's parse errors quote the input), and the
// default strict dump() throws on invalid UTF-8 inside the handler, which turns a 400 into a 500.
std::string error_body(const std::string& message) {
    return json{{"error", message}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

void send_error(httplib::Response& res, int status, const std::string& message) {
    res.status = status;
    res.set_content(error_body(message), "application/json");
}

// top_k follows Lemonade's /v1/classify rule: an integer from 1 to 1,000,000.
int top_k_from_json(const json& v) {
    if (!v.is_number_integer() || v.get<long long>() < 1 || v.get<long long>() > kMaxTopK) {
        throw InvalidInput("top_k must be an integer from 1 to 1000000");
    }
    return static_cast<int>(v.get<long long>());
}

int top_k_from_field(const std::string& s) {
    if (s.empty() || s.size() > 7 || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        throw InvalidInput("top_k must be an integer from 1 to 1000000");
    }
    return top_k_from_json(json(std::stoll(s)));
}

uint64_t content_length(const httplib::Request& req) {
    return req.has_header("Content-Length") ? req.get_header_value_u64("Content-Length") : 0;
}

void drain(const httplib::Request& req, const httplib::ContentReader& reader) {
    if (req.is_multipart_form_data()) {
        reader([](const httplib::MultipartFormData&) { return true; }, [](const char*, size_t) { return true; });
    } else {
        reader([](const char*, size_t) { return true; });
    }
}

// Reads the body through httplib's ContentReader, so nothing is buffered
// before the handler has an admission slot. Receivers never abort: an
// oversize or unwanted part is discarded while the stream is still read to
// its end, which keeps the connection usable.
//
// Multipart: exactly one file part named "image" or "file", plus an optional
// "top_k" field. JSON: {"image": "<base64 or data:image/(jpeg|png);base64,...>", "top_k": k}.
void read_image_request(const httplib::Request& req, const httplib::Response& res,
                        const httplib::ContentReader& reader, uint64_t max_image_bytes, uint64_t max_body_bytes, std::string& bytes, int& top_k) {
    top_k = 0;
    bytes.clear();
    bool too_large = false;
    bool ok = false;
    if (req.is_multipart_form_data()) {
        enum class Part { Image, TopK, Other } cur = Part::Other;
        size_t image_parts = 0;
        std::string top_k_field;
        bool has_top_k = false;
        ok = reader(
            [&](const httplib::MultipartFormData& f) {
                if (f.name == "image" || f.name == "file") {
                    cur = ++image_parts == 1 && !too_large ? Part::Image : Part::Other;
                } else if (f.name == "top_k") {
                    cur = Part::TopK;
                    has_top_k = true;
                    top_k_field.clear();
                } else {
                    cur = Part::Other;
                }
                return true;
            },
            [&](const char* d, size_t n) {
                if (cur == Part::Image) {
                    if (bytes.size() + n > max_image_bytes) {
                        too_large = true;
                        cur = Part::Other;
                        std::string().swap(bytes);
                    } else {
                        bytes.append(d, n);
                    }
                } else if (cur == Part::TopK && top_k_field.size() < 16) {
                    top_k_field.append(d, std::min<size_t>(n, 16));
                }
                return true;
            });
        if (ok && too_large) {
            throw PayloadTooLarge("image is larger than the " + std::to_string(max_image_bytes) + " byte limit");
        }
        if (ok) {
            if (image_parts != 1) throw InvalidInput("exactly one image part ('image' or 'file') required");
            if (has_top_k) top_k = top_k_from_field(top_k_field);
            return;
        }
    } else {
        std::string body;
        const uint64_t declared = content_length(req);
        if (declared <= max_body_bytes) body.reserve(static_cast<size_t>(declared));
        ok = reader([&](const char* d, size_t n) {
            if (too_large) return true;
            if (body.size() + n > max_body_bytes) {
                too_large = true;
                std::string().swap(body);
            } else {
                body.append(d, n);
            }
            return true;
        });
        if (ok && too_large) {
            throw PayloadTooLarge("request body is over the " + std::to_string(max_body_bytes) + " byte limit");
        }
        if (ok) {
            json obj;
            std::string error;
            if (!parse_flat_json_object(body, obj, error)) {
                if (error == "request body is not valid JSON") error += " (send multipart/form-data or a JSON object)";
                throw InvalidInput(error);
            }
            std::string().swap(body);
            if (!obj.contains("image") || !obj["image"].is_string()) {
                throw InvalidInput("'image' (a base64 string) is required");
            }
            if (obj.contains("top_k")) top_k = top_k_from_json(obj["top_k"]);
            std::string image = std::move(obj["image"].get_ref<std::string&>());
            obj = json();
            std::string_view s = image;
            if (s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0) {
                throw InvalidInput("remote image URLs are not supported; send base64 or a data: URL");
            }
            if (s.rfind("data:", 0) == 0) {
                bool prefix_ok = false;
                for (std::string_view prefix : {"data:image/jpeg;base64,", "data:image/png;base64,"}) {
                    if (s.rfind(prefix, 0) == 0) {
                        s.remove_prefix(prefix.size());
                        prefix_ok = true;
                        break;
                    }
                }
                if (!prefix_ok) throw InvalidInput("data URLs must be data:image/jpeg;base64 or data:image/png;base64");
            }
            if (!imgproc::strict_base64_decode(s, bytes)) throw InvalidInput("'image' is not valid base64");
            return;
        }
    }
    std::string().swap(bytes);
    // A failed read has set res.status: 413 for a Content-Length over the
    // payload limit, 400 for a malformed or truncated body.
    if (res.status == 413) {
        throw PayloadTooLarge("request body is over the " + std::to_string(max_body_bytes) + " byte limit");
    }
    throw InvalidInput("cannot read the request body (malformed multipart or truncated)");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::unique_ptr<Model> text_model;
        std::unique_ptr<ImageModel> image;
        if (is_image_model(args.model_path)) {
            ImageServeOptions opts;
            opts.max_image_bytes = args.max_image_bytes;
            opts.limits.max_pixels = args.max_image_pixels;
            opts.limits.budget_factor = args.decode_budget_factor;
            opts.limits.max_budget = args.max_decode_bytes;
            opts.max_concurrent_decodes = args.max_concurrent_decodes;
            image = std::make_unique<ImageModel>(args.model_path, args.threads, opts, args.verbose);
        } else {
            text_model = std::make_unique<Model>(args.model_path, args.threads, args.verbose);
        }

        httplib::Server srv;
        // Per-recv idle limits, not a cap on a request's total time.
        srv.set_read_timeout(kSocketTimeout);
        srv.set_write_timeout(kSocketTimeout);
        // Base64 inflates by 4/3; 64 KiB covers multipart headers and the JSON wrapper.
        const uint64_t max_body_bytes = args.max_image_bytes * 4 / 3 + (64u << 10);
        // Image requests read their body only after taking one of these, so
        // http_threads bounds waiting connections, not buffered bodies. One
        // more than the decode slots lets the next body arrive during a decode.
        CountingSemaphore admission(args.max_concurrent_decodes + 1);
        if (image) {
            const size_t http_threads = static_cast<size_t>(args.http_threads);
            srv.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads); };
            srv.set_payload_max_length(static_cast<size_t>(max_body_bytes));
            if (args.decode_budget_factor * args.max_image_pixels + (4u << 20) > args.max_decode_bytes) {
                std::fprintf(stderr,
                             "ort-server: warning: --decode-budget-factor x --max-image-pixels exceeds "
                             "--max-decode-bytes; images near the pixel cap may fail the decode budget\n");
            }
        }

        srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            json health{{"status", "ok"}, {"onnxruntime", ORT_SERVER_ONNXRUNTIME_VERSION}};
            if (image) health["task"] = "image-classification";
            res.set_content(health.dump(), "application/json");
        });
        srv.Post("/classify", [&](const httplib::Request& req, httplib::Response& res) {
            if (!text_model) {
                send_error(res, 400, "this server hosts an image-classification model; POST /classify/image");
                return;
            }
            Model& model = *text_model;
            std::string text;
            int top_k = 0;
            try {
                json body = json::parse(req.body);
                text = body.contains("text") ? body.at("text").get<std::string>()
                                             : body.at("input").get<std::string>();
                top_k = body.value("top_k", 0);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(error_body(e.what()), "application/json");
                return;
            }
            try {
                res.set_content(model.classify(text, top_k).dump(), "application/json");
            } catch (const InvalidInput& e) {
                res.status = 400;  // the request is at fault, not the model
                res.set_content(error_body(e.what()), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(error_body(e.what()), "application/json");
            }
        });
        srv.Post("/classify/image", [&](const httplib::Request& req, httplib::Response& res,
                                        const httplib::ContentReader& reader) {
            if (!image) {
                drain(req, reader);
                send_error(res, 400, "this server hosts a text model; POST /classify");
                return;
            }
            if (content_length(req) > max_body_bytes) {
                drain(req, reader);
                send_error(res, 413, "request body is over the " + std::to_string(max_body_bytes) + " byte limit");
                return;
            }
            if (!admission.acquire_for(kAdmissionWait)) {
                drain(req, reader);
                send_error(res, 503, "busy: no request slot became free within 30 s");
                return;
            }
            SlotGuard slot(admission);
            try {
                std::string bytes;
                int top_k = 0;
                read_image_request(req, res, reader, args.max_image_bytes, max_body_bytes, bytes, top_k);
                res.set_content(image->classify(bytes, top_k).dump(), "application/json");
            } catch (const InvalidInput& e) {
                send_error(res, 400, e.what());
            } catch (const PayloadTooLarge& e) {
                send_error(res, 413, e.what());
            } catch (const Busy& e) {
                send_error(res, 503, e.what());
            } catch (const std::exception& e) {
                send_error(res, 500, e.what());
            }
        });

        if (!srv.listen("127.0.0.1", args.port)) {
            fprintf(stderr, "ort-server: failed to bind 127.0.0.1:%d\n", args.port);
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ort-server: %s\n", e.what());
        return 1;
    }
}
