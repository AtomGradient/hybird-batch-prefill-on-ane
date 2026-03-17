/**
 * Python bindings for ANE-LM via pybind11.
 *
 * Exposes:
 *   - ane_lm.Model: load, generate, generate_streaming, reset
 *   - ane_lm.Tokenizer: encode, decode, apply_chat_template
 *   - ane_lm.GenerationResult: text, prompt_tokens, prompt_tps, generation_tokens, generation_tps
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "../models/llm/qwen3_5.h"
#include "../models/llm/qwen3.h"
#include "../core/tokenizer.h"
#include "../core/sampling.h"
#include "../generate.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <memory>

namespace py = pybind11;

namespace {

/// Auto-detect model architecture from config.json
std::string detect_architecture(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) return "";

    nlohmann::json config;
    try {
        f >> config;
    } catch (...) {
        return "";
    }

    std::string model_type = config.value("model_type", "");

    // Check text_config for nested model_type (VLM models)
    if (config.contains("text_config")) {
        auto& tc = config["text_config"];
        if (tc.contains("model_type")) {
            std::string text_type = tc["model_type"].get<std::string>();
            if (text_type == "qwen3_5_text") return "qwen3_5";
        }
    }

    return model_type;
}

/// Complete generation result
struct GenerationResult {
    std::string text;
    int prompt_tokens = 0;
    double prompt_tps = 0.0;
    int generation_tokens = 0;
    double generation_tps = 0.0;
};

/// Python-facing Model class
class PyModel {
public:
    PyModel() = default;

    bool load(const std::string& model_dir) {
        model_dir_ = model_dir;
        std::string arch = detect_architecture(model_dir);

        if (arch == "qwen3_5" || arch == "qwen3_5_text") {
            model_ = std::make_unique<ane_lm::Qwen35Model>();
        } else if (arch == "qwen3") {
            model_ = std::make_unique<ane_lm::Qwen3Model>();
        } else {
            throw std::runtime_error(
                "Unsupported architecture: '" + arch + "'. "
                "ane-lm supports: qwen3_5, qwen3"
            );
        }

        if (!model_->load(model_dir)) {
            throw std::runtime_error("Failed to load model from: " + model_dir);
        }

        if (!tokenizer_.init(model_dir)) {
            throw std::runtime_error("Failed to load tokenizer from: " + model_dir);
        }

        loaded_ = true;
        architecture_ = arch;
        return true;
    }

    /// Full generation — returns complete result
    GenerationResult generate(
        const std::string& prompt,
        int max_tokens = 256,
        float temperature = 0.6f,
        float repetition_penalty = 1.2f,
        bool enable_thinking = false
    ) {
        check_loaded();

        ane_lm::SamplingParams params;
        params.temperature = temperature;
        params.repetition_penalty = repetition_penalty;

        GenerationResult result;
        std::string full_text;

        ane_lm::stream_generate(
            *model_, tokenizer_, prompt, max_tokens, enable_thinking, params,
            [&](const ane_lm::GenerationResponse& r) {
                if (r.token == -1) {
                    // Final stats
                    result.prompt_tokens = r.prompt_tokens;
                    result.prompt_tps = r.prompt_tps;
                    result.generation_tokens = r.generation_tokens;
                    result.generation_tps = r.generation_tps;
                } else {
                    full_text += r.text;
                }
            }
        );

        result.text = full_text;
        return result;
    }

    /// Streaming generation — calls Python callback for each token
    GenerationResult generate_streaming(
        const std::string& prompt,
        py::function callback,
        int max_tokens = 256,
        float temperature = 0.6f,
        float repetition_penalty = 1.2f,
        bool enable_thinking = false
    ) {
        check_loaded();

        ane_lm::SamplingParams params;
        params.temperature = temperature;
        params.repetition_penalty = repetition_penalty;

        GenerationResult result;
        std::string full_text;

        ane_lm::stream_generate(
            *model_, tokenizer_, prompt, max_tokens, enable_thinking, params,
            [&](const ane_lm::GenerationResponse& r) {
                if (r.token == -1) {
                    result.prompt_tokens = r.prompt_tokens;
                    result.prompt_tps = r.prompt_tps;
                    result.generation_tokens = r.generation_tokens;
                    result.generation_tps = r.generation_tps;
                } else {
                    full_text += r.text;
                    // Release GIL would be nice but ANE model is not thread-safe
                    // So we hold GIL during callback
                    try {
                        callback(r.text, r.generation_tokens, r.generation_tps);
                    } catch (py::error_already_set& e) {
                        // Python exception in callback — stop generation
                        throw;
                    }
                }
            }
        );

        result.text = full_text;
        return result;
    }

    /// Multi-turn chat generation
    GenerationResult chat(
        const std::vector<std::pair<std::string, std::string>>& messages,
        int max_tokens = 256,
        float temperature = 0.6f,
        float repetition_penalty = 1.2f,
        bool enable_thinking = false
    ) {
        check_loaded();

        ane_lm::SamplingParams params;
        params.temperature = temperature;
        params.repetition_penalty = repetition_penalty;

        GenerationResult result;
        std::string full_text;

        ane_lm::stream_generate(
            *model_, tokenizer_, messages, max_tokens, enable_thinking, params,
            [&](const ane_lm::GenerationResponse& r) {
                if (r.token == -1) {
                    result.prompt_tokens = r.prompt_tokens;
                    result.prompt_tps = r.prompt_tps;
                    result.generation_tokens = r.generation_tokens;
                    result.generation_tps = r.generation_tps;
                } else {
                    full_text += r.text;
                }
            }
        );

        result.text = full_text;
        return result;
    }

    /// Reset model state (clear caches)
    void reset() {
        if (model_) model_->reset();
    }

    /// Encode text to token ids
    std::vector<int> encode(const std::string& text) {
        check_loaded();
        return tokenizer_.encode(text);
    }

    /// Decode token ids to text
    std::string decode(const std::vector<int>& ids) {
        check_loaded();
        return tokenizer_.decode(ids);
    }

    bool is_loaded() const { return loaded_; }
    std::string architecture() const { return architecture_; }
    std::string model_path() const { return model_dir_; }
    int vocab_size() const { return model_ ? model_->vocab_size() : 0; }

private:
    void check_loaded() {
        if (!loaded_) {
            throw std::runtime_error("Model not loaded. Call load(model_dir) first.");
        }
    }

    std::unique_ptr<ane_lm::LLMModel> model_;
    ane_lm::Tokenizer tokenizer_;
    bool loaded_ = false;
    std::string model_dir_;
    std::string architecture_;
};

} // anonymous namespace


PYBIND11_MODULE(ane_lm, m) {
    m.doc() = "ANE-LM: Neural Engine batch prefill inference for Qwen3.5/Qwen3";
    m.attr("__version__") = "0.1.0";

    // GenerationResult
    py::class_<GenerationResult>(m, "GenerationResult")
        .def(py::init<>())
        .def_readwrite("text", &GenerationResult::text)
        .def_readwrite("prompt_tokens", &GenerationResult::prompt_tokens)
        .def_readwrite("prompt_tps", &GenerationResult::prompt_tps)
        .def_readwrite("generation_tokens", &GenerationResult::generation_tokens)
        .def_readwrite("generation_tps", &GenerationResult::generation_tps)
        .def("__repr__", [](const GenerationResult& r) {
            return "<GenerationResult text='" + r.text.substr(0, 50) + "...' "
                   "prompt=" + std::to_string(r.prompt_tokens) + "tok@" +
                   std::to_string((int)r.prompt_tps) + "tps "
                   "gen=" + std::to_string(r.generation_tokens) + "tok@" +
                   std::to_string((int)r.generation_tps) + "tps>";
        });

    // Model
    py::class_<PyModel>(m, "Model")
        .def(py::init<>())
        .def("load", &PyModel::load, py::arg("model_dir"),
             "Load model and tokenizer from directory")
        .def("generate", &PyModel::generate,
             py::arg("prompt"),
             py::arg("max_tokens") = 256,
             py::arg("temperature") = 0.6f,
             py::arg("repetition_penalty") = 1.2f,
             py::arg("enable_thinking") = false,
             "Generate text from prompt. Returns GenerationResult.")
        .def("generate_streaming", &PyModel::generate_streaming,
             py::arg("prompt"),
             py::arg("callback"),
             py::arg("max_tokens") = 256,
             py::arg("temperature") = 0.6f,
             py::arg("repetition_penalty") = 1.2f,
             py::arg("enable_thinking") = false,
             "Generate with per-token callback(text, n_tokens, tps). Returns GenerationResult.")
        .def("chat", &PyModel::chat,
             py::arg("messages"),
             py::arg("max_tokens") = 256,
             py::arg("temperature") = 0.6f,
             py::arg("repetition_penalty") = 1.2f,
             py::arg("enable_thinking") = false,
             "Multi-turn chat. messages = [(role, content), ...]. Returns GenerationResult.")
        .def("reset", &PyModel::reset, "Reset model state (clear KV caches)")
        .def("encode", &PyModel::encode, py::arg("text"), "Encode text to token ids")
        .def("decode", &PyModel::decode, py::arg("ids"), "Decode token ids to text")
        .def_property_readonly("is_loaded", &PyModel::is_loaded)
        .def_property_readonly("architecture", &PyModel::architecture)
        .def_property_readonly("model_path", &PyModel::model_path)
        .def_property_readonly("vocab_size", &PyModel::vocab_size)
        .def("__repr__", [](const PyModel& m) {
            if (!m.is_loaded()) return std::string("<ane_lm.Model (not loaded)>");
            return "<ane_lm.Model arch='" + m.architecture() + "' path='" + m.model_path() + "'>";
        });

    // Convenience function: load + generate in one call
    m.def("generate", [](const std::string& model_dir, const std::string& prompt,
                          int max_tokens, float temperature) {
        PyModel model;
        model.load(model_dir);
        return model.generate(prompt, max_tokens, temperature);
    },
    py::arg("model_dir"), py::arg("prompt"),
    py::arg("max_tokens") = 256, py::arg("temperature") = 0.6f,
    "Convenience: load model + generate in one call");
}
