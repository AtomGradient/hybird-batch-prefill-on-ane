/**
 * C bridge for ANE-LM — exposes C API for Swift/xcframework integration.
 *
 * This wraps the C++ ANE-LM model classes behind an opaque C handle,
 * allowing Swift to call via the module map without C++ interop.
 */

#include "ane_inference_bridge.h"
#include "models/llm/qwen3_5.h"
#include "models/llm/qwen3.h"
#include "core/tokenizer.h"
#include "core/sampling.h"
#include "generate.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <memory>
#include <cstring>
#include <cstdlib>

namespace {

struct AneModelHandle {
    std::unique_ptr<ane_lm::LLMModel> model;
    ane_lm::Tokenizer tokenizer;
    std::string architecture;
    int last_prefill_n = 0;
};

std::string detect_arch(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) return "";
    nlohmann::json config;
    try { f >> config; } catch (...) { return ""; }

    std::string model_type = config.value("model_type", "");
    if (config.contains("text_config")) {
        auto& tc = config["text_config"];
        if (tc.contains("model_type")) {
            std::string tt = tc["model_type"].get<std::string>();
            if (tt == "qwen3_5_text") return "qwen3_5";
        }
    }
    return model_type;
}

// Static buffer for architecture string
static thread_local char arch_buf[64];

} // anonymous namespace


extern "C" {

ane_model_t ane_model_load(const char* model_dir) {
    if (!model_dir) return nullptr;

    auto* handle = new (std::nothrow) AneModelHandle();
    if (!handle) return nullptr;

    std::string dir(model_dir);
    handle->architecture = detect_arch(dir);

    if (handle->architecture == "qwen3_5" || handle->architecture == "qwen3_5_text") {
        handle->model = std::make_unique<ane_lm::Qwen35Model>();
    } else if (handle->architecture == "qwen3") {
        handle->model = std::make_unique<ane_lm::Qwen3Model>();
    } else {
        delete handle;
        return nullptr;
    }

    if (!handle->model->load(dir)) {
        delete handle;
        return nullptr;
    }

    if (!handle->tokenizer.init(dir)) {
        delete handle;
        return nullptr;
    }

    return static_cast<ane_model_t>(handle);
}

void ane_model_free(ane_model_t model) {
    delete static_cast<AneModelHandle*>(model);
}

void ane_model_reset(ane_model_t model) {
    if (!model) return;
    auto* h = static_cast<AneModelHandle*>(model);
    h->model->reset();
    h->last_prefill_n = 0;
}

int ane_model_vocab_size(ane_model_t model) {
    if (!model) return 0;
    return static_cast<AneModelHandle*>(model)->model->vocab_size();
}

const char* ane_model_architecture(ane_model_t model) {
    if (!model) return "";
    auto* h = static_cast<AneModelHandle*>(model);
    strncpy(arch_buf, h->architecture.c_str(), sizeof(arch_buf) - 1);
    arch_buf[sizeof(arch_buf) - 1] = '\0';
    return arch_buf;
}

int ane_prefill_only(ane_model_t model, const char* prompt, double* tps_out) {
    if (!model || !prompt) return 0;
    auto* h = static_cast<AneModelHandle*>(model);

    h->model->reset();
    auto result = ane_lm::prefill_only(*h->model, h->tokenizer, prompt);
    h->last_prefill_n = result.n_tokens;

    if (tps_out) *tps_out = result.tps;
    return result.n_tokens;
}

int ane_save_state(ane_model_t model, const char* path, int n_prompt_tokens) {
    if (!model || !path) return 0;
    auto* h = static_cast<AneModelHandle*>(model);
    return h->model->save_state(path, n_prompt_tokens) ? 1 : 0;
}

char* ane_generate(ane_model_t model, const char* prompt,
                   int max_tokens, float temperature,
                   double* stats_out) {
    if (!model || !prompt) return nullptr;
    auto* h = static_cast<AneModelHandle*>(model);

    h->model->reset();

    ane_lm::SamplingParams params;
    params.temperature = temperature;

    std::string full_text;
    double prompt_tps = 0, gen_tps = 0;
    int prompt_tokens = 0, gen_tokens = 0;

    ane_lm::stream_generate(
        *h->model, h->tokenizer, std::string(prompt),
        max_tokens, false, params,
        [&](const ane_lm::GenerationResponse& r) {
            if (r.token == -1) {
                prompt_tokens = r.prompt_tokens;
                prompt_tps = r.prompt_tps;
                gen_tokens = r.generation_tokens;
                gen_tps = r.generation_tps;
            } else {
                full_text += r.text;
            }
        }
    );

    if (stats_out) {
        stats_out[0] = (double)prompt_tokens;
        stats_out[1] = prompt_tps;
        stats_out[2] = (double)gen_tokens;
        stats_out[3] = gen_tps;
    }

    char* result = (char*)malloc(full_text.size() + 1);
    if (result) {
        memcpy(result, full_text.c_str(), full_text.size() + 1);
    }
    return result;
}

void ane_free_string(char* str) {
    free(str);
}

} // extern "C"
