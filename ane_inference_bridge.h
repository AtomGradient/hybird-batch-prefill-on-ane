#ifndef ANE_INFERENCE_BRIDGE_H
#define ANE_INFERENCE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to ANE-LM model
typedef void* ane_model_t;

/// Load model from directory. Returns NULL on failure.
/// Supports: qwen3_5, qwen3 (FP16/BF16, text-only or VLM)
ane_model_t ane_model_load(const char* model_dir);

/// Free model and all resources.
void ane_model_free(ane_model_t model);

/// Reset model state (clear KV caches and DeltaNet state).
void ane_model_reset(ane_model_t model);

/// Get vocabulary size.
int ane_model_vocab_size(ane_model_t model);

/// Get architecture string ("qwen3_5" or "qwen3").
const char* ane_model_architecture(ane_model_t model);

/// ANE batch prefill only (no decode).
/// Returns number of prompt tokens. tps_out receives prefill tok/s.
/// After this call, model state is populated — call ane_save_state() to export.
int ane_prefill_only(ane_model_t model, const char* prompt, double* tps_out);

/// Save prefill state to binary file for hybrid GPU decode.
/// Returns 1 on success, 0 on failure.
int ane_save_state(ane_model_t model, const char* path, int n_prompt_tokens);

/// Full generation (ANE prefill + ANE/CPU decode).
/// Returns generated text (caller must free with ane_free_string).
/// stats_out must point to 4 doubles: [prompt_tokens, prompt_tps, gen_tokens, gen_tps]
char* ane_generate(ane_model_t model, const char* prompt,
                   int max_tokens, float temperature,
                   double* stats_out);

/// Free string returned by ane_generate.
void ane_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // ANE_INFERENCE_BRIDGE_H
