#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace ane_lm {

// ANE minimum spatial dimension required by hardware
constexpr int ANE_SPATIAL = 32;

// Opaque kernel handle
struct ANEKernel;

// Per-layer ANE kernels
struct LayerANEKernels {
    ANEKernel* first_proj = nullptr;
    ANEKernel* o_proj = nullptr;
    ANEKernel* fused_ffn = nullptr;
};

// Global state
void ane_init();
bool ane_available();
void ane_set_persist_cache(bool enabled);
int ane_compile_count();
int ane_cache_loads();

// Kernel compilation (from BF16 weights)
ANEKernel* ane_compile_matmul(const uint16_t* bf16_weights, int out_dim, int in_dim);
ANEKernel* ane_compile_fused_2(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                int in_dim);
ANEKernel* ane_compile_fused_3(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                const uint16_t* bf16_c, int c_out,
                                int in_dim);
ANEKernel* ane_compile_fused_ffn(const uint16_t* gate_bf16, const uint16_t* up_bf16,
                                  const uint16_t* down_bf16, int dim, int inter_ch);

// Kernel compilation (from pre-converted ANE blob files)
ANEKernel* ane_compile_matmul_blob(const std::string& blob_path, int out_dim, int in_dim);
ANEKernel* ane_compile_fused_2_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     int in_dim);
ANEKernel* ane_compile_fused_3_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     const std::string& c_path, int c_out,
                                     int in_dim);
ANEKernel* ane_compile_fused_ffn_blob(const std::string& gate_path, const std::string& up_path,
                                       const std::string& down_path, int dim, int inter_ch);

// Kernel execution
bool ane_matvec(ANEKernel* k, float* output, const float* input, int in_dim, int out_dim);

// Batch kernel execution: process up to ANE_SPATIAL (32) vectors in one ANE call.
// input:  batch_size contiguous vectors, input[b * in_dim + c]
// output: batch_size contiguous vectors, output[b * out_dim + c]
// batch_size must be <= ANE_SPATIAL (32).
bool ane_matvec_batch(ANEKernel* k, float* output, const float* input,
                      int in_dim, int out_dim, int batch_size);

// Kernel cleanup
void ane_free(ANEKernel* k);
void ane_free_layer(LayerANEKernels* lk);

} // namespace ane_lm
