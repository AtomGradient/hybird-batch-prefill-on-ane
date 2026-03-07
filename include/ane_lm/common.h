#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <mach/mach_time.h>

namespace ane_lm {

// Global verbose flag (default: off)
extern bool g_verbose;

// Verbose logging macro — prints only when g_verbose is true
#define LOG(...) do { if (ane_lm::g_verbose) fprintf(stderr, __VA_ARGS__); } while (0)

// BF16 <-> FP32 conversion
inline float bf16_to_f32(uint16_t bf) {
    uint32_t u = (uint32_t)bf << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

// IEEE 754 FP16 <-> FP32 conversion
inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {
            float f = (float)mant / 1024.0f;
            f *= (1.0f / 16384.0f);
            if (sign) f = -f;
            return f;
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000 | (mant << 13);
    } else {
        u = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint16_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

inline uint16_t bf16_to_f16(uint16_t bf) {
    return f32_to_f16(bf16_to_f32(bf));
}

inline void bf16_to_f32_vec(float *out, const uint16_t *in, int n) {
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(in[i]);
}

inline void bf16_to_f16_vec(uint16_t *out, const uint16_t *in, int n) {
    for (int i = 0; i < n; i++) out[i] = bf16_to_f16(in[i]);
}

// Timing utility
struct Timer {
    uint64_t start;
    static mach_timebase_info_data_t tb;
    static bool tb_init;

    Timer() {
        if (!tb_init) {
            mach_timebase_info(&tb);
            tb_init = true;
        }
        start = mach_absolute_time();
    }

    double elapsed_ms() const {
        uint64_t elapsed = mach_absolute_time() - start;
        return (double)elapsed * tb.numer / tb.denom / 1e6;
    }

    void reset() { start = mach_absolute_time(); }
};

} // namespace ane_lm
