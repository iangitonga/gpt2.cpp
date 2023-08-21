#pragma once

#include <cmath>
#include <cstdint>

namespace gten
{

// SCALAR DATA TYPES.
typedef int32_t Int32;
typedef uint16_t Float16;
typedef float Float32;

// Allows data type information to be stored and passed around as variables because we
// cannot do that with the types themselves.
enum class Dtype { Int32, Float16, Float32 };

// Convenient shorthands for the enum class above.
static Dtype kInt32 = Dtype::Int32;
static Dtype kFloat16 = Dtype::Float16;
static Dtype kFloat32 = Dtype::Float32;

// Convert Dtype to a str for error reporting.
static const char* dtype_str(Dtype dtype) {
    if (dtype == Dtype::Int32)
        return "Int32";
    else if (dtype == Dtype::Float16)
        return "Float16";
    else
        return "Float32";
}


// FP32 <-> FP16 Conversions.
inline Float32 fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        Float32 as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

inline uint32_t fp32_to_bits(Float32 f) {
    union {
        Float32 as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

inline Float32 fp16_to_fp32(Float16 h) noexcept
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

// MAX:  65504.0 (0)[11110]{1111111111}
// MIN: -65504.0 (1)[11110]{1111111111}

inline Float16 fp32_to_fp16(Float32 f) noexcept
{
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

} // namespace gten

// Assert that the given boolean is true. If false, print message and terminate program.
// TODO: Replace with C++ 20 __VA_OPT__, __VA_ARGS__ may not work on non-gcc compilers.
#define GTEN_ASSERT(boolean, message, ...)                                              \
    if (!(boolean)) {                                                                   \
        std::fprintf(stderr, "\x1B[1;31m");                                             \
        std::fprintf(stderr, "GTEN ERROR [File `%s` line %d]: ", __FILE__, __LINE__);   \
        std::fprintf(stderr, message, ##__VA_ARGS__);                                   \
        std::fprintf(stderr, "\n");                                                     \
        std::exit(EXIT_FAILURE);                                                        \
    } 
