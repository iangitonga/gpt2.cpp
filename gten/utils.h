#pragma once

#include <chrono>
#include <cstdint>
#include <cmath>
#include <iostream>

#include "gten_types.h"

// TODO: Allow AVX without F16C for FP32 mode.
#if defined(__AVX__) && defined(__F16C__)
#define GTEN_SIMD 1
#define GTEN_AVX 1

#include <immintrin.h>

#else
#define GTEN_SIMD 0
#define GTEN_AVX 0
#endif

namespace gten
{

#if GTEN_SIMD

#if GTEN_AVX

// FUNDAMENTAL VECTOR DATA TYPES.
typedef __m256 Vec_f32x8;

// FLOATING POINT VECTOR OPERATIONS

inline Vec_f32x8 vec_f32x8_load(const Float16* src_ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128((__m128i_u *)(const_cast<Float16*>(src_ptr))));
}

inline Vec_f32x8 vec_f32x8_load(const Float32* src_ptr) {
    return _mm256_loadu_ps(const_cast<Float32*>(src_ptr));
}

inline void vec_f32x8_store(Vec_f32x8 vec, Float32* dest_ptr) {
    _mm256_storeu_ps(dest_ptr, vec);
}

inline void vec_f32x8_store(Vec_f32x8 vec, Float16* dest_ptr) {
    return _mm_storeu_si128((__m128i_u *)dest_ptr, _mm256_cvtps_ph(vec, 0));
}

inline Vec_f32x8 vec_f32x8_add(Vec_f32x8 a, Vec_f32x8 b) {
    return _mm256_add_ps(a, b);
}

inline Vec_f32x8 vec_f32x8_mul(Vec_f32x8 a, Vec_f32x8 b) {
    return _mm256_mul_ps(a, b);
}

// Return A * B + C
inline Vec_f32x8 vec_f32x8_fma(Vec_f32x8 a, Vec_f32x8 b, Vec_f32x8 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

inline Float32 vec_f32x8_sum(Vec_f32x8 vec) {
    Float32* f = (Float32 *)(&vec);
    return f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
}

inline Vec_f32x8 vec_f32x8_setzero() {
    return _mm256_setzero_ps();
}
#endif

#else // NO SIMD AVAILABLE.

// FLOATING PONT SCALAR OPERATIONS

struct Vec_f32x8 {
    Float32 a[8];
};

static Float32* init_cache() {
    // This is a memory leak because we never delete the cache but its not a problem
    // because the cache lasts the entire lifetime of the program.
    Float32* cache = new Float32[65536];
    Float16 idx = 0;
    for (int i = 0; i < 65536; i++) {
        cache[i] = fp16_to_fp32(idx);
        idx += 1;
    }
    return cache;
}

// Global lookup table to avoid recomputations.
static const Float32* G_fp16_to_fp32_cache = init_cache();

inline Vec_f32x8 vec_f32x8_load(const Float16* src_ptr) {
    Vec_f32x8 v;
    v.a[0] = G_fp16_to_fp32_cache[src_ptr[0]];
    v.a[1] = G_fp16_to_fp32_cache[src_ptr[1]];
    v.a[2] = G_fp16_to_fp32_cache[src_ptr[2]];
    v.a[3] = G_fp16_to_fp32_cache[src_ptr[3]];
    v.a[4] = G_fp16_to_fp32_cache[src_ptr[4]];
    v.a[5] = G_fp16_to_fp32_cache[src_ptr[5]];
    v.a[6] = G_fp16_to_fp32_cache[src_ptr[6]];
    v.a[7] = G_fp16_to_fp32_cache[src_ptr[7]];
    return v;
}

inline Vec_f32x8 vec_f32x8_load(const Float32* src_ptr) {
    Vec_f32x8 v;
    v.a[0] = src_ptr[0];
    v.a[1] = src_ptr[1];
    v.a[2] = src_ptr[2];
    v.a[3] = src_ptr[3];
    v.a[4] = src_ptr[4];
    v.a[5] = src_ptr[5];
    v.a[6] = src_ptr[6];
    v.a[7] = src_ptr[7];
    return v;
}

inline void vec_f32x8_store(Vec_f32x8 v, Float16* dest_ptr) {
    for (int i = 0; i < 8; i++) {
        dest_ptr[i] = fp32_to_fp16(v.a[i]);
    }
}

inline void vec_f32x8_store(Vec_f32x8 v, Float32* dest_ptr) {
    dest_ptr[0] = v.a[0];
    dest_ptr[1] = v.a[1];
    dest_ptr[2] = v.a[2];
    dest_ptr[3] = v.a[3];
    dest_ptr[4] = v.a[4];
    dest_ptr[5] = v.a[5];
    dest_ptr[6] = v.a[6];
    dest_ptr[7] = v.a[7];
}

inline Vec_f32x8 vec_f32x8_add(Vec_f32x8 v0, Vec_f32x8 v1) {
    Vec_f32x8 sum;
    sum.a[0] = v0.a[0] + v1.a[0];
    sum.a[1] = v0.a[1] + v1.a[1];
    sum.a[2] = v0.a[2] + v1.a[2];
    sum.a[3] = v0.a[3] + v1.a[3];
    sum.a[4] = v0.a[4] + v1.a[4];
    sum.a[5] = v0.a[5] + v1.a[5];
    sum.a[6] = v0.a[6] + v1.a[6];
    sum.a[7] = v0.a[7] + v1.a[7];
    return sum;
}

inline Vec_f32x8 vec_f32x8_mul(Vec_f32x8 v0, Vec_f32x8 v1) {
    Vec_f32x8 fma;
    fma.a[0] = v0.a[0] * v1.a[0];
    fma.a[1] = v0.a[1] * v1.a[1];
    fma.a[2] = v0.a[2] * v1.a[2];
    fma.a[3] = v0.a[3] * v1.a[3];
    fma.a[4] = v0.a[4] * v1.a[4];
    fma.a[5] = v0.a[5] * v1.a[5];
    fma.a[6] = v0.a[6] * v1.a[6];
    fma.a[7] = v0.a[7] * v1.a[7];
    return fma;
}

inline Vec_f32x8 vec_f32x8_fma(Vec_f32x8 v0, Vec_f32x8 v1, Vec_f32x8 v2) {
    Vec_f32x8 fma;
    fma.a[0] = v0.a[0] * v1.a[0] + v2.a[0];
    fma.a[1] = v0.a[1] * v1.a[1] + v2.a[1];
    fma.a[2] = v0.a[2] * v1.a[2] + v2.a[2];
    fma.a[3] = v0.a[3] * v1.a[3] + v2.a[3];
    fma.a[4] = v0.a[4] * v1.a[4] + v2.a[4];
    fma.a[5] = v0.a[5] * v1.a[5] + v2.a[5];
    fma.a[6] = v0.a[6] * v1.a[6] + v2.a[6];
    fma.a[7] = v0.a[7] * v1.a[7] + v2.a[7];
    return fma;
}

inline Float32 vec_f32x8_sum(Vec_f32x8 v) {
    Float32 sum = v.a[0] + v.a[1] + v.a[2] + v.a[3] + v.a[4] + v.a[5] + v.a[6] + v.a[7];
    return sum;
}

inline Vec_f32x8 vec_f32x8_setzero() {
    Vec_f32x8 v;
    v.a[0] = 0.0f;
    v.a[1] = 0.0f;
    v.a[2] = 0.0f;
    v.a[3] = 0.0f;
    v.a[4] = 0.0f;
    v.a[5] = 0.0f;
    v.a[6] = 0.0f;
    v.a[7] = 0.0f;
    return v;
}

#endif


// Floating point conversion functions below are implemented using templates to ensure
// type-safety because C++ performs implicit conversions between integal types silently
// which can lead to subtle bugs if the conversion below functions are invoked with
// inputs of the unintended type. With the templates implementation below, because we
// explicitly provide and enforce input types, we cannot have such bugs.

// Convert floating point value to Float32. Conversion to Float32 is only allowed from
// Float32 or Float16. Attempt to convert from any other type will cause a runtime
// error. I tried "static_assert" to throw compile-time error but it didn't work.
template<typename input_t>
inline Float32 fpcvt_to_fp32(input_t value) noexcept {
    if constexpr(std::is_same<input_t, Float32>::value) {
        return value;
    }
    else if constexpr(std::is_same<input_t, Float16>::value) {
        return fp16_to_fp32(value);
    }
    else {
        GTEN_ASSERT(false, "Conversion to FP32 is only allowed for FP32 and FP16 types.");
        // Just to avoid "no return statement in function returning non-void" error in
        // case we instantiate using a disallowed type.
        return 0;
    }
}

// Convert Float32 value to a given type. The allowed types are Float32 and Float16.
// Attemt to convert to any other type will cause a runtime error.
// I tried "static_assert" to throw compile-time error but it didn't work.
template<typename output_t>
inline output_t fpcvt_from_fp32(Float32 value) noexcept {
    if constexpr(std::is_same<output_t, Float32>::value) {
        return value;
    }
    else if constexpr(std::is_same<output_t, Float16>::value) {
        return fp32_to_fp16(value);
    }
    else {
        GTEN_ASSERT(false, "Conversion from FP32 is only allowed for FP32 and FP16 types.");
        return 0;
    }
}

} // namespace gten

