
#include "log.h"
#include "quants.h"
#include "tensor.h"
#include "simd_ops.h"

#ifdef _OPENMP
#define GTEN_OMP 1
#include <omp.h>
#endif

namespace gten {
namespace ops {


// Stores buffers required by ops.
class OpsState {
public:
    OpsState() {
        // max bufsize op: gelu 2 x n_embd * 4 == 2 * n_mlp
        const int max_bufsize = 2 * 4 * 1600;
        buf_ = new float[max_bufsize];
        buf_numel_ = max_bufsize;
    }
    ~OpsState() { delete[] buf_; }

    // Obtain a ptr to a buffer of size `numel` floats.
    float* buf(int numel) const {
        GTEN_ASSERT(numel <= buf_numel_);
        return buf_;
    }

private:
    float* buf_ = nullptr;
    int buf_numel_ = 0;
};

static const OpsState g_ops_state = OpsState();


static void vec_add_f16(const Float16* a, const Float16* b, Float16* out, int vec_size)
{
#ifdef GTEN_SIMD_AVX
    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;

    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(a + i);
        Vec_f32x8 x1 = vec_f32x8_load(b + i);
        Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
        vec_f32x8_store(x_sum, out + i);
    }

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(a[i]);
        const float x1 = fp16_to_fp32(b[i]);
        out[i] = fp32_to_fp16(x0 + x1);
    }
#else

    for (int i = 0; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(a[i]);
        const float x1 = fp16_to_fp32(b[i]);
        out[i] = fp32_to_fp16(x0 + x1);
    }

#endif
}


static void vec_add_f32(const float* a, const float* b, float* out, int vec_size)
{
#ifdef GTEN_SIMD_AVX

    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;

    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(a + i);
        Vec_f32x8 x1 = vec_f32x8_load(b + i);
        Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
        vec_f32x8_store(x_sum, out + i);
    }

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = a[i];
        const float x1 = b[i];
        out[i] = x0 + x1;
    }

#else
    const int unrolled_vec_size = (vec_size / 8) * 8;

    for (int i = 0; i < unrolled_vec_size; i += 8) {
        out[i] = a[i] + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
        out[i + 4] = a[i + 4] + b[i + 4];
        out[i + 5] = a[i + 5] + b[i + 5];
        out[i + 6] = a[i + 6] + b[i + 6];
        out[i + 7] = a[i + 7] + b[i + 7];
    } 

    // leftovers
    for (int i = unrolled_vec_size; i < vec_size; i++) {
        out[i] = a[i] + b[i];
    }

#endif
}

static float vec_dot_product_f16(const Float16* vec_a, const Float16* vec_b, int vec_size)
{
#ifdef GTEN_SIMD_AVX

    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;
    
    Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();
    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(vec_a + i);
        Vec_f32x8 x1 = vec_f32x8_load(vec_b + i);
        // dot_prod += vec_f32x8_sum(vec_f32x8_mul(x0, x1));
        dot_prod_accum = vec_f32x8_fma(x0, x1, dot_prod_accum);
    }
    
    float dot_prod = vec_f32x8_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fp16_to_fp32(vec_a[i]);
        const float x1 = fp16_to_fp32(vec_b[i]);
        dot_prod += x0 * x1;
    }

#else

    float dot_prod = 0.0f;

    for (int i = 0; i < vec_size; i += 1)
    {
        dot_prod += fp16_to_fp32(vec_a[i]) * fp16_to_fp32(vec_b[i]);
    }

#endif

    return dot_prod;
}


static float vec_dot_product_f32(const float* vec_a, const float* vec_b, int vec_size)
{
#ifdef GTEN_SIMD_AVX

    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;
    
    Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();
    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(vec_a + i);
        Vec_f32x8 x1 = vec_f32x8_load(vec_b + i);
        // dot_prod += vec_f32x8_sum(vec_f32x8_mul(x0, x1));
        dot_prod_accum = vec_f32x8_fma(x0, x1, dot_prod_accum);
    }
    
    float dot_prod = vec_f32x8_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = vec_a[i];
        const float x1 = vec_b[i];
        dot_prod += x0 * x1;
    }

# else
    const int unrolled_vec_size = (vec_size / 8) * 8;

    float dot_prod = 0.0f;
    for (int i = 0; i < unrolled_vec_size; i += 8) {
        dot_prod += vec_a[i] * vec_b[i];
        dot_prod += vec_a[i + 1] * vec_b[i + 1];
        dot_prod += vec_a[i + 2] * vec_b[i + 2];
        dot_prod += vec_a[i + 3] * vec_b[i + 3];
        dot_prod += vec_a[i + 4] * vec_b[i + 4];
        dot_prod += vec_a[i + 5] * vec_b[i + 5];
        dot_prod += vec_a[i + 6] * vec_b[i + 6];
        dot_prod += vec_a[i + 7] * vec_b[i + 7];
    }

    // leftovers
    for (int i = unrolled_vec_size; i < vec_size; i++) {
        dot_prod += vec_a[i] * vec_b[i];
    }

#endif
    return dot_prod;
}


static float vec_dot_product_q8(const Qint8* a, const Float16* a_ds, const Qint8* b, const Float16* b_ds, int blk_size, int vec_size)
{
    // GTEN_ASSERTM(vec_size % blk_size == 0, "row size: %d is incompatible with block size: %d", vec_size, blk_size);

    // Ensure that when the input vector is inside a block rather than the
    // vector having multiple blocks inside it, we still make the right computations.
    const int nblocks = blk_size < vec_size ? vec_size / blk_size : 1;
    blk_size = blk_size < vec_size ? blk_size : vec_size;

    // blk_size % 8 == 0 || blk_sizw % 16 == 0

#ifdef GTEN_SIMD_AVX
    // Dot product accumulator with 4 slots. The sum of the four accumulators gives the
    // total dot product.
    __m128 dot_accum = _mm_set1_ps(0.0f);

    for (int i = 0; i < nblocks; i++)
    {
        // dotprod = aq0*ad * bq0*bd + aq1*ad * bq1*bd + ... + aqN*ad + bqN*bd
        //         = adbd(aq0 * bq0) + adbd(aq1 * bq1) + ... + adbd(aqN * bqN)
        //         = adbd(aq0 * bq0 + aq1 * bq1 + ... + aqN * bqN)
        // We compute integer arithmetic inside the brackets and scaled by the block
        // quantisation deltas.

        // Integer dot product accumulator for current block.
        __m128i blk_dot_accum = _mm_set1_epi32(0);

        for (int j = 0; j < blk_size; j += 16)
        {
            const int idx_offs = i * blk_size + j;

            // Load 64-bit(8 1-byte quants) in the lower half. [8-quants, -------].
            const __m128i a00 = _mm_loadu_si64(a + idx_offs);
            const __m128i a01 = _mm_loadu_si64(a + idx_offs + 8);

            const __m128i b00 = _mm_loadu_si64(b + idx_offs);
            const __m128i b01 = _mm_loadu_si64(b + idx_offs + 8);

            // Convert 8 quants in the lower half to 16-bit ints.
            const __m128i a02 = _mm_cvtepi8_epi16(a00);
            const __m128i a03 = _mm_cvtepi8_epi16(a01);

            const __m128i b02 = _mm_cvtepi8_epi16(b00);
            const __m128i b03 = _mm_cvtepi8_epi16(b01);

            // Multiply the 8 16-bit ints to obtain 8 32-bit ints and add adjacent
            // values to obtain 4 32-bit ints.
            // TODO: Can we instead do 16-bit to 16-bit e.g _mullo_epi16
            const __m128i c00 = _mm_madd_epi16(a02, b02);
            const __m128i c01 = _mm_madd_epi16(a03, b03);

            // Add the results and add the output to the accumulator.
            const __m128i c02 = _mm_add_epi32(c00, c01);
            blk_dot_accum = _mm_add_epi32(blk_dot_accum, c02);
        }

        // const __m128 a_blk_delta = _mm_broadcast_ss(a_ds + i);
        // const __m128 b_blk_delta = _mm_broadcast_ss(b_ds + i); 
        // const __m128 ab_blk_delta = _mm_mul_ps(a_blk_delta, b_blk_delta);
        const __m128 ab_blk_delta = _mm_set1_ps(fp16_to_fp32(a_ds[i]) * fp16_to_fp32(b_ds[i]));

        const __m128 blk_dot_accum_f = _mm_cvtepi32_ps(blk_dot_accum);
        dot_accum = _mm_add_ps(dot_accum, _mm_mul_ps(blk_dot_accum_f, ab_blk_delta));
    }

    const __m128 dotsum0 = _mm_hadd_ps(dot_accum, dot_accum);
    const __m128 dotsum1 = _mm_hadd_ps(dotsum0, dotsum0);
    const float dot_prod = _mm_cvtss_f32(dotsum1);

#else

    float dot_prod = 0.0f;

    for (int i = 0; i < nblocks; i++)
    {
        // accumulator for integer block-dot products.
        int blk_dot_prod_i[2] = {0, 0};

        for (int j = 0; j < blk_size; j += 8) {
            const int idx = i * blk_size + j;
            blk_dot_prod_i[0] += a[idx] * b[idx];
            blk_dot_prod_i[1] += a[idx + 1] * b[idx + 1];
            blk_dot_prod_i[0] += a[idx + 2] * b[idx + 2];
            blk_dot_prod_i[1] += a[idx + 3] * b[idx + 3];
            blk_dot_prod_i[0] += a[idx + 4] * b[idx + 4];
            blk_dot_prod_i[1] += a[idx + 5] * b[idx + 5];
            blk_dot_prod_i[0] += a[idx + 6] * b[idx + 6];
            blk_dot_prod_i[1] += a[idx + 7] * b[idx + 7];
        }

        const float blk_dot_prod = float(blk_dot_prod_i[0] + blk_dot_prod_i[1]);
        const float a_delta = fp16_to_fp32(a_ds[i]);
        const float b_delta = fp16_to_fp32(b_ds[i]);
        dot_prod += blk_dot_prod * a_delta * b_delta;
    }
#endif

    return dot_prod;
}

static void vec_layer_norm_f16(const Float16* vec, int vec_size, const Float16* weight, const Float16* bias, Float16* out)
{
    // Calculate the mean.
    float sum = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        sum += fp16_to_fp32(vec[i]);
    }
    const float mean = sum / vec_size;

    // Calculate the std-dev.
    float sum_squares = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        float x = fp16_to_fp32(vec[i]);
        sum_squares += (x - mean) * (x - mean);
    }
    const float variance = sum_squares / (float)vec_size;
    const float stddev = std::sqrt(variance);

    // Normalize.
    float eps = 1e-05f;
    for (int i = 0; i < vec_size; i++) {
        float x = fp16_to_fp32(vec[i]);
        float w = fp16_to_fp32(weight[i]);
        float b = fp16_to_fp32(bias[i]);

        // Epsilon added to standard deviation prevents div by zero.
        float normalized = ((x - mean) / (stddev + eps)) * w + b;
        out[i] = fp32_to_fp16(normalized);
    }
}

static void vec_layer_norm_f32(const float* vec, int vec_size, const Float16* weight, const Float16* bias, float* out)
{
    // Calculate the mean.
    float sum = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        sum += vec[i];
    }
    const float mean = sum / vec_size;

    // Calculate the std-dev.
    float sum_squares = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        const float x = vec[i];
        sum_squares += (x - mean) * (x - mean);
    }
    const float variance = sum_squares / (float)vec_size;
    const float stddev = std::sqrt(variance);

    // Normalize.
    const float eps = 1e-05f;
    for (int i = 0; i < vec_size; i++) {
        const float x = vec[i];
        const float w = fp16_to_fp32(weight[i]);
        const float b = fp16_to_fp32(bias[i]);

        // Epsilon added to standard deviation prevents div by zero.
        const float normalized = ((x - mean) / (stddev + eps)) * w + b;
        out[i] = normalized;
    }
}


static void tensor_row_index_impl_f16(const Tensor& src, const Tensor& indices, Tensor& out, int last_token_only)
{
    const Float16* w_data = src.data_ptr<Float16>();
    const int* indices_data = indices.data_ptr<int>();
    Float16* out_data = out.data_ptr<Float16>();
    const int rowsize = src.size(1);
    const size_t rowsizebytes = rowsize * src.itemsize();

    const int n_ctx = indices.size(0);
    const int ctx_start = last_token_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < indices.numel(); i++) {
        const void* src_row_data = w_data + indices_data[i] * rowsize;
        void* out_row_data = out_data + i * rowsize;
        std::memcpy(out_row_data, src_row_data, rowsizebytes);
    }
}

static void tensor_row_index_impl_q8(const Tensor& src, const Tensor& indices, Tensor& out, bool last_token_only)
{
    const Qint8* src_data = src.data_ptr<Qint8>();
    const int* indices_data = indices.data_ptr<int>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int rowsize = src.size(1);
    const size_t rowsizebytes = rowsize * src.itemsize();

    const Qparams& w_qparams = src.qparams();
    Qparams& out_qparams = out.qparams();

    const int block_size = w_qparams.block_size();
    const int n_blocks = rowsize / block_size;

    const int n_ctx = indices.size(0);
    const int ctx_start = last_token_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < indices.numel(); i++) {
        const Qint8* src_row_data = src_data + indices_data[i] * rowsize;
        Qint8* dest_row_data = out_data + i * rowsize;

        std::memcpy(dest_row_data, src_row_data, rowsizebytes);

        // Copy row quantization deltas.
        const Float16* w_row_deltas = w_qparams.row_deltas(indices_data[i]);
        Float16* out_row_deltas = out_qparams.row_deltas(i);
        std::memcpy(out_row_deltas, w_row_deltas, n_blocks * sizeof(Float16));
    }
}

/// @brief Copies the indexed rows of the source tensor to output tensor.
/// @param src A 2-d tensor to be indexed.
/// @param indices A 1-d tensor of indices with dtype = int.
/// @param out A 2d tensor with enough capacity to fit the indexed rows. Its dtype
///  must be the same as source tensor.
/// @param last_token_only Whether to index the last token only, if others are cached.
void token_embed(const Tensor& weight, const Tensor& tokens, Tensor& out, bool last_token_only = false)
{
    GTEN_ASSERT(weight.is_2d());
    GTEN_ASSERT(tokens.is_1d() && tokens.dtype() == kInt32);
    const int n_ctx = tokens.size(0);
    const int n_embd = weight.size(1);
    GTEN_ASSERT(out.shape_eq({n_ctx, n_embd}));
    GTEN_ASSERT(weight.dtype() == out.dtype());

    if (weight.is_quantized()) {
        tensor_row_index_impl_q8(weight, tokens, out, last_token_only);
    } else {
        tensor_row_index_impl_f16(weight, tokens, out, last_token_only);
    }
}

static void emb_matmul_impl_q8(const Tensor& x, const Tensor& w, Tensor& out)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);

    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();

    const int block_size = x_qparams.block_size();

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < n_vocab; wrow++) {
            const Qint8* x_row_data = x_data + xrow * n_embd;
            const Float16* x_row_deltas = x_qparams.row_deltas(xrow);
            const Qint8* w_row_data = w_data + wrow * n_embd;
            const Float16* w_row_deltas = w_qparams.row_deltas(wrow);

            const float dot_prod = vec_dot_product_q8(x_row_data, x_row_deltas, w_row_data, w_row_deltas, block_size, n_embd);
            out_data[wrow] = dot_prod;
        }
    }
}

static void emb_matmul_impl_f16(const Tensor& x, const Tensor& w, Tensor& out)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < n_vocab; wrow++) {
            const Float16* x_row_data = x_data + xrow * n_embd;
            const Float16* w_row_data = w_data + wrow * n_embd;
            float dot_prod = vec_dot_product_f16(x_row_data, w_row_data, n_embd);
            out_data[wrow] = dot_prod;
        }
    }
}

/// @brief Computes a matmul between input's last ctx vector and emb table to produce logits
///   for the next token.
/// @param x Input tensor of shape (n_ctx, n_embd).
/// @param w Embedding table tensor of shape (n_vocab, n_embd).
/// @param out Output tensor of shape (n_vocab).
static void emb_matmul(const Tensor& x, const Tensor& weight, Tensor& out) {
    GTEN_ASSERT(x.is_2d());
    const int n_embd = x.size(1);
    GTEN_ASSERT(weight.is_2d() && weight.size(1) == n_embd);
    const int n_vocab = weight.size(0);
    GTEN_ASSERT(out.is_1d() && out.dtype() == kFloat32 && out.size(0) == n_vocab);

    if (weight.is_quantized()) {
        emb_matmul_impl_q8(x, weight, out);
    } else {
        emb_matmul_impl_f16(x, weight, out);
    }
}

static void affine_proj_2d_impl_q8(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, const bool last_ctx_only)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();
    Qparams& out_qparams = out.qparams();

    const int block_size = x_qparams.block_size();

    float* out_buf = g_ops_state.buf(d_out);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Qint8* xrow_data = x_data + xrow * x_st0;
            const Float16* x_row_deltas = x_qparams.row_deltas(xrow);
            const Qint8* wrow_data = w_data + wrow * w_st0;
            const Float16* w_row_deltas = w_qparams.row_deltas(wrow);

            const float dot_prod = vec_dot_product_q8(xrow_data, x_row_deltas, wrow_data, w_row_deltas, block_size, n_embd);
            float bias_scalar = fp16_to_fp32(bias_data[wrow]);

            out_buf[wrow] = dot_prod + bias_scalar;
        }

        Qint8* outrow_data = out_data + xrow * out_st0;
        quantize_row(xrow, out_buf, out_qparams, outrow_data);
    }
}

static void affine_proj_2d_impl_f16(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, const bool last_ctx_only)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < d_out; wrow++) {
            const Float16* xrow_data = x_data + xrow * x_st0;
            const Float16* wrow_data = w_data + wrow * w_st0;
            float dot_prod = vec_dot_product_f16(xrow_data, wrow_data, n_embd);
            float bias_scalar = fp16_to_fp32(bias_data[wrow]);
            out_data[xrow * out_st0 + wrow] = fp32_to_fp16(dot_prod + bias_scalar);
        }
    }
}


/// @brief Computes matmul plus bias and puts the result in out.
/// @param x A tensor of shape (n_ctx, n_embd).
/// @param w A tensor of shape (d_out, n_embd).
/// @param bias A tensor of shape (d_out).
/// @param out A tensor of shape (n_ctx, d_out).
void affine_proj_2d(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, const bool last_ctx_only=false)
{
    GTEN_ASSERT(x.is_2d());
    const int n_embd = x.size(1);
    GTEN_ASSERT(w.is_2d() && w.size(1) == n_embd);
    GTEN_ASSERT(bias.is_1d() && bias.dtype() == kFloat16);
    const int n_ctx = x.size(0);
    const int n_out = w.size(0);
    GTEN_ASSERT(out.shape_eq({n_ctx, n_out}));
    GTEN_ASSERT(x.dtype() == out.dtype())
    
    if (w.dtype() == kQint8) {
        affine_proj_2d_impl_q8(x, w, bias, out, last_ctx_only);
    } else {
        affine_proj_2d_impl_f16(x, w, bias, out, last_ctx_only);
    }
}


void affine_proj_2d_transposed_impl_f16(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, const bool last_ctx_only)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;

#ifdef GTEN_OMP
    #pragma omp parallel for collapse(2)
#endif
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        for (int wrow = 0; wrow < n_embd; wrow++) {
            const Float16* x_row_data = x_data + xrow * x_st0;
            const Float16* w_row_data = w_data + wrow * w_st0;
            const float dot_prod = vec_dot_product_f16(x_row_data, w_row_data, n_embd);
            const float bias_scalar = fp16_to_fp32(bias_data[wrow]);
            out_data[wrow * out_st0 + xrow] = fp32_to_fp16(dot_prod + bias_scalar);
        }
    }
}

void affine_proj_2d_transposed_impl_q8(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, const bool last_ctx_only)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();

    const int block_size = out.qparams().block_size();
    const int n_blocks = n_embd / block_size;

    float* out_buf = g_ops_state.buf(n_embd);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        const Qint8* x_row_data = x_data + xrow * x_st0;
        const Float16* x_ds = x_qparams.row_deltas(xrow);

        for (int wrow = 0; wrow < n_embd; wrow++) {
            const Qint8* w_row_data = w_data + wrow * w_st0;
            const Float16* w_ds = w_qparams.row_deltas(wrow);

            const float dot_prod = vec_dot_product_q8(x_row_data, x_ds, w_row_data, w_ds, block_size, n_embd);
            const float bias_scalar = fp16_to_fp32(bias_data[wrow]);
            
            out_buf[wrow] = dot_prod + bias_scalar;
        }

        Float16* out_row_deltas = out.qparams().row_deltas(xrow);

        // Quantize the row but store it as a column (i.e transposed). 
        for (int i = 0; i < n_blocks; i++) {
            const float delta = compute_quantization_delta(out_buf + i * block_size, block_size);

            out_row_deltas[i] = fp32_to_fp16(delta);

            for (int j = 0; j < block_size; j++) {
                const int wrow = i * block_size + j;
                out_data[wrow * out_st0 + xrow] = quantize(out_buf[i * block_size + j], delta);
            }
        }
    }
}


/// @brief Computes matmul plus bias and puts the (transposed) result in out.
/// @param x A tensor of shape (n_ctx, n_embd).
/// @param w A tensor of shape (d_out, n_embd).
/// @param bias A tensor of shape (d_out).
/// @param out A tensor of shape (d_out, n_ctx).
void affine_proj_2d_transposed(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, const bool last_ctx_only=false)
{
    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_out = w.size(0);

    GTEN_ASSERT(x.is_2d());
    GTEN_ASSERT(w.is_2d() && w.size(1) == n_embd);
    GTEN_ASSERT(bias.is_1d() && bias.dtype() == kFloat16 && bias.size(0) == n_out);
    GTEN_ASSERT(out.shape_eq({n_out, n_ctx}));
    GTEN_ASSERT(x.dtype() == w.dtype());

    if (w.dtype() == kQint8) {
        affine_proj_2d_transposed_impl_q8(x, w, bias, out, last_ctx_only);
    } else {
        affine_proj_2d_transposed_impl_f16(x, w, bias, out, last_ctx_only);
    }
}

static void add_impl_q8(const Tensor& x0, const Tensor& x1, Tensor& out, const bool last_ctx_only)
{
    const Qint8* x0_data = x0.data_ptr<Qint8>();
    const Qint8* x1_data = x1.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x0.size(0);
    const int n_embd = x0.size(1);
    const int st0 = x0.stride(0);

    Qparams& out_qparams = out.qparams();

    float* buf = g_ops_state.buf(n_embd * 3);
    float* x0_buf = buf;
    float* x1_buf = buf + n_embd;
    float* out_buf = buf + n_embd + n_embd;

    // auto [x0_buf, x1_buf, x2_buf] = g_ops_state.buf(n_embd, n_embd, n_embd)
    // Fewer lines of code. Just as efficient. no ptr arithmetic.

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < n_ctx; i++)
    {
        const Qint8* x0_row_data = x0_data + i * st0;
        const Qint8* x1_row_data = x1_data + i * st0;

        dequantize_row(i, x0_row_data, x0.qparams(), x0_buf);
        dequantize_row(i, x1_row_data, x1.qparams(), x1_buf);

        vec_add_f32(x0_buf, x1_buf, out_buf, n_embd);

        Qint8* out_row_data = out_data + i * st0;
        quantize_row(i, out_buf, out_qparams, out_row_data);
    }
}

static void add_impl_f16(const Tensor& x0, const Tensor& x1, Tensor& out, const bool last_ctx_only)
{
    const Float16* x0_data = x0.data_ptr<Float16>();
    const Float16* x1_data = x1.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x0.size(0);
    const int n_embd = x0.size(1);
    const int st0 = x0.stride(0);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    const Float16* x0_offs_data = x0_data + ctx_start * st0;
    const Float16* x1_offs_data = x1_data + ctx_start * st0;
    Float16* out_offs_data = out_data + ctx_start * st0;
    const int vec_size = x0.numel() - ctx_start * st0; 
    vec_add_f16(x0_offs_data, x1_offs_data, out_offs_data, vec_size);
}

static void add(const Tensor& x0, const Tensor& x1, Tensor& out, const bool last_ctx_only = false)
{
    GTEN_ASSERT(x0.is_2d());
    GTEN_ASSERT(x1.is_2d());
    GTEN_ASSERT(out.is_2d());
    GTEN_ASSERT(x0.shape_eq(x1.shape()));
    GTEN_ASSERT(x0.shape_eq(out.shape()));
    GTEN_ASSERT(x0.dtype() == x1.dtype() && x0.dtype() == out.dtype());

    if (x0.is_quantized())
    {
        add_impl_q8(x0, x1, out, last_ctx_only);
    } else {
        add_impl_f16(x0, x1, out, last_ctx_only);
    }
}


void gelu_impl_q8(const Tensor& inp, Tensor& out, const bool last_ctx_only)
{
    // TODO: Replace with lookup table.
    const int n_ctx = out.size(0);
    const int d_in = out.size(1);
    
    const Qint8* inp_data = inp.data_ptr<Qint8>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const Qparams& inp_qparams = inp.qparams();
    Qparams& out_qparams = out.qparams();

    float* buf = g_ops_state.buf(2 * d_in);
    float* inp_buf = buf;
    float* out_buf = buf + d_in;

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int i = ctx_start; i < n_ctx; ++i) {
        const Qint8* inp_row_data = inp_data + i * d_in;
        dequantize_row(i, inp_row_data, inp_qparams, inp_buf);

        for (int j = 0; j < d_in; j++) {
            float x = inp_buf[j];
            float res = 0.5 * x 
                        * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                        * (x + 0.044715f * std::pow(x, 3.0f))));
            out_buf[j] = res;
        }

        Qint8* out_row_data = out_data + i * d_in;
        quantize_row(i, out_buf, out_qparams, out_row_data);
    }
}


void gelu_impl_f16(const Tensor& inp, Tensor& out, const bool last_ctx_only)
{
    // TODO: Replace with lookup table.
    const int n_ctx = out.size(0);
    const int n_embd = out.size(1);
    
    const Float16* inp_data = inp.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int numel = inp.numel();
    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int i = ctx_start * n_embd; i < numel; ++i) {
        float x = fp16_to_fp32(inp_data[i]);
        float res = 0.5 * x 
                        * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                        * (x + 0.044715f * std::pow(x, 3.0f))));
        out_data[i] = fp32_to_fp16(res);
    }
}

/// @brief Computes GELU function on the input and puts the result in out.
/// @param inp Input tensor.
/// @param out Output tensor.
/// TODO: Replace w lookup table.
void gelu(const Tensor& x, Tensor& out, const bool last_ctx_only = false)
{
    GTEN_ASSERT(x.is_2d() && x.shape_eq(out.shape()));
    GTEN_ASSERT(x.dtype() == out.dtype());

    if (x.is_quantized()) {
        gelu_impl_q8(x, out, last_ctx_only);
    } else {
        gelu_impl_f16(x, out, last_ctx_only);
    }
}

static void layer_norm_impl_q8(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, bool last_ctx_only)
{
    const Qint8* x_data = x.data_ptr<Qint8>();
    const Float16* w_data = w.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Qint8* out_data = out.data_ptr<Qint8>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int st0 = x.stride(0);

    // return Qparam.
    const Qparams& x_qparams = x.qparams();
    const Qparams& w_qparams = w.qparams();
    Qparams& out_qparams = out.qparams();

    float* buf = g_ops_state.buf(2 * n_embd);
    float* inp_buf = buf;
    float* out_buf = buf + n_embd;

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        const Qint8* inp_row_data = x_data + xrow * st0;
        Qint8* out_row_data = out_data + xrow * st0;

        dequantize_row(xrow, inp_row_data, x_qparams, inp_buf);

        vec_layer_norm_f32(inp_buf, n_embd, w_data, bias_data, out_buf);

        quantize_row(xrow, out_buf, out_qparams, out_data + xrow * n_embd);
    }
}

static void layer_norm_impl_f16(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, bool last_ctx_only)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int st0 = x.stride(0);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int xrow = ctx_start; xrow < n_ctx; xrow++) {
        const Float16* inp_row_data = x_data + xrow * st0;
        Float16* out_row_data = out_data + xrow * st0;

        vec_layer_norm_f16(inp_row_data, st0, w_data, bias_data, out_row_data);
    }
}

/// @brief Computes affine layer-normalization.
/// @param x Input tensor of shape (n_ctx, n_embd).
/// @param w Weight tensor of shape (n_embd).
/// @param bias Tensor of shape (n_embd).
/// @param out Tensor of shape (n_ctx, n_embd).
/// @param last_ctx_only If true, compute norm for the last embedding in input only.
static void layer_norm(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, bool last_ctx_only = false)
{
    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);

    GTEN_ASSERT(x.is_2d());
    GTEN_ASSERT(w.is_1d() && w.dtype() == kFloat16 && w.size(0) == n_embd);
    GTEN_ASSERT(bias.is_1d() && bias.dtype() == kFloat16 && bias.size(0) == n_embd);
    GTEN_ASSERT(out.shape_eq(x.shape()));
    GTEN_ASSERT(x.dtype() == out.dtype());

    if (x.is_quantized()) {
        layer_norm_impl_q8(x, w, bias, out, last_ctx_only);
    } else {
        layer_norm_impl_f16(x, w, bias, out, last_ctx_only);
    }
}

/*

[Q|K]: [physical_layout](n_ctx, n_head, d_head)
[
  [[----1----],        
   [----2----]],

  [[----3----],
   [----4----]]
]

[Q|K]: [logical_layout](n_head, n_ctx, d_head)
Note: K should be (n_head, d_head, n_ctx) but it is transposed for efficiency.
[
  [[----1----],        
   [----3----]], <---

  [[----2----],  <---
   [----4----]]
]

Q @ K: [logical_layout]
[n_head=2, n_ctx=2, d_head]
[                    [                       [                          [
  [[----A----],        [[----A----],           [[A.A, A.B],               [[A.A, ___],
   [----B----]],        [----B----]],           [B.A, B.B]],               [B.A, B.B]],
                   @                     =                      mask =>
  [[----C----],        [[----C----],           [[C.C, C.D],               [[C.C, ___],
   [----D----]]         [----D----]]            [D.C, D.D]]                [D.C, D.D]]
]                    ]                       ]                          ]

-------------------------------------------------------------
[n_head=2, n_ctx=3, d_head]
[                    [                       [                             [
  [[----A----],        [[----A----],           [[A.A, A.B, A.E],             [[A.A, ___, _._],
   [----B----],         [----B----],            [B.A, B.B, B.E],              [B.A, B.B, _._],
   [----E----]],        [----E----]],           [E.A, E.B, E.E]],             [E.A, E.B, E.E]],
                   @                     =                         mask =>
  [[----C----],        [[----C----],           [[C.C, C.D, C.F],             [[C.C, ___, _._],
   [----D----],         [----D----],            [D.C, D.D, D.F],              [D.C, D.D, _._],
   [----F----]],        [----F----]],           [F.C, F.D, F.F]],             [F.C, F.D, F.F]],
]                    ]                       ]                             ]

*/

void qk_masked_softmax_f16(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const bool last_ctx_only) {
    const Float16* q_data = q.data_ptr<Float16>();
    const Float16* k_data = k.data_ptr<Float16>();
    Float16* out_data = qk_out.data_ptr<Float16>();

    const int nhead = q.size(0);
    const int n_ctx = q.size(1);
    const int d_head = q.size(2);

    const int qst0 = q.stride(0);
    const int qst1 = q.stride(1);
    const int kst0 = k.stride(0);
    const int kst1 = k.stride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    float* out_buf = g_ops_state.buf(n_ctx);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;

    for (int h = 0; h < nhead; h++) {
        for (int qrow = ctx_start; qrow < n_ctx; qrow++) {
            // For each vector in the current head of Q, we only compute the
            // dot_products that are not subsequently masked. That reduces the
            // number of dot products on each head by half.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                const Float16* qrow_data = q_data + (h * qst0 + qrow * qst1);
                const Float16* kcol_data = k_data + (h * kst0 + kcol * kst1); // col_data is contigous.
                const float dot_prod = vec_dot_product_f16(qrow_data ,kcol_data, d_head);
                out_buf[kcol] = dot_prod * scale_factor;
            }

            /// Masking operation.
            /// TODO: There is a potential optimization here. Instead of masking by setting
            /// the masked positions to -inf and then computing softmax, we can instead set
            /// the masked positions to zero and skipping computing softmax over masked
            /// positions because they map to zero after softmax is applied over them. This
            /// reduces the number exps and writes by 0.5*n_head*n_ctx*n_ctx.
            const int kcol_start = qrow + 1;
            for (int kcol = kcol_start; kcol < n_ctx; kcol++) {
                out_buf[kcol] = -std::numeric_limits<float>::infinity();  // zero TODO
            }

            // SOFTMAX
            // We use the function sm(x - x_max) = e^(x - x_max) / sum(e^(x - xmax)) instead
            // of the original function sm(x) = e^(x) / sum(e^(x)) because the former is more
            // numerically stable as it prevent overflows in the exponent. The output results
            // is the same in both cases.
            float max = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                if (x > max)
                    max = x;
            }

            float sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                const float exp_val = std::exp(x - max);
                out_buf[i] = exp_val;
                sum_exp += exp_val;
            }

            for (int i = 0; i < n_ctx; i++) {
                const float qkw = out_buf[i];
                out_buf[i] = qkw / sum_exp;
            }

            for (int i = 0; i < n_ctx; i++) {
                out_data[h * qkst0 + qrow * qkst1 + i] = fp32_to_fp16(out_buf[i]);
            }
        }
    }
}


void qk_masked_softmax_q8(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, const bool last_ctx_only)
{
    const Qint8* q_data = q.data_ptr<Qint8>();
    const Qint8* k_data = k.data_ptr<Qint8>();
    Qint8* out_data = qk_out.data_ptr<Qint8>();

    const int nhead = q.size(0);
    const int n_ctx = q.size(1);
    const int d_head = q.size(2);

    const int qst0 = q.stride(0);
    const int qst1 = q.stride(1);
    const int kst0 = k.stride(0);
    const int kst1 = k.stride(1);
    const int qkst0 = qk_out.stride(0);
    const int qkst1 = qk_out.stride(1);

    const Qparams& q_params = q.qparams();
    const Qparams& k_params = k.qparams();
    const int block_size = k_params.block_size();

    float* out_buf = g_ops_state.buf(n_ctx);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;

    for (int qrow = ctx_start; qrow < n_ctx; qrow++) {
       for (int h = 0; h < nhead; h++) {
            // `kcol_max` represents number of the dot products that are not subsequently masked.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                
                const Qint8* qrow_data = q_data + (h * qst0 + qrow * qst1);
                const Qint8* kcol_data = k_data + (h * kst0 + kcol * kst1); // col_data is contigous.

                // blck size can be larger than head size.
                // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                // [0, 0, 0, 0, 0, | 0, 0, 0, 0, 0]

                // deltas for the row of the current head.
                const Float16* q_delta = q_params.row_deltas(qrow) + h * d_head / q_params.block_size();
                const Float16* k_delta = k_params.row_deltas(kcol) + h * d_head / k_params.block_size();

                const float dot_prod = vec_dot_product_q8(qrow_data, q_delta, kcol_data, k_delta, block_size, d_head);
                // const float dot_prod = dot_product_deprecated(qrow_data, q_delta, kcol_data, k_delta, block_size, d_head);
                out_buf[kcol] = dot_prod * scale_factor;
            }

            /// Masking operation.
            const int kcol_start = qrow + 1;
            for (int kcol = kcol_start; kcol < n_ctx; kcol++) {
                out_buf[kcol] = -std::numeric_limits<float>::infinity();  // zero TODO
            }

            // SOFTMAX
            // Max
            float max = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                if (x > max)
                    max = x;
            }

            // out[i] = exp(xi - xi_max)
            float sum_exp = 0;
            for (int i = 0; i < n_ctx; i++) {
                const float x = out_buf[i];
                const float exp_val = std::exp(x - max);
                out_buf[i] = exp_val;
                sum_exp += exp_val;
            }

            // out[i] = out[i] / sum[exp(xi - xi_max)]
            for (int i = 0; i < n_ctx; i++) {
                const float qkw = out_buf[i];
                out_buf[i] = qkw / sum_exp;
            }

            const float delta = 1.0f / 127.0f;  // quantization delta where amax=1 because it is softmaxed.
            for (int i = 0; i < n_ctx; i++) {
                out_data[h * qkst0 + qrow * qkst1 + i] = quantize(out_buf[i], delta);
            }
        }
    }
}


/// @brief Computes attention. i.e (QK @ V).
/// @param qk A tensor of shape (nhead, n_ctx, n_ctx).
/// @param v A tensor of shape (n_head, d_head, n_ctx).
/// @param qkv_out A tensor of shape (n_ctx, n_head, d_head).
void qkv_matmul_f16(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const bool last_ctx_only)
{
    const Float16* qk_data = qk.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();
    Float16* out_data = qkv_out.data_ptr<Float16>();

    const int n_ctx = qk.size(1);
    const int nhead = v.size(0);
    const int dhead = v.size(1);

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int vst0 = v.stride(0);
    const int vst1 = v.stride(1);
    // qkv shape: [ctx, head, dhead]
    const int qkv_st0 = qkv_out.stride(0);
    const int qkv_st1 = qkv_out.stride(1);

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    // out: [c, h, d]
    for (int h = 0; h < nhead; h++) {
        for (int qkr = ctx_start; qkr < n_ctx; qkr++) {
            for (int vc = 0; vc < dhead; vc++) {
                const Float16* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);
                const Float16* vc_data = v_data + (h * vst0 + vc * vst1);
                const float dot_prod = vec_dot_product_f16(qkr_data, vc_data, n_ctx);
                out_data[h * qkv_st1 + qkr * qkv_st0 + vc] = fp32_to_fp16(dot_prod);
            }
        }
    }
}


void qkv_matmul_q8(const Tensor& qk, const Tensor& v, Tensor& qkv_out, const bool last_ctx_only)
{
    const Qint8* qk_data = qk.data_ptr<Qint8>();
    const Qint8* v_data = v.data_ptr<Qint8>();
    Qint8* out_data = qkv_out.data_ptr<Qint8>();

    const int n_ctx = qk.size(1);
    const int nhead = v.size(0);
    const int dhead = v.size(1);

    const int qkst0 = qk.stride(0);
    const int qkst1 = qk.stride(1);
    const int vst0 = v.stride(0);
    const int vst1 = v.stride(1);
    // qkv shape: [ctx, head, dhead]
    const int qkv_st0 = qkv_out.stride(0);
    const int qkv_st1 = qkv_out.stride(1);

    const float delta = 1.0f / 127.0f;
    // out: [c, h, d] nhead, dhead
    // qkv_st1: dhead
    // v: n_head, d_head, n_ctx

    Qparams& qkv_out_qparams = qkv_out.qparams();
    const Qparams& v_qparams = v.qparams();

    float* buf = g_ops_state.buf(n_ctx + n_ctx + nhead * dhead);
    float* qk_buf = buf;
    float* v_buf = buf + n_ctx;
    float* out_buf = buf + n_ctx + nhead * dhead;

    const int ctx_start = last_ctx_only ? n_ctx - 1 : 0;
    for (int qkr = ctx_start; qkr < n_ctx; qkr++) {
        for (int h = 0; h < nhead; h++) {
            const Qint8* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);  // qk_row_data
            
            dequantize_row_scale(qkr_data, delta, qk_buf, n_ctx);

            for (int vc = 0; vc < dhead; vc++) {
                const Qint8* vc_data = v_data + (h * vst0 + vc * vst1);

                const int col_idx = h * dhead + vc;
                dequantize_col(col_idx, vc_data, v_qparams, v_buf, n_ctx);

                 const float dot_prod = vec_dot_product_f32(qk_buf, v_buf, n_ctx);
                out_buf[h * qkv_st1 + vc] = dot_prod;
            }
        }

        Qint8* out_row_data = out_data + qkr * qkv_st0;
        quantize_row(qkr, out_buf, qkv_out_qparams, out_row_data);
    }
}


static void qkv_attn_impl(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, int max_ctx, const bool last_ctx_only)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);
    const int nhead = qk.size(0);
    const int dhead = n_embd / nhead;

    const Tensor q0 = q.view({n_ctx, nhead, dhead}).permute({1, 0, 2});
    const Tensor k0 = k.view({n_ctx, nhead, dhead}).permute({1, 0, 2});
    qk.set_strides({max_ctx * max_ctx, max_ctx, 1});


    const float scale_factor = 1.0f / std::sqrt((float)dhead);

    if (qk.is_quantized()) {
        ops::qk_masked_softmax_q8(q0, k0, qk, scale_factor, last_ctx_only);
    } else {
        ops::qk_masked_softmax_f16(q0, k0, qk, scale_factor, last_ctx_only);
    }
    

    Tensor v0 = v.view({nhead, dhead, n_ctx});
    v0.set_strides({dhead * max_ctx, max_ctx, 1});

    Tensor qkv0 = qkv.view({n_ctx, nhead, dhead});
    if (qk.is_quantized()) {
        ops::qkv_matmul_q8(qk, v0, qkv0, last_ctx_only);
    } else {
        ops::qkv_matmul_f16(qk, v0, qkv0, last_ctx_only);
    }
}

/// @brief Computes attention scale(Q @ K)V.
/// @param q Tensor of shape (n_ctx, n_embd).
/// @param k Tensor of shape (n_ctx, n_embd).
/// @param v Tensor of shape (n_embd, n_ctx).
/// @param qk Tensor of shape (nhead, n_ctx, n_ctx).
/// @param qkv Output tensor of shape (n_ctx, n_embd).
/// @param max_ctx Maximum context size. It allows skipping computations for
///   cached qk values when ctx_offs is provided. 
static void qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, const int max_ctx, const bool last_ctx_only=false)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);
    const int n_head = qk.size(0);

    GTEN_ASSERT(q.is_2d() && q.shape_eq(k.shape()));
    GTEN_ASSERT(k.is_2d());
    GTEN_ASSERT(v.is_2d() && v.shape_eq({n_embd, n_ctx}));
    GTEN_ASSERT(qk.is_3d() && qk.shape_eq({n_head, n_ctx, n_ctx}));
    GTEN_ASSERT(qkv.is_2d());
    GTEN_ASSERT(q.dtype() == k.dtype() && k.dtype() == v.dtype() && v.dtype() == qk.dtype() && qk.dtype() == qkv.dtype())
    GTEN_ASSERT(max_ctx > 0 && max_ctx >= n_ctx);

    qkv_attn_impl(q, k, v, qk, qkv, max_ctx, last_ctx_only);
}

} // namespace ops
} // namespace gten
