#include <cmath>
#include <iostream>
#include <thread>

#include "utils.h"
#include "modules.h"


#define GTEN_CHECK_DTYPE_EQUAL(inp_dtype, expected_dtype)     \
    GTEN_ASSERT(                                              \
        inp_dtype == expected_dtype,                          \
        "Expected tensor to have dtype=%s but got dtype=%s.", \
        dtype_str(expected_dtype),                            \
        dtype_str(inp_dtype))

#define GTEN_CHECK_NDIMS_EQUAL(inp_ndims, expected_ndims)    \
    GTEN_ASSERT(                                             \
        inp_ndims == expected_ndims,                         \
        "Expected a %d-dim tensor but got a %d-dim tensor.", \
        expected_ndims,                                      \
        inp_ndims)

#define GTEN_CHECK_DIMSIZE_EQUAL(dim, inp_dimsize, expected_dimsize)  \
    GTEN_ASSERT(                                                      \
        inp_dimsize == expected_dimsize,                              \
        "Expected tensor to have dim-%d=%d but got dim-%d=%d.",       \
        dim, expected_dimsize, dim, inp_dimsize)

// #define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)                  
//     GTEN_ASSERT(                                                             
//         inp_ctx_size <= max_ctx_size,                                        
//         "The given input's context size=%d exceeds max context size of %d.", 
//         inp_ctx_size,                                                        
//         max_ctx_size)

#define GTEN_CHECK_INP_CTX_SIZE(inp_ctx_size, max_ctx_size)


// Activate max ctx.
// Check values not negative.
// Replace gelu w tables.


namespace gten {

Embedding::Embedding(int n_vocab, int d_embed, int max_ctx)
    : weight{Tensor({n_vocab, d_embed}, kFloat16)},
      emb_acv_{Tensor({max_ctx, d_embed}, kFloat16)},
      proj_acv_{Tensor({n_vocab}, kFloat32)},
      max_ctx_(max_ctx)
{
}

Tensor Embedding::forward(const Tensor& tokens)
{
    Timer timer{&exec_time_emb_ms_};

    GTEN_CHECK_DTYPE_EQUAL(tokens.dtype(), kInt32);
    GTEN_CHECK_NDIMS_EQUAL(tokens.ndims(), 1);
    GTEN_CHECK_INP_CTX_SIZE(tokens.numel(), max_ctx_);
    
    return forward_impl(tokens);
}

Tensor Embedding::forward_impl(const Tensor& inp)
{
    const int d_embed = weight.size(1);
    emb_acv_.resize({inp.numel(), d_embed});
    Float16* out_data = emb_acv_.data_ptr<Float16>();
    const Int32* inp_data = inp.data_ptr<Int32>();
    const Float16* weight_data = weight.data_ptr<Float16>();

    if (emb_acv_cached_) {
        const int token_i = inp.numel() - 1;
        const int emb_i = inp_data[token_i] * d_embed;
        const void *src = reinterpret_cast<const void*>(weight_data + emb_i);
        void *dest = reinterpret_cast<void*>(out_data + token_i * d_embed);
        std::memcpy(dest, src, d_embed * weight.itemsize());
    }
    else {
        emb_acv_cached_ = true;

        const int ntokens = inp.numel();
        for (int token_i = 0; token_i < ntokens; token_i++) {
            int emb_i = inp_data[token_i] * d_embed;
            const void *src = reinterpret_cast<const void*>(weight_data + emb_i);
            void *dest = reinterpret_cast<void*>(out_data + token_i * d_embed);
            std::memcpy(dest, src, d_embed * weight.itemsize());
        }
    }

    return emb_acv_;    
}


Tensor Embedding::forward_proj(const Tensor &inp)
{
    Timer timer(&exec_time_proj_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight.size(1));
    GTEN_CHECK_INP_CTX_SIZE(inp.size(0), max_ctx_);

    return forward_proj_impl(inp);
}


// static Float32 dot_product(const Float16* a, const Float16* b, int vec_size)
// {
//     Float32 dot_prod = 0.0f;
//     for (int i = 0; i < vec_size; ++i)
//     {
//         // dot_prod += fpcvt_to_fp32(a[i]) * fpcvt_to_fp32(b[i]);
//         dot_prod += G_fp16_to_fp32_cache[a[i]] * G_fp16_to_fp32_cache[b[i]];
//     }
//     return dot_prod;
// }

void flat_matmul(const Tensor& a, const Tensor& b, Tensor& out, int a_offs, int out_offs)
{
    // checks. dims
    const int d0 = a.size(0);
    const int d1 = a.size(1);
    const int d2 = b.size(0);

    const Float16* a_data = a.data_ptr<Float16>();
    const Float16* b_data = b.data_ptr<Float16>();
    Float32* out_data = out.data_ptr<Float32>(); 

    for (int i = a_offs/d1; i < d0; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d1; ++k) {
                out_data[out_offs + j] = fpcvt_to_fp32(a_data[i*d1 + k]) * fpcvt_to_fp32(b_data[j*d2 + k]);
            }
        }
    }
}

// how does offsets work
// offset: num_elements
// dims: affects dim0
// row_idx = offset/row_length
// [a, b], [b, c]


Tensor Embedding::forward_proj_impl(const Tensor& inp)
{
    // Output probs must be float32.
    Float32* out_data = proj_acv_.data_ptr<Float32>();
    const Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* weight_data = weight.data_ptr<Float16>();

    const int n_vocab = weight.size(0);
    const int ctx_i = inp.size(0) - 1;
    const int d_embed = inp.size(1);

    // (1, n_embed) x (n_vocab, n_embed)
    const int inp_offset = ctx_i * d_embed;

    for (int emb_i = 0; emb_i < n_vocab; emb_i++)
    {
        Vec_f32x8 dot_accum = { vec_f32x8_setzero() };
        for (int i = 0; i < d_embed; i += 8) {
            Vec_f32x8 x = vec_f32x8_load(inp_data + inp_offset + i);
            Vec_f32x8 w = vec_f32x8_load(weight_data + (emb_i * d_embed + i));
            dot_accum = vec_f32x8_fma(x, w, dot_accum);
        }
        out_data[emb_i] = vec_f32x8_sum(dot_accum);
    }

    return proj_acv_;  
}

PosEmbedding::PosEmbedding(int max_ctx, int d_embed)
    : weight{Tensor({max_ctx, d_embed}, kFloat16)}, max_ctx_(max_ctx)
{
}

Tensor PosEmbedding::forward(int n_ctx)
{
    GTEN_CHECK_INP_CTX_SIZE(n_ctx, max_ctx_);

    Timer timer{&exec_time_ms_};
    
    return forward_impl(n_ctx);
}

Tensor PosEmbedding::forward_impl(int n_ctx)
{
    const Float16* weight_data = weight.data_ptr<Float16>();

    void* src_ptr = (void*)weight_data;
    const int d_embed = weight.size(1);

    // Shares the data with the weight.
    Tensor acv{src_ptr, {n_ctx, d_embed}, weight.dtype()};

    return acv;
}

LayerNorm::LayerNorm(int max_ctx, int d_embed)
    : weight{Tensor({d_embed}, kFloat16)},
      bias{Tensor({d_embed}, kFloat16)},
      acv_{Tensor({max_ctx, d_embed}, kFloat16)},
      max_ctx_{max_ctx}
{
}


Tensor LayerNorm::forward(const Tensor &inp)
{
    Timer timer(&exec_time_ms_);

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    const int d_embed = weight.size(0);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), d_embed);

    return forward_impl(inp);
}

template<typename T>
T* get_ptr() {
    return reinterpret_cast<T*>(nullptr);
}

Tensor LayerNorm::forward_impl(const Tensor &inp)
{
    acv_.resize({inp.size(0), inp.size(1)});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* inp_data = inp.data_ptr<Float16>();
    const Float16* weight_data = weight.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();

    const int n_ctx = inp.size(0);
    const int d_embed = weight.size(0);

    if (acv_cached_)
    {
        const int ctx_offset = (n_ctx - 1) * d_embed;
        // Mean calculation.
        Float32 mean_accum = 0.0f;
        for (int i = 0; i < d_embed; i++)
            mean_accum += fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
        Float32 mean = mean_accum / (Float32)d_embed;

        // Standard deviation calculation.
        Float32 variance_accum = 0.0f;
        for (int i = 0; i < d_embed; i++) {
            Float32 x = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
            variance_accum += (x - mean) * (x - mean);
        }
        Float32 std_dev = std::sqrt(variance_accum / (Float32)d_embed);

        // Normalization.
        for (int i = 0; i < d_embed; i++) {
            Float32 unnormalized = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
            Float32 w = fpcvt_to_fp32<Float16>(weight_data[i]);
            Float32 b = fpcvt_to_fp32<Float16>(bias_data[i]);
            Float32 normalized = ((unnormalized - mean) / (std_dev + eps_)) * w + b;
            acv_data[i + ctx_offset] = fpcvt_from_fp32<Float16>(normalized);
        }
    }
    else
    {
        acv_cached_ = true;

        for (int ctx_i = 0; ctx_i < n_ctx; ctx_i++)
        {
            const int ctx_offset = ctx_i * d_embed;

            // Mean calculation.
            Float32 mean_accum = 0.0f;
            for (int i = 0; i < d_embed; i++)
                mean_accum += fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
            Float32 mean = mean_accum / (Float32)d_embed;

            // Standard deviation calculation.
            Float32 variance_accum = 0.0f;
            for (int i = 0; i < d_embed; i++) {
                Float32 x = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
                variance_accum += (x - mean) * (x - mean);
            }
            Float32 std_dev = std::sqrt(variance_accum / (Float32)d_embed);

            // Normalization.
            for (int i = 0; i < d_embed; i++) {
                Float32 x = fpcvt_to_fp32<Float16>(inp_data[i + ctx_offset]);
                Float32 w = fpcvt_to_fp32<Float16>(weight_data[i]);
                Float32 b = fpcvt_to_fp32<Float16>(bias_data[i]);
                // Epsilon added to standard deviation prevents div by zero.
                Float32 normalized = ((x - mean) / (std_dev + eps_)) * w + b;
                acv_data[i + ctx_offset] = fpcvt_from_fp32<Float16>(normalized);
            }
        }
    }

    return acv_;
}


GELU::GELU(int max_ctx, int d_out, bool cache_ctx_acv)
    : acv_{Tensor({max_ctx, d_out}, kFloat16)}, cache_acv_{cache_ctx_acv}
{
}

Tensor GELU::forward(const Tensor& inp)
{
    Timer timer{&exec_time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    // GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), n_out);
    // Assert inp numel = acv numel.
    // Resize acv to inp shape.

    return forward_impl(inp);
}

Tensor GELU::forward_impl(const Tensor& inp)
{
    // TODO: Replace with lookup table.
    const int n_ctx = inp.size(0);
    const int d_out = acv_.size(1);

    acv_.resize({n_ctx, d_out});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* inp_data = inp.data_ptr<Float16>();


    int start_idx;
    if (cache_acv_ && acv_cached_) {
        start_idx = (n_ctx - 1) * d_out;
    } else {
        start_idx = 0;
        acv_cached_ = true;
    }

    const int ne = inp.numel();
    for (int i = start_idx; i < ne; ++i) {
        Float32 x = fpcvt_to_fp32<Float16>(inp_data[i]);
        Float32 res = 0.5 * x 
                            * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                            * (x + 0.044715f * std::pow(x, 3.0f))));
        acv_data[i] = fpcvt_from_fp32<Float16>(res);
    }
    
    return acv_;
}

Residual::Residual(int max_ctx, int d_out)
    : acv_{Tensor({max_ctx, d_out}, kFloat16)}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1)
{
    Timer timer{&exec_time_ms_};

    GTEN_CHECK_DTYPE_EQUAL(inp0.dtype(), inp1.dtype());
    GTEN_CHECK_NDIMS_EQUAL(inp0.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(inp1.ndims(), 2);
    // TODO: Check shape inp1 == inp0

    return forward_impl(inp0, inp1);
}

Tensor Residual::forward_impl(const Tensor& inp0, const Tensor& inp1)
{
    const int n_ctx = inp0.size(0);
    const int d_embed = inp0.size(1);

    acv_.resize({n_ctx, d_embed});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* inp0_data = inp0.data_ptr<Float16>();
    const Float16* inp1_data = inp1.data_ptr<Float16>();

    if (acv_cached_) {
        uint32_t n_iter = d_embed;
        uint32_t offset = inp0.numel() - d_embed;
        for (uint32_t i = 0; i < n_iter; i += 8) {
            Vec_f32x8 x0 = vec_f32x8_load(inp0_data + offset + i);
            Vec_f32x8 x1 = vec_f32x8_load(inp1_data + offset + i);
            Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
            vec_f32x8_store(x_sum, acv_data + offset + i);
        }
    }
    else {
        acv_cached_ = true;

        int n_iter = inp0.numel();
        for (int i = 0; i < n_iter; i += 8) {
            Vec_f32x8 x0 = vec_f32x8_load(inp0_data + i);
            Vec_f32x8 x1 = vec_f32x8_load(inp1_data + i);
            Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
            vec_f32x8_store(x_sum, acv_data + i);
        }
    }

    return acv_;
}

Linear::Linear(int d_in, int d_out, int max_ctx, bool cache_acv, bool transpose_out)
    : weight{Tensor({d_out, d_in}, kFloat16)},
      bias{Tensor({d_out}, kFloat16)},
      acv_{Tensor({max_ctx, d_out}, kFloat16)},
      cache_acv_{cache_acv}, transpose_out_{transpose_out}
{
}

Tensor Linear::forward(const Tensor &inp)
{
    Timer timer{&exec_time_ms_};

    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), weight.size(1));

    return forward_impl(inp);
}

Tensor Linear::forward_impl(const Tensor& inp)
{
    const int n_ctx = inp.size(0);
    const int d_embed = inp.size(1);
    const int d_out = weight.size(0);
    
    acv_.resize({n_ctx, d_out});
    Float16* acv_data = acv_.data_ptr<Float16>();
    const Float16* weight_data = weight.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    const Float16* inp_data = inp.data_ptr<Float16>();

    int ctx_start_idx;
    if (cache_acv_ && acv_cached_) {
        ctx_start_idx = n_ctx - 1;
    } else {
        ctx_start_idx = 0;
        acv_cached_ = true;
    }

    for (int ctx_idx = ctx_start_idx; ctx_idx < n_ctx; ctx_idx++) {
        for (int out_idx = 0; out_idx < d_out; out_idx++) {
            Vec_f32x8 dot_accum = vec_f32x8_setzero();
            for (int i = 0; i < d_embed; i += 8) {
                Vec_f32x8 x0 = vec_f32x8_load(inp_data + (ctx_idx * d_embed + i));
                Vec_f32x8 x1 = vec_f32x8_load(weight_data + (out_idx * d_embed + i));
                dot_accum = vec_f32x8_fma(x0, x1, dot_accum);
            }
            Float32 bias = fpcvt_to_fp32<Float16>(bias_data[out_idx]);
            Float32 res =  vec_f32x8_sum(dot_accum) + bias;
            const int out_i = transpose_out_ ? out_idx * n_ctx + ctx_idx : ctx_idx * d_out + out_idx;
            acv_data[out_i] = fpcvt_from_fp32<Float16>(res);
        }
    }

    return acv_;
}


static void qk_attn_matmul(const Tensor& q, const Tensor& k, Tensor& qk_out,
                           const int n_head, const bool mask_output, int ctx_offset=0)
{
    const int q_ctx = q.size(0);
    const int d_embed = q.size(1);
    const int k_ctx = k.size(0);
    const int d_head = d_embed / n_head;
    const Float32 scale_factor = 1.0f / std::sqrt((Float32)d_head);

    const Float16* q_data = q.data_ptr<Float16>();
    const Float16* k_data = k.data_ptr<Float16>();
    Float16* qk_data = qk_out.data_ptr<Float16>();

    for (int head = 0; head < n_head; head++) {
        for (int q_row = ctx_offset; q_row < q_ctx; q_row++) {
            const int k_max = mask_output ? q_row + 1 : k_ctx;
            for (int k_col = 0; k_col < k_max; k_col++) {
                Vec_f32x8 dot_accum = { vec_f32x8_setzero() };
                for (int i = 0; i < d_head; i += 8) {
                    int q_idx = head * d_head + q_row * d_embed + i;
                    int k_idx = head * d_head + k_col * d_embed + i;
                    Vec_f32x8 qw = vec_f32x8_load(q_data + q_idx);
                    Vec_f32x8 kw = vec_f32x8_load(k_data + k_idx);
                    dot_accum = vec_f32x8_fma(qw, kw, dot_accum);
                }
                int qk_data_i = head * q_ctx * k_ctx + q_row * k_ctx + k_col;
                qk_data[qk_data_i] = fpcvt_from_fp32<Float16>(vec_f32x8_sum(dot_accum) * scale_factor);
            }
        }
    }

    if (mask_output) {
        for (int head = 0; head < n_head; head++) {
            for (int q_row = ctx_offset; q_row < q_ctx; q_row++) {
                const int k_start = q_row + 1;
                for (int k_col = k_start; k_col < k_ctx; k_col++) {
                    int qk_data_i = head * q_ctx * k_ctx + q_row * k_ctx + k_col;
                    // Use memcpy?
                    qk_data[qk_data_i] = fpcvt_from_fp32<Float16>(-std::numeric_limits<float>::infinity());
                }
            }
        }
    }
}

static void qk_softmax(Tensor& qk_acv, int n_heads, int ctx_offset=0)
{
    Float16* qk_data = qk_acv.data_ptr<Float16>();

    const int q_ctx = qk_acv.size(1);
    const int k_ctx = qk_acv.size(2);

    for (int head = 0; head < n_heads; head++) {
        for (int q_row = ctx_offset; q_row < q_ctx; q_row++)
        {
            Float32 max = -std::numeric_limits<float>::infinity();

            const int base_idx = head * q_ctx * k_ctx + q_row * k_ctx;

            for (int i = 0; i < k_ctx; i++) {
                Float32 x = fpcvt_to_fp32(qk_data[base_idx + i]);
                if (x > max)
                    max = x;
            }

            Float32 sum_exp = 0;
            for (int i = 0; i < k_ctx; i++) {
                Float32 x = fpcvt_to_fp32(qk_data[base_idx + i]);
                Float32 exp_val = std::exp(x - max);
                qk_data[base_idx + i] = fpcvt_from_fp32<Float16>(exp_val);
                sum_exp += exp_val;
            }

            for (int i = 0; i < k_ctx; i++) {
                Float32 qkw = fpcvt_to_fp32(qk_data[base_idx + i]);
                qk_data[base_idx + i] = fpcvt_from_fp32<Float16>(qkw / sum_exp);
            }
        }
    }
}

static void qkv_attn_matmul(const Tensor& qk, const Tensor& v, Tensor& qkv_acv, int n_head, int ctx_offset=0)
{
    const Float16* qk_data = qk.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();
    Float16* qkv_data = qkv_acv.data_ptr<Float16>();

    const int q_ctx = qk.size(1);
    const int kv_ctx = qk.size(2);
    const int d_embed = v.size(1);
    const int d_head = d_embed / n_head;

    const int vec_niter = kv_ctx / 8;
    const int vec_n_ctx = vec_niter * 8;

    for (int head = 0; head < n_head; head++)
    {
        for (int qk_row = ctx_offset; qk_row < q_ctx; qk_row++)
        {
            for (int v_row = 0; v_row < d_head; v_row++)
            {
                float dot_prod = 0.0f;
                Vec_f32x8 dot_prod_accum = { vec_f32x8_setzero() };

                for (int i = 0; i < vec_n_ctx; i += 8)
                {
                    const int qk_i = head * q_ctx * kv_ctx + qk_row * kv_ctx + i;
                    const int v_i = head * kv_ctx * d_head + v_row*kv_ctx + i;

                    Vec_f32x8 qkw = vec_f32x8_load(qk_data + qk_i);
                    Vec_f32x8 vw = vec_f32x8_load(v_data + v_i);
                    dot_prod_accum = vec_f32x8_fma(qkw, vw, dot_prod_accum);
                }
                dot_prod = vec_f32x8_sum(dot_prod_accum);

                for (int i = vec_n_ctx; i < kv_ctx; i++)
                {
                    const int qk_i = head * q_ctx * kv_ctx + qk_row * kv_ctx + i;
                    const int v_i = head * kv_ctx * d_head + v_row*kv_ctx + i;
                    Float32 qkw = fpcvt_to_fp32(qk_data[qk_i]);
                    Float32 vw = fpcvt_to_fp32(v_data[v_i]);
                    dot_prod += qkw*vw;
                }

                int qkv_data_i = head * d_head + qk_row * d_embed + v_row;
                qkv_data[qkv_data_i] = fpcvt_from_fp32<Float16>(dot_prod);
            }
        }
    }
}


// Like fn above where v is not transposed.
// TODO: Implement Linear module where output is cached and transposed.
static void qkv_attn_matmul_nt(const Tensor& qk, const Tensor& v, Tensor& qkv_acv, int n_heads, int ctx_offset=0)
{
    const Float16* qk_data = qk.data_ptr<Float16>();
    const Float16* v_data = v.data_ptr<Float16>();
    Float16* qkv_data = qkv_acv.data_ptr<Float16>();

    const int q_ctx = qk.size(1);
    const int kv_ctx = qk.size(2);
    const int d_embed = v.size(1);
    const int d_head = d_embed / n_heads;

    for (int head = 0; head < n_heads; head++) {
        for (int qk_row = ctx_offset; qk_row < q_ctx; qk_row++){
            for (int v_col = 0; v_col < d_head; v_col++) {
                Float32 dot_prod = 0;
                for (int i = 0; i < kv_ctx; i++) {
                    int qk_i = head * q_ctx * kv_ctx + qk_row * kv_ctx + i;
                    int v_i = head * d_head + i * d_embed + v_col;
                    Float32 qkw =fpcvt_to_fp32<Float16>(qk_data[qk_i]);
                    Float32 vw = fpcvt_to_fp32<Float16>(v_data[v_i]);
                    dot_prod += qkw * vw;
                }
                int qkv_data_i = head * d_head + qk_row * d_embed + v_col;
                qkv_data[qkv_data_i] = fpcvt_from_fp32<Float16>(dot_prod);
            }
        }
    }

}

MultiHeadSelfAttn::MultiHeadSelfAttn(int n_head, int d_embed, int max_ctx, bool mask_attn, bool cache_v_ctx)
    : query{Linear(d_embed, d_embed, max_ctx, /*cache_acv=*/mask_attn)},
      key{Linear(d_embed, d_embed, max_ctx, /*cache_acv=*/mask_attn)},
      value{Linear(d_embed, d_embed, max_ctx, /*cache_acv=*/cache_v_ctx, /*transpose_out=*/!cache_v_ctx)},
      qkv_proj{Linear(d_embed, d_embed, max_ctx, /*cache_acv=*/mask_attn)},
      qk_acv_{Tensor({n_head, max_ctx, max_ctx}, kFloat16)},
      qkv_acv_{Tensor({max_ctx, d_embed}, kFloat16)},
      n_heads_{n_head},
      mask_attn_{mask_attn},
      cache_v_ctx_{cache_v_ctx}
{
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp)
{
    GTEN_CHECK_NDIMS_EQUAL(inp.ndims(), 2);
    GTEN_CHECK_DIMSIZE_EQUAL(1, inp.size(1), qkv_acv_.size(1));

    Tensor q = query.forward(inp);
    Tensor k = key.forward(inp);
    Tensor v = value.forward(inp);

    const Tensor qkv = mask_attn_ ? masked_qkv_attn(q, k, v)
                                  : non_masked_qkv_attn(q, k, v);
    const Tensor out = qkv_proj.forward(qkv);
    return out;
}


Tensor MultiHeadSelfAttn::masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    Timer timer{&time_attn_ms_};

    const int n_ctx = q.size(0);
    const int d_embed = q.size(1);
    qk_acv_.resize({n_heads_, n_ctx, n_ctx});
    qkv_acv_.resize({n_ctx, d_embed});

    if (qkv_cached_) {
        qk_attn_matmul(q,k, qk_acv_, n_heads_, /*masked=*/true, /*ctx_offset=*/n_ctx-1);
        qk_softmax(qk_acv_, n_heads_, /*ctx_offset=*/n_ctx-1);
        if (cache_v_ctx_) {
            qkv_attn_matmul_nt(qk_acv_, v, qkv_acv_, n_heads_, /*ctx_offset=*/n_ctx-1);
        } else {
            qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_, /*ctx_offset=*/n_ctx-1);
        }
    }
    else {
        qkv_cached_ = true;

        qk_attn_matmul(q, k, qk_acv_, n_heads_, /*masked=*/true);
        qk_softmax(qk_acv_, n_heads_);
        if (cache_v_ctx_) {
            qkv_attn_matmul_nt(qk_acv_, v, qkv_acv_, n_heads_);
        } else {
            qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_);
        }
    }

    return qkv_acv_;
}


Tensor MultiHeadSelfAttn::non_masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v)
{
    Timer timer{&time_attn_ms_};

    const int n_ctx = q.size(0);
    const int d_embed = q.size(1);

    // SCALED DOT-PRODUCT: (Q @ K) * scale.
    qk_acv_.resize({n_heads_, n_ctx, n_ctx});
    qk_attn_matmul(q, k, qk_acv_, n_heads_, /*masked=*/false);

    // SOFTMAX
    qk_softmax(qk_acv_, n_heads_);

    // ATTENTION: QK @ V
    qkv_acv_.resize({n_ctx, d_embed});
    qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_);

    return qkv_acv_;
}


MultiHeadCrossAttn::MultiHeadCrossAttn(int attn_heads, int d_embed, int max_q_ctx, int kv_ctx)
    : query{Linear(d_embed, d_embed, max_q_ctx, /*cache_acv=*/true)},
      key{Linear(d_embed, d_embed, kv_ctx, /*cache_acv=*/false)},
      value{Linear(d_embed, d_embed, kv_ctx, /*cache_acv=*/false, /*transpose_out=*/true)},
      qkv_proj{Linear(d_embed, d_embed, max_q_ctx, /*cache_acv*/false)},
      qk_acv_{Tensor({attn_heads, max_q_ctx, kv_ctx}, kFloat16)},
      qkv_acv_{Tensor({max_q_ctx, d_embed}, kFloat16)},
      n_heads_{attn_heads}
{
}

Tensor MultiHeadCrossAttn::forward(const Tensor& x, const Tensor& xa)
{
    GTEN_CHECK_NDIMS_EQUAL(x.ndims(), 2);
    GTEN_CHECK_NDIMS_EQUAL(xa.ndims(), 2);

    Tensor q = query.forward(x);
    if (!kv_cached_) {
        k_cache = key.forward(xa);
        v_cache = value.forward(xa);
        kv_cached_ = true;
    }

    const Tensor qkv = qkv_attn(q, k_cache, v_cache);
    const Tensor out = qkv_proj.forward(qkv);
    return out;
}

Tensor MultiHeadCrossAttn::qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v)
{
    const int q_ctx = q.size(0);
    const int kv_ctx = k.size(0);
    qk_acv_.resize({n_heads_, q_ctx, kv_ctx});

    const int d_embed = k.size(1);
    qkv_acv_.resize({q_ctx, d_embed});

    if (qkv_cached_) {
        qk_attn_matmul(q,k, qk_acv_, n_heads_, /*masked=*/false, /*ctx_offset=*/q_ctx-1);
        qk_softmax(qk_acv_, n_heads_, /*ctx_offset=*/q_ctx-1);
        qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_, /*ctx_offset=*/q_ctx-1);
    }
    else {
        qkv_cached_ = true;

        qk_attn_matmul(q, k, qk_acv_, n_heads_, /*masked=*/false);
        qk_softmax(qk_acv_, n_heads_);
        qkv_attn_matmul(qk_acv_, v, qkv_acv_, n_heads_);
    }

    return qkv_acv_;
}

ResidualAttnBlock::ResidualAttnBlock(int n_attn_heads, int d_embed, int d_mlp, int max_ctx, bool mask_attn)
    : attn_ln{LayerNorm(max_ctx, d_embed)},
      attn{MultiHeadSelfAttn(n_attn_heads, d_embed, max_ctx, mask_attn, /*cache_v_ctx=*/mask_attn)},
      inp_res{Residual(max_ctx, d_embed)},
      mlp_ln{LayerNorm(max_ctx, d_embed)},
      mlp_fc{Linear(d_embed, d_mlp, max_ctx, /*cache_acv=*/true)},
      gelu{GELU(max_ctx, d_mlp, /*cache_ctx_acv=*/true)},
      mlp_proj{Linear(d_mlp, d_embed, max_ctx, /*cache_acv=*/true)},
      attn_res{Residual(max_ctx, d_embed)}
{
}

Tensor ResidualAttnBlock::forward(const Tensor &inp)
{
    Tensor attn_out = inp_res.forward(inp, attn.forward(attn_ln.forward(inp)));
    Tensor out = attn_res.forward(attn_out,
        mlp_proj.forward(gelu.forward(mlp_fc.forward(mlp_ln.forward(attn_out)))));
    return out;
}

ResidualCrossAttnBlock::ResidualCrossAttnBlock(int attn_heads, int d_embed, int d_mlp, int max_q_ctx, int kv_ctx)
    : attn_ln{LayerNorm(max_q_ctx, d_embed)},
      attn{MultiHeadSelfAttn(attn_heads, d_embed, max_q_ctx, /*mask_attn=*/true, /*cache_v_ctx=*/false)},
      attn_res{Residual(max_q_ctx, d_embed)},
      cross_attn_ln{LayerNorm(max_q_ctx, d_embed)},
      cross_attn{MultiHeadCrossAttn(attn_heads, d_embed, max_q_ctx, kv_ctx)},
      cross_attn_res{Residual(max_q_ctx, d_embed)},
      mlp_ln{LayerNorm(max_q_ctx, d_embed)},
      mlp_fc{Linear(d_embed, d_mlp, max_q_ctx, /*cache_acv=*/true)},
      gelu{GELU(max_q_ctx, d_mlp, /*cache_ctx_acv=*/true)},
      mlp_proj{Linear(d_mlp, d_embed, max_q_ctx, /*cache_acv=*/true)},
      mlp_res{Residual(max_q_ctx, d_embed)}
{
}

Tensor ResidualCrossAttnBlock::forward(const Tensor& x, const Tensor& xa)
{
    Tensor out = attn_res.forward(x, attn.forward(attn_ln.forward(x)));
    out = cross_attn_res.forward(out, cross_attn.forward(cross_attn_ln.forward(out), xa));
    out = mlp_res.forward(out, mlp_proj.forward(gelu.forward(mlp_fc.forward(mlp_ln.forward(out)))));
    return out;
}

Conv1d::Conv1d(int n_filters, int in_channels, int in_frames, int strides)
    : weight{Tensor({n_filters, in_channels*3}, kFloat16)},  // 3 is the filtersize.
      bias{Tensor({n_filters}, kFloat16)},
      strides_{strides}
{
    const int out_frames = in_frames / strides_;
    if (strides_ == 1) {
        acv_ = Tensor({n_filters, out_frames}, kFloat16);
    } else {
        acv_ = Tensor({out_frames, n_filters}, kFloat16);
    }

    const int out_channels = in_channels*filtersize_;
    im2col_out_ = Tensor({out_frames, out_channels}, kFloat16);
}


Tensor Conv1d::forward(const Tensor& inp)
{
    Timer timer{&exec_time_ms_};
    // Pad and im2col [out_frames, inp_channels*filter_size].
    // Set im2col out data to zeros so that padded areas are set to zero by default.
    void* im2col_out_data = im2col_out_.data_ptr<void>();
    std::memset(im2col_out_data, 0, im2col_out_.nbytes());

    Tensor im2col = (strides_ == 1) ? im2col_stride1(inp) : im2col_stride2(inp);

    // matmul.
    const Float16* im2col_data = im2col.data_ptr<Float16>();
    const Float16* weight_data = weight.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* acv_data = acv_.data_ptr<Float16>();

    // weight: [(n_filters, in_channels*filter_size)]
    // bias: n_filters
    const int n_filters = weight.size(0);     // 384
    const int filter_dimsize = weight.size(1);// 384*3
    const int out_frames = im2col.size(0);    // 1500

    for (int i = 0; i < n_filters; i++) {
        for (int j = 0; j < out_frames; j++) {
            Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();
            for (int k = 0; k < filter_dimsize; k += 8) {
                Vec_f32x8 w = vec_f32x8_load(weight_data + i*filter_dimsize + k);
                Vec_f32x8 x = vec_f32x8_load(im2col_data + j*filter_dimsize + k);
                dot_prod_accum = vec_f32x8_fma(w, x, dot_prod_accum);
            }
            Float32 dot_prod = vec_f32x8_sum(dot_prod_accum) + fpcvt_to_fp32(bias_data[i]);

            const int acv_idx = (strides_ == 1) ? i*out_frames + j : j*n_filters + i;
            acv_data[acv_idx] = fpcvt_from_fp32<Float16>(dot_prod);
        }
    }

    return acv_;
}

// Assumes out_chs > 2
Tensor Conv1d::im2col_stride1(const Tensor& inp)
{
    const int inp_chs = inp.size(0);
    const int inp_frames = inp.size(1);
    const int out_chs = inp_chs*filtersize_;
    const int out_frames = inp_frames;

    const Float32* inp_data = inp.data_ptr<Float32>();
    Float16* out_data = im2col_out_.data_ptr<Float16>();

    // filter size accounting for padded 0.
    const int pad_filtersize = filtersize_ - 1;
    // Rearrange the first window accounting for zero-padding.
    for (int i = 0; i < inp_chs; i++) {
        for (int j = 0; j < pad_filtersize; j++) {
            const int inp_idx = i*inp_frames + j;
            const int out_idx = i*filtersize_ + j + 1;
            out_data[out_idx] = fpcvt_from_fp32<Float16>(inp_data[inp_idx]);
        }
    }
    
    // Number of output frames left after subtracting two padded rows.
    const int middle_frames = out_frames - 2;
    // offset the first padded row that we just computed above.
    const int out_row_offset = out_chs;
    // Rearrange the middle windows which are not padded.
    for (int i = 0; i < middle_frames; i++) {
        for (int j = 0; j < inp_chs; j++) {
            for (int k = 0; k < filtersize_; k++) {
                const int inp_idx = i + j*inp_frames + k;
                const int out_idx = out_row_offset + i*out_chs + j*filtersize_ + k;
                out_data[out_idx] = fpcvt_from_fp32<Float16>(inp_data[inp_idx]); 
            }   
        }
    }
    
    // Offset to the start of the final filter position on the input.
    const int inp_row_offset = inp_frames + 2 - filtersize_ - 1;
    // Offset to the start of the last row of the output.
    const int out_offset = (out_frames - 1) * out_chs;
    // Rearrange the last window accounting for zero-padding.
    for (int i = 0; i < inp_chs; i++){
        for (int j = 0; j < pad_filtersize; j++){
            const int inp_idx = inp_row_offset + i*inp_frames + j;
            const int out_idx = out_offset + i*filtersize_ + j;
            out_data[out_idx] = fpcvt_from_fp32<Float16>(inp_data[inp_idx]);
        }   
    }
    
    return im2col_out_;
}

// Assume out_chs > 2.
Tensor Conv1d::im2col_stride2(const Tensor& inp)
{
    const int inp_chs = inp.size(0);
    const int inp_frames = inp.size(1);
    const int out_chs = inp_chs * filtersize_;
    const int out_frames = inp_frames / 2;

    const Float16* inp_data = inp.data_ptr<Float16>();
    Float16* out_data = im2col_out_.data_ptr<Float16>();
    
    // filter size accounting for padded 0.
    const int pad_filtersize = filtersize_ - 1;
    // filter size accounting for padded 0.
    for (int i = 0; i < inp_chs; i++) {
        for (int j = 0; j < pad_filtersize; j++) {
            const int inp_idx = i*inp_frames + j;
            const int out_idx = i*filtersize_ + j + 1;
            out_data[out_idx] = inp_data[inp_idx];
        }
    }

    // Number of output frames left after subtracting the padded row.
    const int middle_frames = out_frames - 1;
    // offset the first padded row that we just computed above.
    const int out_row_offset = out_chs;
    // Rearrange the middle windows which are not padded. The last window is not handled
    // because it is skipped when strides=2.
    for (int i = 0; i < middle_frames; i++) {
        for (int j = 0; j < inp_chs; j++) {
            for (int k = 0; k < filtersize_; k++) {
                const int inp_idx = i+i+1 + j*inp_frames + k;
                const int out_idx = out_row_offset + i*out_chs + j*filtersize_ + k;
                out_data[out_idx] = inp_data[inp_idx];
            }   
        }
    }
    
    return im2col_out_;
}

} // namespace gten
