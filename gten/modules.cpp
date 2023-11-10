#include <cmath>
#include <iostream>
#include <cstring>

#include "utils.h"
#include "modules.h"
#include "ops.h"


namespace gten {

Embedding::Embedding(int n_vocab, int n_embd, int max_ctx, TensorDtype wdtype)
    : weight{Tensor({n_vocab, n_embd}, wdtype)},
      emb_acv_{Tensor({max_ctx, n_embd}, kFloat16)},
      proj_acv_{Tensor({n_vocab}, kFloat32)},
      max_ctx_(max_ctx)
{
}

Tensor Embedding::forward(const Tensor& tokens) {
    Timer timer{&exec_time_emb_ms_};
    
    return forward_impl(tokens);
}

Tensor Embedding::forward_impl(const Tensor& inp)
{
    const int n_embd = weight.size(1);
    emb_acv_.resize({inp.numel(), n_embd});

    if (emb_acv_cached_) {
        const int ctx_offs = inp.numel() - 1;
        ops::tensor_row_index(weight, inp, emb_acv_, ctx_offs);
    } else {
        emb_acv_cached_ = true;

        ops::tensor_row_index(weight, inp, emb_acv_);
    }

    return emb_acv_;    
}


Tensor Embedding::forward_proj(const Tensor &inp) {
    Timer timer(&exec_time_proj_ms_);

    return forward_proj_impl(inp);
}


Tensor Embedding::forward_proj_impl(const Tensor& inp) {
    // (n_ctx, n_embed) x (n_vocab, n_embed)
    ops::emb_matmul(inp, weight, proj_acv_);

    return proj_acv_;  
}

PosEmbedding::PosEmbedding(int max_ctx, int n_embd, TensorDtype wdtype)
    : weight{Tensor({max_ctx, n_embd}, wdtype)}, max_ctx_(max_ctx)
{
    if (wdtype == kQint8) {
        acv_ = Tensor({max_ctx, n_embd}, kFloat16);
    }
}

Tensor PosEmbedding::forward(int n_ctx) {
    Timer timer{&exec_time_ms_};
    
    return forward_impl(n_ctx);
}

Tensor PosEmbedding::forward_impl(int n_ctx)
{
    if (weight.dtype() == kQint8) {
        /// TODO: CACHE PREV VALUES. impl counter;
        const Qint8* weight_data = weight.data_ptr<Qint8>();
        Float16* out_data = acv_.data_ptr<Float16>();
        const int n_embd = weight.size(1);
        const float s = weight.scale();
        const int z = weight.zerop();

        acv_.resize({n_ctx, n_embd});

        for (int i = 0; i < n_ctx; i++) {
            for (int j = 0; j < n_embd; j++) {
                const float w = ops::deq(weight_data[i * n_embd + j], s, z);
                out_data[i * n_embd + j] = fpcvt_stoh(w);
            }
        }
    } else {
        const Float16* weight_data = weight.data_ptr<Float16>();

        void* src_ptr = (void*)weight_data;
        const int n_embd = weight.size(1);

        // Shares the data with the weight tensor.
        acv_ = Tensor{src_ptr, {n_ctx, n_embd}, weight.dtype()};
    }
    
    return acv_;
}

LayerNorm::LayerNorm(int max_ctx, int n_embd, TensorDtype wdtype)
    : weight{Tensor({n_embd}, wdtype)},
      bias{Tensor({n_embd}, wdtype)},
      acv_{Tensor({max_ctx, n_embd}, kFloat16)},
      max_ctx_{max_ctx}
{
}


Tensor LayerNorm::forward(const Tensor &inp) {
    Timer timer(&exec_time_ms_);

    return forward_impl(inp);
}


Tensor LayerNorm::forward_impl(const Tensor &inp) {
    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);

    acv_.resize({n_ctx, n_embd});

    
    if (acv_cached_) {
        const int ctx_offs = inp.size(0) - 1;
        ops::layer_norm(inp, weight, bias, acv_, ctx_offs);
    } else {
        acv_cached_ = true;

        ops::layer_norm(inp, weight, bias, acv_);
    }

    return acv_;
}


GELU::GELU(int max_ctx, int n_out)
    : acv_{Tensor({max_ctx, n_out}, kFloat16)}, max_ctx_{max_ctx}
{
}

Tensor GELU::forward(const Tensor& inp)
{
    Timer timer{&exec_time_ms_};

    return forward_impl(inp);
}

Tensor GELU::forward_impl(const Tensor& inp) {
    // TODO: Replace with lookup table.
    const int n_ctx = inp.size(0);
    const int n_out = acv_.size(1);

    acv_.resize({n_ctx, n_out});

    if (acv_cached_) {
        const int ctx_offs = n_ctx - 1;
        ops::gelu(inp, acv_, ctx_offs);
    } else {
        acv_cached_ = true;
        ops::gelu(inp, acv_);
    }
    
    return acv_;
}

Residual::Residual(int max_ctx, int n_out)
    : acv_{Tensor({max_ctx, n_out}, kFloat16)}, max_ctx_{max_ctx}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1) {
    Timer timer{&exec_time_ms_};

    return forward_impl(inp0, inp1);
}

Tensor Residual::forward_impl(const Tensor& inp0, const Tensor& inp1)
{
    const int n_ctx = inp0.size(0);
    const int n_embd = inp0.size(1);

    acv_.resize({n_ctx, n_embd});

    if (acv_cached_) {
        const int ctx_offs = n_ctx - 1;
        ops::add(inp0, inp1, acv_, ctx_offs);
    } else {
        acv_cached_ = true;
        ops::add(inp0, inp1, acv_);
    }

    return acv_;
}

Linear::Linear(int n_in, int n_out, int max_ctx, TensorDtype wdtype)
    : weight{Tensor({n_out, n_in}, wdtype)},
      bias{Tensor({n_out}, wdtype)},
      acv_{Tensor({max_ctx, n_out}, kFloat16)},
      max_ctx_{max_ctx}
{
}

Tensor Linear::forward(const Tensor &inp) {
    Timer timer{&exec_time_ms_};

    return forward_impl(inp);
}

Tensor Linear::forward_impl(const Tensor& inp)
{
    const int n_ctx = inp.size(0);
    const int n_out = weight.size(0);
    
    acv_.resize({n_ctx, n_out});

    if (acv_cached_) {
        const int ctx_offs = n_ctx - 1;
        ops::affine_proj_2d(inp, weight, bias, acv_, ctx_offs);
    } else {
        acv_cached_ = true;
        ops::affine_proj_2d(inp, weight, bias, acv_);
    }

    return acv_;
}

Tensor Linear::forward_transposed(const Tensor &inp) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp.size(0);
    const int n_out = weight.size(0);
    
    acv_.resize({n_out, n_ctx});
    /// TODO: Allow strides-lock on tensors.
    acv_.set_strides({max_ctx_, 1});

    if (acv_cached_) {
        const int ctx_offs = n_ctx - 1;
        ops::affine_proj_2d_transposed(inp, weight, bias, acv_, ctx_offs);
    } else {
        acv_cached_ = true;
        ops::affine_proj_2d_transposed(inp, weight, bias, acv_);
    }

    return acv_;
}

MultiHeadSelfAttn::MultiHeadSelfAttn(int n_head, int n_embd, int max_ctx, TensorDtype wdtype)
    : query{Linear(n_embd, n_embd, max_ctx, wdtype)},
      key{Linear(n_embd, n_embd, max_ctx, wdtype)},
      value{Linear(n_embd, n_embd, max_ctx, wdtype)},
      qkv_proj{Linear(n_embd, n_embd, max_ctx, wdtype)},
      qk_acv_{Tensor({n_head, max_ctx, max_ctx}, kFloat16, /*zero_mem=*/true)},
      qkv_acv_{Tensor({max_ctx, n_embd}, kFloat16)},
      n_head_{n_head}, max_ctx_{max_ctx}
{
}

Tensor MultiHeadSelfAttn::forward(const Tensor &inp) {
    Tensor q = query.forward(inp);
    Tensor k = key.forward(inp);
    Tensor v = value.forward_transposed(inp);

    const Tensor qkv = masked_qkv_attn(q, k, v);
    const Tensor out = qkv_proj.forward(qkv);
    return out;
}

void MultiHeadSelfAttn::reset_acv_cache()
{
    query.reset_acv_cache();
    key.reset_acv_cache();
    value.reset_acv_cache();
    qkv_proj.reset_acv_cache();
    qkv_cached_=false;

    // Important because it is allocated with 'zero_mem'=true.
    std::memset(qk_acv_.data_ptr<void>(), 0, qk_acv_.nbytes());
}

Tensor MultiHeadSelfAttn::masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v) {
    Timer timer{&time_attn_ms_};

    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);

    qk_acv_.resize({n_head_, n_ctx, n_ctx});
    qkv_acv_.resize({n_ctx, n_embd});

    if (qkv_cached_) {
        const int ctx_offs = n_ctx - 1;
        ops::qkv_attn(q, k, v, qk_acv_, qkv_acv_, max_ctx_, ctx_offs);
    } else {
        qkv_cached_ = true;
        ops::qkv_attn(q, k, v, qk_acv_, qkv_acv_, max_ctx_);
    }


    return qkv_acv_;
}

ResidualAttnBlock::ResidualAttnBlock(int n_attn_heads, int n_embd, int d_mlp, int max_ctx, TensorDtype wdtype)
    : attn_ln{LayerNorm(max_ctx, n_embd, wdtype)},
      attn{MultiHeadSelfAttn(n_attn_heads, n_embd, max_ctx, wdtype)},
      inp_res{Residual(max_ctx, n_embd)},
      mlp_ln{LayerNorm(max_ctx, n_embd, wdtype)},
      mlp_fc{Linear(n_embd, d_mlp, max_ctx, wdtype)},
      gelu{GELU(max_ctx, d_mlp)},
      mlp_proj{Linear(d_mlp, n_embd, max_ctx, wdtype)},
      attn_res{Residual(max_ctx, n_embd)}
{
}

Tensor ResidualAttnBlock::forward(const Tensor &inp)
{
    Tensor attn_out = inp_res.forward(inp, attn.forward(attn_ln.forward(inp)));
    Tensor out = attn_res.forward(attn_out,
        mlp_proj.forward(gelu.forward(mlp_fc.forward(mlp_ln.forward(attn_out)))));
    return out;
}

} // namespace gten
