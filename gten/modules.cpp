#include <cmath>
#include <iostream>
#include <cstring>

#include "modules.h"
#include "ops.h"


namespace gten {

Embedding::Embedding(int n_vocab, int n_embd, int max_ctx, Dtype dtype, int qblock_size)
    : weight{Tensor({n_vocab, n_embd}, dtype, qblock_size)},
      emb_acv{Tensor({max_ctx, n_embd}, dtype, qblock_size)},
      proj_acv_{Tensor({n_vocab}, kFloat32)},
      max_ctx_(max_ctx)
{
}

Tensor Embedding::forward(const Tensor& tokens) {
    Timer timer{&exec_time_emb_ms_};
    
    const int n_embd = weight.size(1);
    emb_acv.resize({tokens.numel(), n_embd});

    if (emb_acv_cached_) {
        ops::token_embed(weight, tokens, emb_acv, /*last_token_only=*/true);
    } else {
        emb_acv_cached_ = true;

        ops::token_embed(weight, tokens, emb_acv);
    }

    return emb_acv;
}


Tensor Embedding::forward_proj(const Tensor &inp) {
    Timer timer(&exec_time_proj_ms_);

    ops::emb_matmul(inp, weight, proj_acv_);

    return proj_acv_;
}

PosEmbedding::PosEmbedding(int max_ctx, int n_embd, Dtype dtype, int qblock_size)
    : weight{Tensor({max_ctx, n_embd}, dtype, qblock_size)},
    acv{Tensor({max_ctx, n_embd}, dtype, qblock_size)}, max_ctx_(max_ctx)
{
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
        Qint8* out_data = acv.data_ptr<Qint8>();

        const int n_embd = weight.size(1);
        acv.resize({n_ctx, n_embd});

        const Qparams& w_qparams = weight.qparams();
        Qparams& out_qparams = acv.qparams();
        const int block_size = w_qparams.block_size();
        const int blocks_per_row = w_qparams.blocks_per_row();

        const int ctx_start = acv_cached_ ? n_ctx - 1 : 0;
        acv_cached_ = true;

        for (int i = ctx_start; i < n_ctx; i++) {
            for (int j = 0; j < blocks_per_row; j++) {
                const Qint8* src = weight_data + i * n_embd + j * block_size;
                Qint8* dest = out_data + i * n_embd + j * block_size;
                std::memcpy(dest, src, block_size * sizeof(Qint8));

                Float16* out_row_deltas = out_qparams.row_deltas(i);
                out_row_deltas[j] = w_qparams.row_deltas(i)[j];
            }
        }
    } else {
        const void* src_data = weight.data_ptr<void>();
        const int n_embd = weight.size(1);

        // Shares the data with the weight tensor.
        acv = Tensor{src_data, {n_ctx, n_embd}, weight.dtype()};
    }

    return acv;
}

LayerNorm::LayerNorm(int max_ctx, int n_embd, Dtype dtype, int qblock_size)
    : weight{Tensor({n_embd}, kFloat16)},
      bias{Tensor({n_embd}, kFloat16)},
      acv{Tensor({max_ctx, n_embd}, dtype, qblock_size)},
      max_ctx_{max_ctx}
{
}


Tensor LayerNorm::forward(const Tensor &inp) {
    Timer timer(&exec_time_ms_);

    const int n_ctx = inp.size(0);
    const int n_embd = inp.size(1);
    acv.resize({n_ctx, n_embd});

    
    if (acv_cached_) {
        ops::layer_norm(inp, weight, bias, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;

        ops::layer_norm(inp, weight, bias, acv);
    }

    return acv;
}


GELU::GELU(int max_ctx, int n_out, Dtype dtype, int qblock_size)
    : acv{Tensor({max_ctx, n_out}, dtype, qblock_size)}, max_ctx_{max_ctx}
{
}

Tensor GELU::forward(const Tensor& inp)
{
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp.size(0);
    const int n_out = acv.size(1);

    acv.resize({n_ctx, n_out});

    if (acv_cached_) {
        ops::gelu(inp, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;
        ops::gelu(inp, acv);
    }
    
    return acv;
}


Residual::Residual(int max_ctx, int n_out, Dtype dtype, int qblock_size)
    : acv{Tensor({max_ctx, n_out}, dtype, qblock_size)}, max_ctx_{max_ctx}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp0.size(0);
    const int n_embd = inp0.size(1);

    acv.resize({n_ctx, n_embd});

    if (acv_cached_) {
        ops::add(inp0, inp1, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;
        ops::add(inp0, inp1, acv);
    }

    return acv;
}

Linear::Linear(int n_in, int n_out, int max_ctx, Dtype dtype, int qblock_size)
    : weight{Tensor({n_out, n_in}, dtype, qblock_size)},
      bias{Tensor({n_out}, kFloat16)},
      acv{Tensor({max_ctx, n_out}, dtype, qblock_size)},
      max_ctx_{max_ctx}
{
}

Tensor Linear::forward(const Tensor &inp) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp.size(0);
    const int n_out = weight.size(0);
    
    acv.resize({n_ctx, n_out});

    if (acv_cached_) {
        ops::affine_proj_2d(inp, weight, bias, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;
        ops::affine_proj_2d(inp, weight, bias, acv);
    }

    return acv;
}


/// TODO: Construct a transposed linear module?
Tensor Linear::forward_transposed(const Tensor &inp) {
    Timer timer{&exec_time_ms_};

    const int n_ctx = inp.size(0);
    const int n_out = weight.size(0);
    
    acv.resize({n_out, n_ctx});
    /// TODO: Allow strides-lock on tensors.
    acv.set_strides({max_ctx_, 1});

    if (acv_cached_) {
        ops::affine_proj_2d_transposed(inp, weight, bias, acv, /*last_ctx_only=*/true);
    } else {
        acv_cached_ = true;
        ops::affine_proj_2d_transposed(inp, weight, bias, acv);
    }

    return acv;
}

MultiHeadSelfAttn::MultiHeadSelfAttn(int n_head, int n_embd, int max_ctx, Dtype dtype, int qblock_size)
    : query{Linear(n_embd, n_embd, max_ctx, dtype, qblock_size)},
      key{Linear(n_embd, n_embd, max_ctx, dtype, qblock_size)},
      value{Linear(n_embd, n_embd, max_ctx, dtype, qblock_size)},
      qkv_proj{Linear(n_embd, n_embd, max_ctx, dtype, qblock_size)},
      qk_acv{Tensor({n_head, max_ctx, max_ctx}, dtype, qblock_size, /*zero_mem=*/true)},
      qkv_acv{Tensor({max_ctx, n_embd}, dtype, qblock_size)},
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
    std::memset(qk_acv.data_ptr<void>(), 0, qk_acv.nbytes());
}

Tensor MultiHeadSelfAttn::masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v) {
    Timer timer{&time_attn_ms_};

    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);

    qk_acv.resize({n_head_, n_ctx, n_ctx});
    qkv_acv.resize({n_ctx, n_embd});

    if (qkv_cached_) {
        ops::qkv_attn(q, k, v, qk_acv, qkv_acv, max_ctx_, /*last_ctx_only=*/true);
    } else {
        qkv_cached_ = true;

        ops::qkv_attn(q, k, v, qk_acv, qkv_acv, max_ctx_);
    }


    return qkv_acv;
}

ResidualAttnBlock::ResidualAttnBlock(int n_attn_heads, int n_embd, int d_mlp, int max_ctx, Dtype dtype, int qblock_size)
    : attn_ln{LayerNorm(max_ctx, n_embd, dtype, qblock_size)},
      attn{MultiHeadSelfAttn(n_attn_heads, n_embd, max_ctx, dtype, qblock_size)},
      inp_res{Residual(max_ctx, n_embd, dtype, qblock_size)},
      mlp_ln{LayerNorm(max_ctx, n_embd, dtype, qblock_size)},
      mlp_fc{Linear(n_embd, d_mlp, max_ctx, dtype, qblock_size)},
      gelu{GELU(max_ctx, d_mlp, dtype, qblock_size)},
      mlp_proj{Linear(d_mlp, n_embd, max_ctx, dtype, qblock_size)},
      attn_res{Residual(max_ctx, n_embd, dtype, qblock_size)}
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
