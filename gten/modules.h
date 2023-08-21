#pragma once

#include <chrono>
#include <iostream>

#include "tensor.h"

namespace gten {

/// Provides an embedding table lookup for tokens.
class Embedding {
public:
    Embedding(int n_vocab, int d_embed, int max_ctx);

    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward(const Tensor& tokens);

    /// Performs matrix mul between the input and the weights tensor. The input tensor
    /// is expected to have shape (n_ctx, n_embed) and the output has shape (1, n_vocab)
    /// which represents the prob dist of the next predicted token given the context.
    Tensor forward_proj(const Tensor& inp);

    int64_t emb_time() const { return exec_time_emb_ms_; }
    int64_t emb_proj_time() const { return exec_time_proj_ms_; }
    void reset_acv_cache() { emb_acv_cached_=false; }

public:
    Tensor weight;

private:
    Tensor emb_acv_;
    bool emb_acv_cached_{false};
    Tensor proj_acv_;
    int max_ctx_;
    int64_t exec_time_emb_ms_{0};
    int64_t exec_time_proj_ms_{0};

private:
    Tensor forward_impl(const Tensor& inp);
    Tensor forward_proj_impl(const Tensor& inp);
};

/// Provides a positional embedding table lookup for tokens.
class PosEmbedding {
public:
    PosEmbedding(int max_ctx, int d_embed);

    /// Return the positional embeddings of the given number of tokens. The number of
    /// tokens must not exceed max_ctx.
    Tensor forward(int n_ctx);

    int64_t time() const { return exec_time_ms_; }

public:
    Tensor weight;

private:
    int max_ctx_{0};
    int64_t exec_time_ms_{0};

private:
    Tensor forward_impl(int n_ctx);
};

/// Applies affine normalization on the input.
class LayerNorm {
public:
    LayerNorm(int max_ctx, int d_embed);

    /// Normalize the input. Both input and output are of shape (n_ctx, n_embed). n_ctx
    /// must not exceed max_ctx.
    Tensor forward(const Tensor& inp);

    int64_t time() const noexcept { return exec_time_ms_; }
    void reset_acv_cache() { acv_cached_=false; }

public:
    Tensor weight;
    Tensor bias;

private:
    Tensor acv_;
    int max_ctx_;
    bool acv_cached_{false};
    float eps_{1e-05f};
    int64_t exec_time_ms_{0};

private:
    Tensor forward_impl(const Tensor &inp);
};


class GELU {
public:
    GELU(int max_ctx, int d_out, bool cache_acv);
    Tensor forward(const Tensor& inp);
    int64_t time() const noexcept { return exec_time_ms_; }
    void reset_acv_cache() { acv_cached_=false; }

private:
    Tensor acv_;
    bool acv_cached_{false};
    bool cache_acv_{false};
    int64_t exec_time_ms_{0};

private:
    Tensor forward_impl(const Tensor& inp);
};

class Residual {
public:
    Residual(int max_ctx, int d_out);
    Tensor forward(const Tensor& inp0, const Tensor& inp1);
    int64_t time() const noexcept { return exec_time_ms_; }
    void reset_acv_cache() { acv_cached_=false; }

private:
    Tensor acv_;
    bool acv_cached_{false};
    int64_t exec_time_ms_{0};

private:
    Tensor forward_impl(const Tensor& inp0, const Tensor& inp1);
};


/// Applies an affine linear transformation on the input.
class Linear {
public:
    Linear(int d_in, int d_out, int max_ctx, bool cache_acv, bool transpose_out = false);
    Tensor forward(const Tensor& inp);
    int64_t time() const noexcept { return exec_time_ms_; }
    void reset_acv_cache() { acv_cached_=false; }

public:
    Tensor weight;
    Tensor bias;

private:
    Tensor acv_;
    bool cache_acv_;
    bool transpose_out_{false};
    bool acv_cached_{false};
    int64_t exec_time_ms_{0};

private:
    Tensor forward_impl(const Tensor& inp);
};

class MultiHeadSelfAttn {
public:
    MultiHeadSelfAttn(int n_heads, int n_embed, int max_ctx, bool mask_attn, bool cache_v_ctx);
    Tensor forward(const Tensor& inp);
    int64_t time_linear() const noexcept { return query.time() + key.time() + value.time() + qkv_proj.time(); }
    int64_t time_attn() const noexcept { return time_attn_ms_; }
    void reset_acv_cache() noexcept {
        query.reset_acv_cache();
        key.reset_acv_cache();
        value.reset_acv_cache();
        qkv_proj.reset_acv_cache();
        qkv_cached_=false;
    }

public:
    Linear query;
    Linear key;
    Linear value;
    Linear qkv_proj;

private:
    Tensor qk_acv_;
    Tensor qkv_acv_;
    int32_t n_heads_;
    bool mask_attn_;
    bool cache_v_ctx_;
    bool qkv_cached_{false};
    int64_t time_attn_ms_{0};

private:
    Tensor masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v);
    Tensor non_masked_qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v);
};


class MultiHeadCrossAttn {
public:
    MultiHeadCrossAttn(int attn_heads, int d_embed, int max_q_ctx, int kv_ct);
    Tensor forward(const Tensor& x, const Tensor& xa);
    int64_t time_linear() const noexcept { return query.time() + key.time() + value.time() + qkv_proj.time(); }
    int64_t time_attn() const noexcept { return time_attn_ms_; }
    void reset_acv_cache() { 
        query.reset_acv_cache();
        key.reset_acv_cache();
        value.reset_acv_cache();
        qkv_proj.reset_acv_cache();
        kv_cached_ = false;
        qkv_cached_ = false;
    }

public:
    Linear query;
    Linear key;
    Linear value;
    Linear qkv_proj;

private:
    Tensor qk_acv_;
    Tensor qkv_acv_;
    int32_t n_heads_;
    bool kv_cached_{false};
    bool qkv_cached_{false};
    int64_t time_attn_ms_{0};

    Tensor k_cache;
    Tensor v_cache;

private:
    Tensor qkv_attn(const Tensor &q, const Tensor &k, const Tensor &v);
};


class ResidualAttnBlock {
public:
    ResidualAttnBlock(int n_attn_heads, int d_embed, int d_mlp, int max_ctx, bool mask_attn);
    Tensor forward(const Tensor& inp);
    int64_t time_linear() const
    { return attn.time_linear() + mlp_fc.time() + mlp_proj.time(); }
    int64_t time_proj() const { return mlp_fc.time() + mlp_proj.time(); }
    int64_t time_attn_lin() const { return attn.time_linear(); }
    int64_t time_attn() const { return attn.time_attn(); }
    int64_t time_ln() const { return attn_ln.time() + mlp_ln.time(); }
    int64_t time_gelu() const { return gelu.time(); }
    int64_t time_res() const { return inp_res.time() + attn_res.time(); }
    void reset_acv_cache() { attn.reset_acv_cache(); attn_ln.reset_acv_cache(); mlp_fc.reset_acv_cache(); mlp_proj.reset_acv_cache();
                             mlp_ln.reset_acv_cache(); gelu.reset_acv_cache(); inp_res.reset_acv_cache(); attn_res.reset_acv_cache();}

public:
    LayerNorm attn_ln;
    MultiHeadSelfAttn attn;
    Residual inp_res;
    LayerNorm mlp_ln;
    Linear mlp_fc;
    GELU gelu;
    Linear mlp_proj;
    Residual attn_res;
};


class ResidualCrossAttnBlock {
public:
    ResidualCrossAttnBlock(int attn_heads, int d_embed, int d_mlp, int max_q_ctx, int kv_ctx);
    Tensor forward(const Tensor& x, const Tensor& xa);
    void reset_acv_cache() {
        attn.reset_acv_cache();
        attn_ln.reset_acv_cache();
        mlp_fc.reset_acv_cache();
        mlp_proj.reset_acv_cache();
        mlp_ln.reset_acv_cache();
        gelu.reset_acv_cache();
        cross_attn_ln.reset_acv_cache();
        cross_attn.reset_acv_cache();
        cross_attn_res.reset_acv_cache();
        attn_res.reset_acv_cache();
        mlp_res.reset_acv_cache();
    }

public:
    LayerNorm attn_ln;
    MultiHeadSelfAttn attn;
    Residual attn_res;
    LayerNorm cross_attn_ln;
    MultiHeadCrossAttn cross_attn;
    Residual cross_attn_res;
    LayerNorm mlp_ln;
    Linear mlp_fc;
    GELU gelu;
    Linear mlp_proj;
    Residual mlp_res;
};


/// @brief Implements 1-dimensional convolution on an input with padding=1
/// and filtersize=3 and strides=1 or 2.
class Conv1d {
public:
    /// @brief Construct a convolution layer.
    /// @param weight A tensor of shape (n_filters, in_channels*filter_size).
    /// @param bias A tensor of shape n_filters.
    /// @param acv A tensor of shape (n_filters, n_out_frames).
    /// @param strides Either 1 or 2.
    Conv1d(int n_filters, int in_channels, int in_frames, int strides);

    /// @brief Compute a convolution on the input.
    /// @param inp A tensor of shape (in_channels, n_frames).
    /// @return A tensor of shape (n_filters, n_frames) if strides=1 or
    /// (n_filters, n_frames/2) if strides=2.
    Tensor forward(const Tensor& inp);
    int64_t exec_time() const { return exec_time_ms_; };

public:
    Tensor weight;
    Tensor bias;

private:
    Tensor acv_;
    Tensor im2col_out_;
    int strides_;
    int filtersize_{3};
    int64_t exec_time_ms_{0};

private:
    // Pads the input Tensor with padding=1 and rearranges its shape from
    // (inp_channels, inp_frames) to (inp_frames, inp_channels*filter_size) so
    // that a convolution can be performed on the transformed input as a matrix
    // multiplication.
    Tensor im2col_stride1(const Tensor& inp);

    // Pads the input Tensor with padding=2 and rearranges its shape from
    // (inp_channels, inp_frames) to (inp_frames/2, inp_channels*filter_size) so
    // that a convolution can be performed on the transformed input as a matrix
    // multiplication.
    Tensor im2col_stride2(const Tensor& inp);
};


class Timer {
public:
    Timer(int64_t* time_tracker)
        : time_tracker_{time_tracker}, start_time_{std::chrono::high_resolution_clock::now()}
    { 
    }
    ~Timer() { stop(); }

    void stop() {
        if (stopped_)
            return;
        auto end_time = std::chrono::high_resolution_clock::now();
        int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_).time_since_epoch().count();
        int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
        int64_t duration = end - start;
        *time_tracker_ += duration;
        stopped_ = true;
    }
private:
    int64_t* time_tracker_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    bool stopped_ = false;
};

} // namespace gten
