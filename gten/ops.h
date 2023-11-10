#include "tensor.h"
#include "utils.h"

#include <cstring>
#include <thread>


#define GTEN_ASSERT_DTYPE(t, exp_dtype) \
    GTEN_ASSERT(t.dtype() == exp_dtype, \
        "Expected dtype=%s but got dtype=%s.", \
        dtype_str(exp_dtype), dtype_str(t.dtype()))

#define GTEN_ASSERT_NDIMS(t, exp_ndims) \
    GTEN_ASSERT(t.ndims() == exp_ndims, \
        "Expected ndims=%d but got ndims=%d", \
        exp_ndims, t.ndims())

#define GTEN_ASSERT_DIMSIZE(t, dim_idx, exp_dimsize) \
    GTEN_ASSERT(t.size(dim_idx) == exp_dimsize, \
        "Expected dim-%d to have size=%d but got size=%d", \
        dim_idx, exp_dimsize, t.size(dim_idx))

#define GTEN_ASSERT_2D_SHAPE(t, exp_d0, exp_d1) \
    GTEN_ASSERT(t.shape_eq({exp_d0, exp_d1}), \
        "Expected shape=(%d, %d) but got shape=%s", \
        exp_d0, exp_d1, t.shape_str().c_str())

#define GTEN_ASSERT_3D_SHAPE(t, exp_d0, exp_d1, exp_d2) \
    GTEN_ASSERT(t.shape_eq({exp_d0, exp_d1, exp_d2}), \
        "Expected shape=(%d, %d, %d) but got shape=%s", \
        exp_d0, exp_d1, exp_d2, t.shape_str().c_str())

#define GTEN_ASSERT_SHAPE_EQ(t0, t1)     \
    GTEN_ASSERT(t0.shape_eq(t1.shape()), \
        "Expected tensors to have same shape but got %s and %s.", \
        t0.shape_str().c_str(), t1.shape_str().c_str());

// #define GTEN_ASSERT_NDIMS_DTYPE(t, exp_ndims, exp_dtype) \
//     GTEN_ASSERT_NDIMS(t, exp_ndims)                      \
//     GTEN_ASSERT_DTYPE(t, exp_dtype)

#define GTEN_ASSERT_NDIMS_DTYPE(t, exp_ndims, exp_dtype)

#define GTEN_VALIDATE_CTX_OFFS(offs, n_ctx) \
    GTEN_ASSERT(offs >= 0, "Expected a non-negative ctx offset but got %d", offs) \
    GTEN_ASSERT(offs < n_ctx, "Expectec ctx_offs=%d < n_ctx=%d", offs, n_ctx)


namespace gten {
namespace ops {


inline float deq(Qint8 x, float s, int z) {
    return float(int(x) - z) * s;
}


static void tensor_row_index_impl(const Tensor& src, const Tensor& indices, Tensor& out, int ctx_offs) {
    const Float16* src_data = src.data_ptr<Float16>();
    const int* indices_data = indices.data_ptr<int>();
    Float16* out_data = out.data_ptr<Float16>();
    const int rowsize = src.size(1);
    const size_t rowsizebytes = rowsize * src.itemsize();

    for (int i = ctx_offs; i < indices.numel(); i++) {
        const void* src_data_start = src_data + indices_data[i] * rowsize;
        void* out_data_start = out_data + i * rowsize;
        std::memcpy(out_data_start, src_data_start, rowsizebytes);
    }
}

static void tensor_row_index_impl_q8(const Tensor& src, const Tensor& indices, Tensor& out, int ctx_offs) {
    const Qint8* src_data = src.data_ptr<Qint8>();
    const int* indices_data = indices.data_ptr<int>();
    Float16* out_data = out.data_ptr<Float16>();
    const int rowsize = src.size(1);

    const float s = src.scale();
    const int z = src.zerop();

    for (int i = ctx_offs; i < indices.numel(); i++) {
        const Qint8* src_data_start = src_data + indices_data[i] * rowsize;
        Float16* out_data_start = out_data + i * rowsize;
        for (int j = 0; j < rowsize; j++) {
            out_data[i * rowsize + j] = fpcvt_stoh(deq(src_data[indices_data[i] * rowsize + j], s, z));
        }
    }
}

/// @brief Copies the indexed rows of the source tensor to output tensor.
/// @param src A 2-d tensor to be indexed.
/// @param indices A 1-d tensor of indices with dtype = int.
/// @param out A 2d tensor with enough capacity to fit the indexed rows. Its dtype
///  must be the same as source tensor.
void tensor_row_index(const Tensor& src, const Tensor& indices, Tensor& out, int ctx_offs = 0) {
    GTEN_ASSERT_NDIMS_DTYPE(src, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(indices, 1, kInt32);
    GTEN_ASSERT_NDIMS_DTYPE(out, 2, kFloat16);
    GTEN_ASSERT_2D_SHAPE(out, indices.size(0), src.size(1));

    if (src.dtype() == kQint8) {
        tensor_row_index_impl_q8(src, indices, out, ctx_offs);
    } else {
        tensor_row_index_impl(src, indices, out, ctx_offs);
    }
}

static float dot_product(const Float16* vec_a, const Float16* vec_b, int vec_size) {
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
        const float x0 = fpcvt_htos(vec_a[i]);
        const float x1 = fpcvt_htos(vec_b[i]);
        dot_prod += x0 * x1;
    }
    
    return dot_prod;
}


static float dot_product(const Float16* x, const Qint8* w, float ws, int wz, int vec_size) {    
    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;
    
    Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();

    float b_data[8];
    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(x + i);

        b_data[0] = deq(w[i], ws, wz);
        b_data[1] = deq(w[i+1], ws, wz);
        b_data[2] = deq(w[i+2], ws, wz);
        b_data[3] = deq(w[i+3], ws, wz);
        b_data[4] = deq(w[i+4], ws, wz);
        b_data[5] = deq(w[i+5], ws, wz);
        b_data[6] = deq(w[i+6], ws, wz);
        b_data[7] = deq(w[i+7], ws, wz);
        const float* b_data_ptr = (float*)&b_data;

        Vec_f32x8 x1 = vec_f32x8_load(b_data_ptr);
        dot_prod_accum = vec_f32x8_fma(x0, x1, dot_prod_accum);
    }
    
    float dot_prod = vec_f32x8_sum(dot_prod_accum);

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fpcvt_htos(x[i]);
        const float x1 = deq(w[i], ws, wz);
        dot_prod += x0 * x1;
    }
    
    return dot_prod;
}


// static float dot_product(const Float16* vec_a, const Qint8* vec_b, float s, int z, int vec_size) {
//     const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;
    
//     Vec_f32x8 dot_prod_accum = vec_f32x8_setzero();

//     float neg_zf = float(-z);
//     __m256 neg_zv = _mm256_broadcast_ss(&neg_zf);
//     __m256 sv = _mm256_broadcast_ss(&s);
//     int b_data[8];

//     for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
//         Vec_f32x8 x0 = vec_f32x8_load(vec_a + i);

//         b_data[0] = vec_b[i];
//         b_data[1] = vec_b[i+1];
//         b_data[2] = vec_b[i+2];
//         b_data[3] = vec_b[i+3];
//         b_data[4] = vec_b[i+4];
//         b_data[5] = vec_b[i+5];
//         b_data[6] = vec_b[i+6];
//         b_data[7] = vec_b[i+7];

//         const int* w0_data_ptr = (int*)&b_data;
//         __m256i w0i = _mm256_loadu_si256((__m256i_u*)w0_data_ptr);
//         __m256  w00 = _mm256_cvtepi32_ps (w0i);
//         __m256  w01 = _mm256_add_ps (w00, neg_zv);
//         __m256  x1 =  _mm256_mul_ps (w01, sv);

//         dot_prod_accum = vec_f32x8_fma(x0, x1, dot_prod_accum);
//     }
    
//     float dot_prod = vec_f32x8_sum(dot_prod_accum);

//     for (int i = simd_vec_size; i < vec_size; i++) {
//         const float x0 = fpcvt_htos(vec_a[i]);
//         const float x1 = deq(vec_b[i], s, z);
//         dot_prod += x0 * x1;
//     }
    
//     return dot_prod;
// }


static void emb_matmul_impl_q8(const Tensor& x, const Tensor& w, Tensor& out) {
    const Float16* x_data = x.data_ptr<Float16>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);

    const float ws = w.scale();
    const int wz = w.zerop();

    #pragma omp parallel for collapse(2)
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < n_vocab; wcol++) {
            const Float16* x_row_data = x_data + xrow * x_st0;
            const Qint8* w_row_data = w_data + wcol * w_st0;
            float dot_prod = dot_product(x_row_data, w_row_data, ws, wz, n_embd);
            out_data[wcol] = dot_prod;
        }
    }
}

static void emb_matmul_impl(const Tensor& x, const Tensor& w, Tensor& out) {
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);

    #pragma omp parallel for collapse(2)
    for (int xrow = n_ctx-1; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < n_vocab; wcol++) {
            const Float16* x_row_data = x_data + xrow * x_st0;
            const Float16* w_row_data = w_data + wcol * w_st0;
            float dot_prod = dot_product(x_row_data, w_row_data, n_embd);
            out_data[wcol] = dot_prod;
        }
    }
}

/// @brief Computes a matmul between input's last ctx vector and emb table to produce logits
///   for the next token.
/// @param x Input tensor of shape (n_ctx, n_embd).
/// @param w Embedding table tensor of shape (n_vocab, n_embd).
/// @param out Output tensor of shape (n_vocab).
static void emb_matmul(const Tensor& x, const Tensor& w, Tensor& out) {
    GTEN_ASSERT_NDIMS_DTYPE(x, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(w, 2, kFloat16);
    GTEN_ASSERT_DIMSIZE(w, 1, x.size(1));
    GTEN_ASSERT_NDIMS_DTYPE(out, 1, kFloat32);
    GTEN_ASSERT_DIMSIZE(out, 0, w.size(0));

    if (w.dtype() == kQint8) {
        emb_matmul_impl_q8(x, w, out);
    } else {
        return emb_matmul_impl(x, w, out);
    }
}

static void affine_proj_2d_impl_q8(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs) {
    const Float16* x_data = x.data_ptr<Float16>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    const Qint8* bias_data = bias.data_ptr<Qint8>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const float ws = w.scale();
    const int wz = w.zerop();
    const float bs = bias.scale();
    const int bz = bias.zerop();

    // 'hardware_concurrency' returns the number of hyperthreads on a system, not necessarily
    // the number of cores. So we divide by two to get the number of physical cores since
    // Intel assigns every core two hyperthreads. This is just an approximation but for modern
    // Intel chips, it is accurate.
    // const int num_threads = std::thread::hardware_concurrency() / 2;

    #pragma omp parallel for collapse(2)
    for (int xrow = ctx_offs; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < d_out; wcol++) {
            const Float16* xrow_data = x_data + xrow * x_st0;
            const Qint8* wrow_data = w_data + wcol * w_st0;
            float dot_prod = dot_product(xrow_data, wrow_data, ws, wz, n_embd);
            float bias_scalar = deq(bias_data[wcol], bs, bz);
            out_data[xrow * out_st0 + wcol] = fpcvt_stoh(dot_prod + bias_scalar);
        }
    }
}

static void affine_proj_2d_impl(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs) {
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

    #pragma omp parallel for collapse(2)
    for (int xrow = ctx_offs; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < d_out; wcol++) {
            const Float16* xrow_data = x_data + xrow * x_st0;
            const Float16* wrow_data = w_data + wcol * w_st0;
            float dot_prod = dot_product(xrow_data, wrow_data, n_embd);
            float bias_scalar = fpcvt_htos(bias_data[wcol]);
            out_data[xrow * out_st0 + wcol] = fpcvt_stoh(dot_prod + bias_scalar);
        }
    }
}


/// @brief Computes matmul plus bias and puts the result in out.
/// @param a A tensor of shape (n_ctx, n_embd).
/// @param b A tensor of shape (d_out, n_embd).
/// @param bias A tensor of shape (d_out).
/// @param out A tensor of shape (n_ctx, d_out).
void affine_proj_2d(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs = 0) {
    GTEN_ASSERT_NDIMS_DTYPE(x, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(w, 2, kFloat16);
    GTEN_ASSERT_DIMSIZE(w, 1, x.size(1));
    GTEN_ASSERT_NDIMS_DTYPE(bias, 1, kFloat16);
    GTEN_ASSERT_DIMSIZE(bias, 0, w.size(0));
    GTEN_ASSERT_2D_SHAPE(out, x.size(0), w.size(0));
    GTEN_VALIDATE_CTX_OFFS(ctx_offs, x.size(0));
    
    if (w.dtype() == kQint8) {
        affine_proj_2d_impl_q8(x, w, bias, out, ctx_offs);
    } else {
        affine_proj_2d_impl(x, w, bias, out, ctx_offs);
    }
}


void affine_proj_2d_transposed_impl(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs) {
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

    #pragma omp parallel for collapse(2)
    for (int xrow = ctx_offs; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < n_embd; wcol++) {
            const Float16* x_row_data = x_data + xrow * x_st0;
            const Float16* w_row_data = w_data + wcol * w_st0;
            const float dot_prod = dot_product(x_row_data, w_row_data, n_embd);
            const float bias_scalar = fpcvt_htos(bias_data[wcol]);
            out_data[wcol * out_st0 + xrow] = fpcvt_stoh(dot_prod + bias_scalar);
        }
    }
}

void affine_proj_2d_transposed_impl_q8(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs) {
    const Float16* x_data = x.data_ptr<Float16>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    const Qint8* bias_data = bias.data_ptr<Qint8>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int d_out = w.size(0);
    const int x_st0 = x.stride(0);
    const int w_st0 = w.stride(0);
    const int out_st0 = out.stride(0);

    const float ws = w.scale();
    const int wz = w.zerop();
    const float bs = bias.scale();
    const int bz = bias.zerop();

    #pragma omp parallel for collapse(2)
    for (int xrow = ctx_offs; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < n_embd; wcol++) {
            const Float16* x_row_data = x_data + xrow * x_st0;
            const Qint8* w_row_data = w_data + wcol * w_st0;
            const float dot_prod = dot_product(x_row_data, w_row_data, ws, wz, n_embd);
            const float bias_scalar = deq(bias_data[wcol], bs, bz);
            out_data[wcol * out_st0 + xrow] = fpcvt_stoh(dot_prod + bias_scalar);
        }
    }
}


/// @brief Computes matmul plus bias and puts the (transposed) result in out.
/// @param x A tensor of shape (n_ctx, n_embd).
/// @param w A tensor of shape (d_out, n_embd).
/// @param bias A tensor of shape (d_out).
/// @param out A tensor of shape (d_out, n_ctx).
void affine_proj_2d_transposed(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs = 0) {
    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int dout = w.size(0);

    GTEN_ASSERT_NDIMS_DTYPE(x, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(w, 2, kFloat16);
    GTEN_ASSERT_DIMSIZE(w, 1, n_embd);
    GTEN_ASSERT_NDIMS_DTYPE(bias, 1, kFloat16);
    GTEN_ASSERT_DIMSIZE(bias, 0, dout);
    GTEN_ASSERT_2D_SHAPE(out, dout, n_ctx);
    GTEN_VALIDATE_CTX_OFFS(ctx_offs, n_ctx);

    if (w.dtype() == kQint8) {
        affine_proj_2d_transposed_impl_q8(x, w, bias, out, ctx_offs);
    } else {
        affine_proj_2d_transposed_impl(x, w, bias, out, ctx_offs);
    }
}


void vec_add(const Float16* a, const Float16* b, Float16* out, int vec_size) {
    const int simd_vec_size = (vec_size / GTEN_SIMD_VEC_SIZE) * GTEN_SIMD_VEC_SIZE;

    for (int i = 0; i < simd_vec_size; i += GTEN_SIMD_VEC_SIZE) {
        Vec_f32x8 x0 = vec_f32x8_load(a + i);
        Vec_f32x8 x1 = vec_f32x8_load(b + i);
        Vec_f32x8 x_sum = vec_f32x8_add(x0, x1);
        vec_f32x8_store(x_sum, out + i);
    }

    for (int i = simd_vec_size; i < vec_size; i++) {
        const float x0 = fpcvt_htos(a[i]);
        const float x1 = fpcvt_htos(b[i]);
        out[i] = fpcvt_stoh(x0 + x1);
    }
}

static void add_impl(const Tensor& x0, const Tensor& x1, Tensor& out, int ctx_offs)
{
    const Float16* x0_data = x0.data_ptr<Float16>();
    const Float16* x1_data = x1.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x0.size(0);
    const int n_embd = x0.size(1);
    const int st0 = x0.stride(0);

    const Float16* x0_offs_data = x0_data + ctx_offs * st0;
    const Float16* x1_offs_data = x1_data + ctx_offs * st0;
    Float16* out_offs_data = out_data + ctx_offs * st0;
    const int vec_size = x0.numel() - ctx_offs * st0; 
    vec_add(x0_offs_data, x1_offs_data, out_offs_data, vec_size);
}

static void add(const Tensor& x0, const Tensor& x1, Tensor& out, int ctx_offs = 0)
{
    GTEN_ASSERT_NDIMS_DTYPE(x0, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(x1, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(out, 2, kFloat16);
    GTEN_ASSERT_SHAPE_EQ(x0, x1);
    GTEN_ASSERT_SHAPE_EQ(x0, out);
    GTEN_VALIDATE_CTX_OFFS(ctx_offs, x0.size(0));

    return add_impl(x0, x1, out, ctx_offs);
}


void gelu_impl(const Tensor& inp, Tensor& out, int ctx_offs)
{
    // TODO: Replace with lookup table.
    const int n_embd = out.size(1);
    
    const Float16* inp_data = inp.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int numel = inp.numel();
    for (int i = ctx_offs * n_embd; i < numel; ++i) {
        float x = fpcvt_htos(inp_data[i]);
        float res = 0.5 * x 
                        * (1.0f + std::tanh(std::sqrt(2.0f / 3.141592653589793f)
                        * (x + 0.044715f * std::pow(x, 3.0f))));
        out_data[i] = fpcvt_stoh(res);
    }
}

/// @brief Computes GELU function on the input and puts the result in out.
/// @param inp Input tensor.
/// @param out Output tensor.
/// TODO: Replace w lookup table.
void gelu(const Tensor& x, Tensor& out, int ctx_offs = 0)
{
    GTEN_ASSERT_NDIMS_DTYPE(x, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(out, 2, kFloat16);
    GTEN_ASSERT_SHAPE_EQ(x, out);
    GTEN_VALIDATE_CTX_OFFS(ctx_offs, x.size(0));

    return gelu_impl(x, out, ctx_offs);
}

static void layer_norm_vec_q8(const Float16* vec, int vec_size, const Qint8* weight, float ws, int wz,
    const Qint8* bias, float bs, int bz, Float16* out)
{
    // Calculate the mean.
    float sum = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        sum += fpcvt_htos(vec[i]);
    }
    const float mean = sum / vec_size;

    // Calculate the std-dev.
    float sum_squares = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        float x = fpcvt_htos(vec[i]);
        sum_squares += (x - mean) * (x - mean);
    }
    const float variance = sum_squares / (float)vec_size;
    const float stddev = std::sqrt(variance);

    // Normalize.
    float eps = 1e-05f;
    for (int i = 0; i < vec_size; i++) {
        float x = fpcvt_htos(vec[i]);
        float w = deq(weight[i], ws, wz);
        float b = deq(bias[i], bs, bz);

        // Epsilon added to standard deviation prevents div by zero.
        float normalized = ((x - mean) / (stddev + eps)) * w + b;
        out[i] = fpcvt_stoh(normalized);
    }
}

static void layer_norm_vec(const Float16* vec, int vec_size, const Float16* weight, const Float16* bias, Float16* out)
{
    // Calculate the mean.
    float sum = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        sum += fpcvt_htos(vec[i]);
    }
    const float mean = sum / vec_size;

    // Calculate the std-dev.
    float sum_squares = 0.0f;
    for (int i = 0; i < vec_size; i++) {
        float x = fpcvt_htos(vec[i]);
        sum_squares += (x - mean) * (x - mean);
    }
    const float variance = sum_squares / (float)vec_size;
    const float stddev = std::sqrt(variance);

    // Normalize.
    float eps = 1e-05f;
    for (int i = 0; i < vec_size; i++) {
        float x = fpcvt_htos(vec[i]);
        float w = fpcvt_htos(weight[i]);
        float b = fpcvt_htos(bias[i]);

        // Epsilon added to standard deviation prevents div by zero.
        float normalized = ((x - mean) / (stddev + eps)) * w + b;
        out[i] = fpcvt_stoh(normalized);
    }
}

static void layer_norm_impl_q8(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Qint8* w_data = w.data_ptr<Qint8>();
    const Qint8* bias_data = bias.data_ptr<Qint8>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int st0 = x.stride(0);

    const float ws = w.scale();
    const int wz = w.zerop();
    const float bs = bias.scale();
    const int bz = bias.zerop();

    for (int xrow = ctx_offs; xrow < n_ctx; xrow++) {
        const Float16* inp_row_data = x_data + xrow * st0;
        Float16* out_row_data = out_data + xrow * st0;

        layer_norm_vec_q8(inp_row_data, st0, w_data, ws, wz, bias_data, bs, bz, out_row_data);
    }
}

static void layer_norm_impl(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs)
{
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    const Float16* bias_data = bias.data_ptr<Float16>();
    Float16* out_data = out.data_ptr<Float16>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int st0 = x.stride(0);

    for (int xrow = ctx_offs; xrow < n_ctx; xrow++) {
        const Float16* inp_row_data = x_data + xrow * st0;
        Float16* out_row_data = out_data + xrow * st0;

        layer_norm_vec(inp_row_data, st0, w_data, bias_data, out_row_data);
    }
}

/// @brief Computes affine layer-normalization.
/// @param x Input tensor of shape (n_ctx, n_embd).
/// @param w Weight tensor of shape (n_embd).
/// @param bias Tensor of shape (n_embd).
/// @param out Tensor of shape (n_ctx, n_embd).
static void layer_norm(const Tensor& x, const Tensor& w, const Tensor& bias, Tensor& out, int ctx_offs = 0)
{
    GTEN_ASSERT_NDIMS_DTYPE(x, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(w, 1, kFloat16);
    GTEN_ASSERT_DIMSIZE(w, 0, x.size(1));
    GTEN_ASSERT_NDIMS_DTYPE(bias, 1, kFloat16);
    GTEN_ASSERT_DIMSIZE(bias, 0, x.size(1));
    GTEN_ASSERT_NDIMS_DTYPE(out, 2, kFloat16);
    GTEN_ASSERT_SHAPE_EQ(out, x);
    GTEN_VALIDATE_CTX_OFFS(ctx_offs, x.size(0));

    if (w.dtype() == kQint8) {
        layer_norm_impl_q8(x, w, bias, out, ctx_offs);
    } else {
        layer_norm_impl(x, w, bias, out, ctx_offs);
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

/// @brief Computes (Q @ K) * scale_factor and masks the results.
/// @param q Tensor of shape (n_head, n_ctx, d_head).
/// @param k Tensor of shape (n_head, n_ctx, d_head).
/// @param qk_out Tensor of shape (n_head, n_ctx, n_ctx).
/// @param ctx_offs Offset to the input context vector.
void qk_masked(const Tensor& q, const Tensor& k, Tensor& qk_out, float scale_factor, int ctx_offs=0) {
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

    for (int h = 0; h < nhead; h++) {
        for (int qrow = ctx_offs; qrow < n_ctx; qrow++) {
            // For each vector in the current head of Q, we only compute the
            // dot_products that are not subsequently masked. That reduces the
            // number of dot products on each head by half.
            const int kcol_max = qrow + 1;
            for (int kcol = 0; kcol < kcol_max; kcol++) {
                const Float16* qrow_data = q_data + (h * qst0 + qrow * qst1);
                const Float16* kcol_data = k_data + (h * kst0 + kcol * kst1); // col_data is contigous.
                const float dot_prod = dot_product(qrow_data ,kcol_data, d_head);
                out_data[h * qkst0 + qrow * qkst1 + kcol] = fpcvt_stoh(dot_prod * scale_factor);
            }
        }
    }

    // masking
    // When offs is non-zero, it means we are only computing attention between the
    // vector representing the previously predicted token and the rest of the
    // vectors in the prompt. if the prev QK was [n_head, n_ctx, n_ctx], the new QK
    // will be of shape [n_head, n_ctx + 1, n_ctx + 1] where a new column and a new
    // row is added. The new column consists of dot products between the earlier
    // prompt vectors and the new prompt vector which need to be masked but we can
    // skip masking because their computations were skipped and thus, in the
    // memory, their values are zero (guaranteed because calloc is used to allocate
    // for the qk acv tensor). Zero values are only correct if we also
    // ignore re-computing softmax over them because values masked to -inf map to zero
    // after softmax. The significance of skipping masking for new columns (except the
    // last column which belongs to the new row) is to allow us to skip recomputing
    // softmax from scratch each time which drags performance down. As for the new row,
    // we can also skip it because none of the values are masked. However, we need to
    // compute softmax over the new row in each head.
    if (ctx_offs == 0) {
        for (int h = 0; h < nhead; h++) {
            /// TODO: Should we skip?
            for (int qrow = 0; qrow < n_ctx; qrow++) {
                const int k_start = qrow + 1;
                for (int kcol = k_start; kcol < n_ctx; kcol++) {
                    out_data[h * qkst0 + qrow * qkst1 + kcol] = fpcvt_stoh(-std::numeric_limits<float>::infinity());
                }
            }
        }
    }
}

/// @brief Computes softmax inline over each subrow of the input.
/// @param qk_acv Tensor of shape (n_head, n_ctx, n_ctx).
void qk_softmax(Tensor& qk_acv, int ctx_offset=0)
{
    Float16* qk_data = qk_acv.data_ptr<Float16>();

    const int nhead = qk_acv.size(0);
    const int q_ctx = qk_acv.size(1);
    const int k_ctx = qk_acv.size(2);

    const int qkst0 = qk_acv.stride(0);
    const int qkst1 = qk_acv.stride(1);

    for (int head = 0; head < nhead; head++) {
        for (int q_row = ctx_offset; q_row < q_ctx; q_row++)
        {
            Float32 max = -std::numeric_limits<float>::infinity();

            const int base_idx = head * qkst0 + q_row * qkst1;

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


/// @brief Computes attention. i.e (QK @ V).
/// @param qk A tensor of shape (nhead, n_ctx, n_ctx).
/// @param v A tensor of shape (n_head, d_head, n_ctx).
/// @param qkv_out A tensor of shape (n_ctx, n_head, d_head).
void qkv_matmul(const Tensor& qk, const Tensor& v, Tensor& qkv_out, int offs=0) {
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

    // out: [c, h, d]
    for (int h = 0; h < nhead; h++) {
        for (int qkr = offs; qkr < n_ctx; qkr++) {
            for (int vc = 0; vc < dhead; vc++) {
                const Float16* qkr_data = qk_data + (h * qkst0 + qkr * qkst1);
                const Float16* vc_data = v_data + (h * vst0 + vc * vst1);
                const float dot_prod = dot_product(qkr_data, vc_data, n_ctx);
                out_data[h * qkv_st1 + qkr * qkv_st0 + vc] = fpcvt_stoh(dot_prod);
            }
        }
    }
    
}


static void qkv_attn_impl(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, int max_ctx, int ctx_offs)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);
    const int nhead = qk.size(0);
    const int dhead = n_embd / nhead;

    const Tensor q0 = q.view({n_ctx, nhead, dhead}).permute({1, 0, 2});
    const Tensor k0 = k.view({n_ctx, nhead, dhead}).permute({1, 0, 2});
    qk.set_strides({max_ctx * max_ctx, max_ctx, 1});

    const Float32 scale_factor = 1.0f / std::sqrt((Float32)dhead);
    ops::qk_masked(q0, k0, qk, scale_factor, ctx_offs);
    ops::qk_softmax(qk, ctx_offs);

    Tensor v0 = v.view({nhead, dhead, n_ctx});
    v0.set_strides({dhead * max_ctx, max_ctx, 1});

    Tensor qkv0 = qkv.view({n_ctx, nhead, dhead});
    ops::qkv_matmul(qk, v0, qkv0, ctx_offs);
}

/// @brief Computes attention scale(Q @ K)V.
/// @param q Tensor of shape (n_ctx, n_embd).
/// @param k Tensor of shape (n_ctx, n_embd).
/// @param v Tensor of shape (n_embd, n_ctx).
/// @param qk Tensor of shape (nhead, n_ctx, n_ctx).
/// @param qkv Output tensor of shape (n_ctx, n_embd).
/// @param max_ctx Maximum context size. It allows skipping computations for
///   cached qk values when ctx_offs is provided. 
static void qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, int max_ctx, int ctx_offs = 0)
{
    const int n_ctx = q.size(0);
    const int n_embd = q.size(1);

    GTEN_ASSERT_NDIMS_DTYPE(q, 2, kFloat16);
    GTEN_ASSERT_NDIMS_DTYPE(k, 2, kFloat16);
    GTEN_ASSERT_SHAPE_EQ(q, k);
    GTEN_ASSERT_NDIMS_DTYPE(v, 2, kFloat16);
    GTEN_ASSERT_2D_SHAPE(v, n_embd, n_ctx);
    GTEN_ASSERT_NDIMS_DTYPE(qk, 3, kFloat16);
    GTEN_ASSERT_3D_SHAPE(qk, qk.size(0), n_ctx, n_ctx);
    GTEN_ASSERT_NDIMS_DTYPE(qkv, 2, kFloat16);
    GTEN_ASSERT_2D_SHAPE(qkv, n_ctx, n_embd);
    GTEN_ASSERT(max_ctx > 0, "max_ctx must be gt 0");
    GTEN_VALIDATE_CTX_OFFS(ctx_offs, n_ctx);

    qkv_attn_impl(q, k, v, qk, qkv, max_ctx, ctx_offs);
}

} // namespace ops
} // namespace gten
