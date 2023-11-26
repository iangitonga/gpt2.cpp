#include <fstream>
#include <iostream>

#include "model.h"

// COMPILE:  g++ -I../ -std=c++17 -O3 -ffast-math -mavx -mf16c -o perplexity perplexity.cpp


static void logits_q8(const Tensor& x, const Tensor& w, Tensor& out)
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

    #pragma omp parallel for
    for (int xrow = 0; xrow < n_ctx; xrow++) {
        const Qint8* x_row_data = x_data + xrow * n_embd;
        const Float16* x_row_deltas = x_qparams.row_deltas(xrow);

        for (int wrow = 0; wrow < n_vocab; wrow++) {
            const Qint8* w_row_data = w_data + wrow * n_embd;

            const Float16* w_row_deltas = w_qparams.row_deltas(wrow);
            const float dot_prod = ops::vec_dot_product_q8(x_row_data, x_row_deltas, w_row_data, w_row_deltas, block_size, n_embd);
            out_data[xrow * n_vocab + wrow] = dot_prod;
        }
    }
}


void logits_f16(const Tensor& x, const Tensor& w, Tensor& out) {
    const Float16* x_data = x.data_ptr<Float16>();
    const Float16* w_data = w.data_ptr<Float16>();
    float* out_data = out.data_ptr<float>();

    const int n_ctx = x.size(0);
    const int n_embd = x.size(1);
    const int n_vocab = w.size(0);

    #pragma omp parallel for collapse(2)
    for (int xrow = 0; xrow < n_ctx; xrow++) {
        for (int wcol = 0; wcol < n_vocab; wcol++) {
            const Float16* x_row_data = x_data + xrow * n_embd;
            const Float16* w_row_data = w_data + wcol * n_embd;
            float dot_prod = gten::ops::vec_dot_product_f16(x_row_data, w_row_data, n_embd);
            out_data[xrow * n_vocab + wcol] = dot_prod;
        }
    }
}

void logits(const Tensor& x, const Tensor& w, Tensor& out, Dtype dtype) {
    if (dtype == kQint8) {
        assert(w.is_quantized());
        assert(x.is_quantized());

        logits_q8(x, w, out);
    } else {
        logits_f16(x, w, out);
    }
}

int main(int argc, char const *argv[])
{
    if (argc < 7) {
        std::cout << "Usage: ./perplexity -m MODEL_PATH -d DATATYPE[f16|qint8] -t TOKENS_PATH" << "\n";
        return -1;
    }

    const std::string model_path = argv[2];
    const std::string dtype_string = argv[4];
    Dtype model_dtype;
    if (dtype_string == "f16") {
        model_dtype = kFloat16;
    } else if (dtype_string == "qint8") {
        model_dtype = kQint8;
    } else {
        std::cout << "Wrong dtype: " << dtype_string << "\n";
        std::cout << "Usage: ./perplexity -m MODEL_PATH -d DATATYPE[f16|qint8] -t TOKENS_PATH" << "\n";
    }
    const std::string tokens_path = argv[6];

    const int ctx_size = 512;
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Dtype : " << dtype_str(model_dtype) << "\n";
    std::cout << "Tokens path : " << tokens_path << "\n";

    std::ifstream ftokens{tokens_path, std::ios_base::binary};
    assert(ftokens.is_open());

    const int file_size_bytes = 45012216;
    const int num_tokens = file_size_bytes / 4;

    const int* tokens_data = new int[num_tokens]; 
    ftokens.read((char*)tokens_data, num_tokens);
    ftokens.close();

    std::ifstream checkpoint{model_path, std::ios::binary};
    assert(checkpoint.is_open());
    verify_magic_number(checkpoint);

    GPT2Config config;
    checkpoint.read(reinterpret_cast<char*>(&config), sizeof(config));
    std::cout << config << "\n\n";
    const int n_embd = config.n_embed;
    const int n_vocab = config.n_vocab;

    GPT2Tokenizer tokenizer = load_tokenizer(checkpoint);
    
    GPT2 model{checkpoint, config, ctx_size, model_dtype};  // Include all tokens.
    const Tensor model_wte = model.wte_weight();

    Tensor emb_proj_acv{{ctx_size, config.n_vocab}, kFloat32};

    std::vector<float> nlls;

    const int num_iters = num_tokens / ctx_size - 1;
    // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    for (int iter_i = 0; iter_i < num_iters; iter_i++)
    {
        model.reset_acv_caches();

        const int* x_data = tokens_data + iter_i * ctx_size;
        const int* y_data = tokens_data + iter_i * ctx_size + 1;
        const Tensor x{x_data, {ctx_size}, kInt32};

        const Tensor pre_logits = model.prelogits(x); // ctx_size, n_embd
        logits(pre_logits, model_wte, emb_proj_acv, model_dtype);
        Tensor logits = emb_proj_acv;
        float* logits_data = logits.data_ptr<float>();    

        for (int i = 0; i < ctx_size; i++)
        {
            float max = -std::numeric_limits<float>::infinity();

            float* row_logits = logits_data + i * n_vocab;

            for (int j = 0; j < n_vocab; j++) {
                const float x = row_logits[j];
                if (x > max)
                    max = x;
            }

            float sum_exp = 0;
            for (int j = 0; j < n_vocab; j++) {
                const float x = row_logits[j];
                const float exp_val = std::exp(x - max);
                row_logits[j] = exp_val;
                sum_exp += exp_val;
            }

            for (int j = 0; j < n_vocab; j++) {
                const float qkw = row_logits[j];
                row_logits[j] = qkw / (sum_exp + 1e-05f);
            }
        }

        float loss_accum = 0.0f;
        for (int i = 0; i < ctx_size; i++)
        {
            const float* next_pred_probs = logits_data + i * n_vocab;
            const float next_pred_prob = -1.0 * std::log(next_pred_probs[y_data[i]]);
            loss_accum += next_pred_prob;
        }

        const float nll_loss = loss_accum / ctx_size;
        nlls.push_back(nll_loss);


        // Compute and print current PPL.
        {
            float nlls_sum = 0.0f;
            for (float nll_val : nlls) {
                nlls_sum += nll_val;
            }
            
            const float current_ppl = expf(nlls_sum / (float)nlls.size());

            std::cerr << "\rIteration [" << iter_i + 1 << "/" << num_iters << "]: PPL: " << current_ppl;
        }
    }

    std::cerr << "\n";
    

    return 0;
}

