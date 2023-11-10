#include <random>

#include "gten/gten.h"
#include "tokenizer.h"


using namespace gten;


struct InferenceOptions {
    std::string model_name {"Gpt2"};
    std::string prompt {""};
    int max_ctx {1024};
    int n_predict {200}; // number of tokens to generate.
    float temp {0.9f};
    int top_k {40};
    bool debug_mode {false};
    bool greedy {false};
    bool showstat {false};
    TensorDtype dtype {kFloat16};

    std::string get_dl_command() const {
#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
        const std::string py_command = "python";
#else
        const std::string py_command = "python3";
#endif
        if (dtype == kQint8) {
            return py_command + " model_dl.py " + model_name + " qint8";
        } else {
            return py_command + " model_dl.py " + model_name + " float16";
        }
    }
 
    std::string get_model_path() const {
        if (dtype == kQint8) {
            return std::string("models/") + model_name + ".q8.gten";
        } else {
            return std::string("models/") + model_name + ".fp16.gten";
        }
    }

    void print_debug_info() const {
        if (debug_mode) {
            std::cout << "Model name     : " << model_name << "\n";
            std::cout << "Model path     : " << get_model_path() << "\n";
            std::cout << "Inference      : " << dtype_str(dtype) << "\n";
            std::cout << "Temperature    : " << temp << "\n";
            std::cout << "Tokens to gen  : " << n_predict << "\n";
            std::cout << "Top-k          : " << top_k << "\n";
        }
    }

    int calculate_max_ctx_size(int num_prompt_tokens) const {
        // max ctx_size for gpt2 models.
        int max_ctx_size = 1024;

        if (num_prompt_tokens >= max_ctx_size) {
            // Prompt length is too large, quit. Technically, we can allow generation of
            // arbitrary-length documents by selecting the last 1024 context tokens and using
            // that to predict the next token but once the prompt reaches max_ctx_size, caching
            // cannot be used and thus prediction becomes very slow.
            GTEN_ASSERT(false, "Prompt length is too large!");
        }
        // How many tokens: n_predict + prompt tokens
        int ctx_size = num_prompt_tokens + n_predict;

        if (ctx_size > max_ctx_size) {
            return max_ctx_size;
        }

        return ctx_size;
    }

};


struct GPT2Config
{
    int32_t n_vocab, n_ctx, n_embed, n_layer, n_head;

    friend std::ostream& operator<<(std::ostream& stream, const GPT2Config& config)
    {
        stream << "\nGPT2Config:" << '\n'
               << "n_vocab: " << config.n_vocab << '\n'
               << "n_ctx  : " << config.n_ctx << '\n'
               << "n_embed: " << config.n_embed << '\n'
               << "n_layer: " << config.n_layer << '\n'
               << "n_head : "  << config.n_head << '\n';
        return stream;
    }
};


class GPT2 {
public:
    GPT2(std::ifstream& checkpoint, const GPT2Config& config, int max_ctx, TensorDtype wdtype);
    Tensor logits(const Tensor &inp);
    void show_performance(int64_t niter, const InferenceOptions& opts) const;
    void sample(const InferenceOptions& opts, GPT2Tokenizer& tokenizer);
    void greedy_sample(const InferenceOptions& opts, GPT2Tokenizer& tokenizer);
    void reset_acv_caches();

public:
    GPT2Config config;

private:
    Embedding wte_;
    PosEmbedding wpe_;
    std::vector<ResidualAttnBlock> blocks_;
    LayerNorm ln_f_;
    Residual res_;
    int64_t time_sample_ms_ = 0;
    int64_t time_load_ms_ = 0;

    void load_from_checkpoint(std::ifstream& checkpoint);
};

static void verify_magic_number(std::ifstream& checkpoint) {
    const int64_t expected_magic = 0x454c49464e455447;
    int64_t magic;
    checkpoint.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERT(magic == expected_magic, "Magic number in the binary does not match the expected one.\n");
}

static GPT2Tokenizer load_tokenizer(std::ifstream& checkpoint) {
    std::string vocab_segment_name;
    int32_t vocab_segment_name_size;
    int32_t vocab_segment_size;
    checkpoint.read(reinterpret_cast<char*>(&vocab_segment_name_size), sizeof(vocab_segment_name_size));
    vocab_segment_name.resize(vocab_segment_name_size);
    checkpoint.read(reinterpret_cast<char*>(vocab_segment_name.data()), vocab_segment_name_size);
    checkpoint.read(reinterpret_cast<char*>(&vocab_segment_size), sizeof(vocab_segment_size));

    return GPT2Tokenizer{checkpoint};
}


GPT2::GPT2(std::ifstream& checkpoint, const GPT2Config& config_, int max_ctx, TensorDtype wdtype)
    : config{config_},
      wte_{Embedding(config.n_vocab, config.n_embed, max_ctx, wdtype)},
      wpe_{PosEmbedding(config.n_ctx, config.n_embed, wdtype)},
      ln_f_{LayerNorm(max_ctx, config.n_embed, wdtype)},
      res_{Residual(max_ctx, config.n_embed)}
{
    blocks_.reserve(config.n_layer);
    for (int i = 0; i < config.n_layer; i++) {
        blocks_.push_back(ResidualAttnBlock(config_.n_head, config_.n_embed, 4*config_.n_embed, max_ctx, wdtype));
    }

    load_from_checkpoint(checkpoint);
}

Tensor GPT2::logits(const Tensor &inp)
{
    Tensor logits = res_.forward(wte_.forward(inp), wpe_.forward(inp.size(0)));
    for (auto &block : blocks_)
        logits = block.forward(logits);
    logits = ln_f_.forward(logits);
    logits = wte_.forward_proj(logits);

    return logits;
}

void GPT2::reset_acv_caches() {
    res_.reset_acv_cache();
    wte_.reset_acv_cache();
    for (auto &block : blocks_)
        block.reset_acv_cache();
    ln_f_.reset_acv_cache();
}


// Used for debugging purposes.
void GPT2::greedy_sample(const InferenceOptions& opts, GPT2Tokenizer& tokenizer)
{
    time_sample_ms_ = 0;

    const int max_ctx_size = 128;

    std::vector<int32_t> tokens = tokenizer.encode(opts.prompt);
    tokens.reserve(max_ctx_size);
    gten::Tensor logits;
    const int logits_size = 50257;

    const int eot_token = 50256;
    const int initial_pos = tokens.size() - 1;
    const int n_iter = max_ctx_size;
    int64_t niter = 0;
    // Use cerr because it is unbuffered.
    std::cerr << "\n\n" << opts.prompt;
    for (int i = initial_pos; i < n_iter; i++)
    {
        // TODO: allow creation of tensors with external non-owning data.
        gten::Tensor input(tokens.data(), {(int32_t)tokens.size()}, gten::kInt32);
        gten::Tensor logits = this->logits(input);

        gten::Timer timer(&time_sample_ms_);
        const float *logits_data = logits.data_ptr<float>();

        float max_prob = -std::numeric_limits<float>::infinity();
        int max_index = 0;
        for (int j = 0; j < logits_size; ++j){
            if (logits_data[j] > max_prob) {
                max_prob = logits_data[j];
                max_index = j;
            }
        }

        int maxi = max_index;
        if (maxi == eot_token)
            break;
        std::cerr << tokenizer.decode(maxi);
        tokens.push_back(maxi);

        niter += 1;
    }
    std::cerr << "\n";

    show_performance(niter, opts);
}

void GPT2::sample(const InferenceOptions& opts, GPT2Tokenizer& tokenizer)
{
    time_sample_ms_ = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int32_t> tokens = tokenizer.encode(opts.prompt);
    const int prompt_length = tokens.size();
    const int max_ctx_size = opts.calculate_max_ctx_size(prompt_length);
    tokens.reserve(max_ctx_size);
    const int logits_size = 50257;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);
    const int eot_token = 50256;

    // Max number of tokens we can produce.
    const int max_iter = max_ctx_size - prompt_length; 
    // Actual num of iters run to report performance per token.
    int64_t n_iter = prompt_length - 1;
    // Use cerr because it is unbuffered.
    std::cerr << "\n[OUTPUT]: \n\n";
    std::cerr << opts.prompt;
    std::cerr << "\x1B[1;34m"; 
    for (int i = 0; i < max_iter; i++)
    {
        n_iter += 1;

        gten::Tensor input{(void*)tokens.data(), {(int32_t)tokens.size()}, gten::kInt32};
        gten::Tensor logits = this->logits(input);

        gten::Timer timer(&time_sample_ms_);
        const float *logits_data = logits.data_ptr<float>();

        logits_probs.clear();
        for (int j = 0; j < logits_size; ++j)
            logits_probs.push_back(std::make_pair((double)logits_data[j] / opts.temp, j));

        const int top_k = opts.top_k;
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j)
        {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j)
            logits_probs[j].first = logits_probs[j].first / sum_exp;

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        uint32_t maxi = dist(gen);
        if (maxi == eot_token) {
            std::cerr << "<|endoftext|>\n";
            break;
        }
        std::cerr << tokenizer.decode(maxi);
        tokens.push_back(maxi);
    }
    std::cerr << "\x1B[0m\n";

    if (opts.showstat)
	   show_performance(n_iter, opts);
}

void GPT2::show_performance(int64_t niter, const InferenceOptions& opts) const
{
    if (niter < 1)
        return;

    int64_t emb_time = wte_.emb_time();
    int64_t emb_proj_time = wte_.emb_proj_time();
    int64_t wpe_time = wpe_.time();
    int64_t linear_time = 0;
    int64_t mlpp_time = 0;
    int64_t attn_lin = 0;
    int64_t attn_time = 0;
    int64_t ln_time = 0;
    int64_t gelu_time = 0;
    int64_t res_time = 0;
    for (const auto &block : blocks_) {
        linear_time += block.time_linear();
        attn_time += block.time_attn();
        ln_time += block.time_ln();
        gelu_time += block.time_gelu();
        mlpp_time += block.time_proj();
        attn_lin += block.time_attn_lin();
        res_time += block.time_res();
    }
    ln_time += ln_f_.time();
    res_time += res_.time();
    const int64_t total = emb_time + emb_proj_time + wpe_time + linear_time + attn_time
                    + ln_time + gelu_time + res_time + time_sample_ms_;

    const int64_t total_inference = emb_time + emb_proj_time + wpe_time + linear_time + attn_time
                    + ln_time + gelu_time + res_time;

    int64_t model_size_mb;
    if (opts.model_name == "Gpt2") {
        model_size_mb = 250;
    } else if (opts.model_name == "Gpt2-medium") {
        model_size_mb = 710;
    } else if (opts.model_name == "Gpt2-large") {
        if (opts.dtype == kFloat16) {
            model_size_mb = 1500;
        } else {
            model_size_mb = 750;
        }
    } else if (opts.model_name == "Gpt2-xl") {
        if (opts.dtype == kFloat16) {
            model_size_mb = 3000;
        } else {
            model_size_mb = 1600;
        }
    } else {
        GTEN_ASSERT(false, "Unexpected model name: %s", opts.model_name.c_str());
    }
    const int64_t tot_mem_usage_mb = G_TensorMemAllocated / (1024 * 1024);
    const int64_t mem_usage_model = model_size_mb;
    const int64_t mem_usage_acvs = tot_mem_usage_mb - mem_usage_model;

    std::cout << "\n-------------------------------\n";
    std::cout << " " << "PERFORMANCE\n";
    std::cout << "-------------------------------\n";
    std::cout << " " << "Mem usage [total]   : " << std::setw(4) << tot_mem_usage_mb << "MB\n";
    std::cout << " " << "Mem usage [model]   : " << std::setw(4) << mem_usage_model << "MB\n";
    std::cout << " " << "Mem usage [actvs]   : " << std::setw(4) << mem_usage_acvs << "MB\n";
    std::cout << "------------------------------\n";
    std::cout << " " << "Inference [per tok] : " << std::setw(5) << total_inference/niter << "ms\n";
    std::cout << " " << "Sample time         : " << std::setw(5) << time_sample_ms_ << "ms\n";
    std::cout << " " << "Load time           : " << std::setw(5) << time_load_ms_ << "ms\n";
    std::cout << " " << "Inference [total]   : " << std::setw(5) << total_inference << "ms\n";
    std::cout << " " << "Total runtime       : " << std::setw(5) << time_load_ms_ + time_sample_ms_ + total_inference << "ms\n\n";

    if (opts.debug_mode) {
        std::cout << "--------------------------------------\n";
        std::cout << "LAYER/OP       | ms/TOK | ms TOTAL\n";
        std::cout << "--------------------------------------\n";
        std::cout << "Embedding      | " << std::setw(3) << emb_time/niter        << "ms | " << std::setw(5) << emb_time << "ms\n";
        std::cout << "Embedding proj | " << std::setw(3) << emb_proj_time/niter   << "ms | " << std::setw(5) << emb_proj_time   << "ms\n";
        std::cout << "Pos embedding  | " << std::setw(3) << wpe_time/niter        << "ms | " << std::setw(5) << wpe_time        << "ms\n";
        std::cout << "Linear (qkv)   | " << std::setw(3) << attn_lin/niter        << "ms | " << std::setw(5) << attn_lin        << "ms\n";
        std::cout << "Linear (mlp)   | " << std::setw(3) << mlpp_time/niter       << "ms | " << std::setw(5) << mlpp_time       << "ms\n";
        std::cout << "Attention      | " << std::setw(3) << attn_time/niter       << "ms | " << std::setw(5) << attn_time       << "ms\n";
        std::cout << "Layer norm     | " << std::setw(3) << ln_time/niter         << "ms | " << std::setw(5) << ln_time         << "ms\n";
        std::cout << "Gelu           | " << std::setw(3) << gelu_time/niter       << "ms | " << std::setw(5) << gelu_time       << "ms\n";
        std::cout << "Residual       | " << std::setw(3) << res_time/niter        << "ms | " << std::setw(5) << res_time        << "ms\n";
        std::cout << "Sampler        | " << std::setw(3) << time_sample_ms_/niter << "ms | " << std::setw(5) << time_sample_ms_ << "ms\n";
        std::cout << "Loading        | " << std::setw(3) << ""                    << "   | " << std::setw(5) << time_load_ms_   << "ms\n";
        std::cout << "--------------------------------------\n";
        std::cout << "TOTAL          | " << std::setw(3) << total/niter    << "ms | " << total << "ms\n";
        std::cout << "--------------------------------------\n";
    }
}


static inline void read_block_header(std::ifstream& fin, bool debug = false)
{
    std::string block_name;
    int32_t block_name_size;
    fin.read(reinterpret_cast<char*>(&block_name_size), sizeof(block_name_size));
    block_name.resize(block_name_size);
    fin.read(reinterpret_cast<char*>(block_name.data()), block_name_size);

    // if (debug)
    //     std::cout << "\n" << "Reading block: " << block_name << "\n";
}

static inline void read_layer_header(std::ifstream& fin, bool debug = false) {
    std::string layer_name;
    int32_t layer_name_size;
    fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
    layer_name.resize(layer_name_size);
    fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

    // if (debug)
    //     std::cout << "Layer: " << layer_name << "\n";
}

static inline void read_into_weight(
    std::ifstream& fin, gten::Tensor& tensor, bool debug = false)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    if (tensor.dtype() == kQint8) {
        float qscale;
        int qzerop;

        fin.read(reinterpret_cast<char*>(&qscale), sizeof(qscale));
        fin.read(reinterpret_cast<char*>(&qzerop), sizeof(qzerop));

        tensor.set_qparams(qscale, qzerop);
    }

    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));

    // if (debug)
        // std::cout << weight_name << " (" << weight_payload_size << ")\n";

    GTEN_ASSERT(
        static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
        "Weight `%s` data size: %ld does not match the expected size: %d.",
        weight_name.c_str(), tensor.nbytes(), weight_payload_size);
    fin.read(tensor.data_ptr<char>(), weight_payload_size);
}

void GPT2::load_from_checkpoint(std::ifstream& checkpoint)
{
    Timer timer{&time_load_ms_};

    // WTE
    read_layer_header(checkpoint);
    read_into_weight(checkpoint, wte_.weight);

    // WPE
    read_layer_header(checkpoint);
    read_into_weight(checkpoint, wpe_.weight);

    // BLOCKS
    for (auto& block : blocks_)
    {
        read_block_header(checkpoint);

        // Query projection layer.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.attn.query.weight);
        read_into_weight(checkpoint, block.attn.query.bias);

        // Key projection layer.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.attn.key.weight);
        read_into_weight(checkpoint, block.attn.key.bias);

        // Value projection layer.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.attn.value.weight);
        read_into_weight(checkpoint, block.attn.value.bias);

        // QKV_out projection layer.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.attn.qkv_proj.weight);
        read_into_weight(checkpoint, block.attn.qkv_proj.bias);

        // Input layernorm.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.attn_ln.weight);
        read_into_weight(checkpoint, block.attn_ln.bias);

        // MLP fully-connected layer.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.mlp_fc.weight);
        read_into_weight(checkpoint, block.mlp_fc.bias);

        // MLP out projection layer.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.mlp_proj.weight);
        read_into_weight(checkpoint, block.mlp_proj.bias);

        // Attention layernorm.
        read_layer_header(checkpoint);
        read_into_weight(checkpoint, block.mlp_ln.weight);
        read_into_weight(checkpoint, block.mlp_ln.bias);
    }
    
    // Block output Layernorm.
    read_layer_header(checkpoint);
    read_into_weight(checkpoint, ln_f_.weight);
    read_into_weight(checkpoint, ln_f_.bias);
}
