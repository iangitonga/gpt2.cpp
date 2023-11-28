#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "model.h"


using namespace gten;


const char *usage = R"(
usage:
gpt2 [options] -p PROMPT  for a single prompt or
gpt2 [options] for a chat interface. 

Optional args.
-sm :      Use small model (117M) for inference. [default].
-md :      Use medium model (345M) for inference.
-lg :      Use large model (762M) for inference.
-xl :      Use extra-large model (1.5B) for inference.
-q8 :      [Experimental]: Use quantized model for inference. Only for large and x-large models.
-debug   : See debug-level information.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--len  N : Number of tokens to generate. Minimum is 1 and max is 1000. [default=200].
--topk K : Top tokens to randomly select from during prediction. [default=40].

Examples:
  ./gpt2 -p "Once upon a time" 
  ./gpt2 -lg -p "Once upon a time"
  ./gpt2 -lg --temp 0.5 -p "Once upon a time"
  ./gpt2
)";

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cerr << "Prompt is not provided.\n";
		std::cerr << usage << "\n";
		return -1;
    }
    
    InferenceOptions options{};

    // -sm -q8 -greedy -debug -stat -p "Once upon a time"
    // options.model_name = "Gpt2";
    // options.dtype = kQint8;
    // options.greedy = true;
    // options.debug_mode = true;
    // options.showstat = true;
    // options.prompt = "Once upon a time";

    for (int i = 1; i < argc; i++)
    {
        std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            std::cout << usage << "\n";
            return -1;
        }
        else if (arg == "-p") {
            if (argc <= i+1) {
                std::cout << "Prompt is missing.\n";
                return -1;
            }
            options.prompt = argv[i+1];
            i += 1;
        }
        else if (arg == "-sm") {
            options.model_name = "Gpt2";
        }
        else if (arg == "-md") {
            options.model_name = "Gpt2-medium";
        }
        else if (arg == "-lg") {
            options.model_name = "Gpt2-large";
        }
        else if (arg == "-xl") {
            options.model_name = "Gpt2-xl";
        }
        else if (arg == "-q8") {
            options.dtype = kQint8;
        }
        else if (arg == "-debug") {
            options.debug_mode = true;
        }
        else if (arg == "-greedy") {
            options.greedy = true;
        }
        else if (arg == "-stat") {
            options.showstat = true;
        }
        else if (arg == "--temp") {
            if (argc <= i+1) {
                std::cerr << "Temp value is missing.\n";
                return -1;
            }
            float temp;
            try {
                temp = std::stof(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid temp value.\n";
                return -1;
            }
            if (temp <= 0.0f) {
                std::cerr << "Temp value must be greater than zero.\n";
                return -1;
            }
            options.temp = temp;
            i += 1; // skip parsed temp.
        }
        else if (arg == "--len") {
            if (argc <= i+1) {
                std::cerr << "Length value is missing.\n";
                return -1;
            }
            int len;
            try {
                len = std::stoi(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid Length value.\n";
                return -1;
            }
            if (len < 1 || len > 1024) {
                std::cerr << "Length must be greater than 1 and less than 1000.\n";
                return -1;
            }
            options.n_predict = len;
            i += 1;
        }
        else if (arg == "--topk") {
            if (argc <= i+1) {
                std::cerr << "topk value is missing.\n";
                return -1;
            }
            int top_k;
            try {
                top_k = std::stoi(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid topk value.\n";
                return -1;
            }
            if (top_k < 1 || top_k > 1000) {
                std::cerr << "topk must be greater than or equal to 1 and less than or equal to 1000.\n";
                return -1;
            }
            options.top_k = top_k;
            i += 1;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            return -1;
        }
    }

    if ((options.model_name != "Gpt2-large" || options.model_name != "Gpt2-xl") && options.dtype == kQint8) {
        std::cerr << "Model `" << options.model_name << "` does not have a quantized version. Try -lg or -xl.\n";
        return -1;
    }

    options.print_debug_info();

    int res = std::system(options.get_dl_command().c_str());
    if (res != 0) {
        std::cerr << "Error: Failed to download " << options.model_name << " model. Check your network connectivity.\n";
        return -1;
    }

    std::ifstream checkpoint{options.get_model_path(), std::ios::binary};
    GTEN_ASSERTM(checkpoint.is_open(), "error opening model: %s", options.get_model_path().c_str());
    verify_magic_number(checkpoint);
    GPT2Config config;
    checkpoint.read(reinterpret_cast<char*>(&config), sizeof(config));
    if (options.debug_mode) {
        std::cout << config;
    }

    GPT2Tokenizer tokenizer = load_tokenizer(checkpoint);
    const int num_prompt_tokens = tokenizer.encode(options.prompt).size();
    const int max_ctx = options.calculate_max_ctx_size(num_prompt_tokens);

    GPT2 model{checkpoint, config, max_ctx, options.dtype};

    if (options.prompt == "") {
        std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
        std::string prompt;
        while (true) {
            std::cout << "\n\n\x1B[0m[You]: ";
            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            options.prompt = prompt;

            if (options.greedy)
                model.greedy_sample(options, tokenizer);
            else
                model.sample(options, tokenizer);

            model.reset_acv_caches();
        }
    }
    else {
        if (options.greedy)
            model.greedy_sample(options, tokenizer);
        else
            model.sample(options, tokenizer);
    }

    return 0;
}
