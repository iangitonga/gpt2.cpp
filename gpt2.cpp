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
-sm :      Use small model (117M) for inference.
-md :      Use medium model (345M) for inference. This model is chosen by default.
-lg :      Use large model (762M) for inference.
-debug   : See debug-level information.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--len  L : Number of words to generate. Minimum is 1 and max is 1000. [default=200].

Examples:
  ./gpt2 -p "Once upon a time" 
  ./gpt2 -lg -p "Once upon a time"
  ./gpt2 -lg --temp 0.5 -p "Once upon a time"
  ./gpt2
)";


int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "Prompt is not provided.\n";
		std::cout << usage << "\n";
		return -1;
    }
    
    InferenceOptions options{};

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
                std::cout << "Temp value is missing.\n";
                return -1;
            }
            float temp;
            try {
                temp = std::stof(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid temp value.\n";
                return -1;
            }
            if (temp <= 0.0f) {
                std::cout << "Temp value must be greater than zero.\n";
                return -1;
            }
            options.temp = temp;
            i += 1; // skip parsed temp.
        }
        else if (arg == "--len") {
            if (argc <= i+1) {
                std::cout << "Length value is missing.\n";
                return -1;
            }
            int len;
            try {
                len = std::stoi(argv[i+1]);
            } catch (...) {
                std::cout << "Invalid Length value.\n";
                return -1;
            }
            if (len < 1 || len > 1024) {
                std::cout << "Length must be greater than 1 and less than 1000.\n";
                return -1;
            }
            options.gen_tokens = len;
            i += 1;
        }
        else {
            std::cout << "Unknown option: " << arg << "\n";
            return -1;
        }
    }

    options.print_debug_info();

    int res = std::system(options.get_dl_command().c_str());
    if (res != 0) {
        std::cout << "Error: Failed to download " << options.model_name << " model due to network issues.\n";
        return -1;
    }

    std::ifstream checkpoint{options.get_model_path(), std::ios::binary};
    GTEN_ASSERT(checkpoint.is_open(), "error opening model: %s", options.get_model_path().c_str());
    verify_magic_number(checkpoint);
    GPT2Config config;
    checkpoint.read(reinterpret_cast<char*>(&config), sizeof(config));
    // std::cout << config;
    GPT2Tokenizer tokenizer = load_tokenizer(checkpoint);
    const int num_prompt_tokens = tokenizer.encode(options.prompt).size();
    const int max_ctx = options.calculate_max_ctx_size(num_prompt_tokens);

    GPT2 model{checkpoint, config, max_ctx};

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