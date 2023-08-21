/**
 * Implementation of GP2 tokenizer based of code obtained from ggml lib (https://github.com/ggerganov/ggml)
 * which is fantastic.
 * */


#pragma once


#include <cstdint>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <regex>

// vocab bin format
// English only: 1char=1byte
// [4byte size][size n_chars]

namespace gten
{

/// @brief A tokenizer that performs words encoding and token id decoding as-per GPT2 vocabulary.
class GPT2Tokenizer
{
public:
    GPT2Tokenizer() {};

    GPT2Tokenizer(std::ifstream& fin)
    {
        std::string word;
        for (int i = 0; i < n_vocab_; i++)
        {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            token_to_id_[word] = i;
            id_to_token_[i] = word;
        }
    }

    GPT2Tokenizer& operator=(gten::GPT2Tokenizer &&rhs)
    {
        if (this != &rhs) {
        token_to_id_ = std::move(rhs.token_to_id_);
        id_to_token_ = std::move(rhs.id_to_token_);
        }
        return *this;
    }

    // Convert a single token id into text.
    const std::string &decode(const int32_t token_id) { return id_to_token_[token_id]; }

    // Convert a string of arbitrary text to a sequence of tokens ids.
    std::vector<int32_t> encode(const std::string &text) const {
        std::vector<std::string> words;

        // first split the text into words
        std::string str = text;
        std::regex re(pat_);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }

        // find the longest tokens that form the words:
        std::vector<int32_t> tokens;
        for (const auto & word : words)
        {
            if (word.size() == 0) continue;

            int i = 0;
            int n = word.size();
            while (i < n) {
                int j = n;
                while (j > i)
                {
                    auto it = token_to_id_.find(word.substr(i, j-i));
                    if (it != token_to_id_.end()) {
                        tokens.push_back(it->second);
                        i = j;
                        break;
                    }
                    --j;
                }
                if (i == n)
                    break;
                if (j == i)
                {
                    auto sub = word.substr(i, 1);
                    if (token_to_id_.find(sub) != token_to_id_.end())
                        tokens.push_back(token_to_id_.at(sub));
                    else
                        fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                    ++i;
                }
            }
        }

        return tokens;
    }

private:
    const int32_t n_vocab_ = 50257;
    const std::string pat_ = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    std::map<std::string, int32_t> token_to_id_;
    std::map<int32_t, std::string> id_to_token_;
};

} // namespace gten
