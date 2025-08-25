#ifndef QWEN_INFERENCE_TOKENIZER_H
#define QWEN_INFERENCE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace qwen {

/**
 * @brief Token index structure for sorting and lookup
 */
struct TokenIndex {
    std::string str;
    int id;
    
    TokenIndex() = default;
    TokenIndex(const std::string& s, int i) : str(s), id(i) {}
};

/**
 * @brief Tokenizer for encoding/decoding text
 */
class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // bytes for utf8 encoding

    Tokenizer() = default;
    ~Tokenizer() = default;

    // Non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Movable
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;

    /**
     * @brief Build tokenizer from file
     */
    bool build_from_file(const std::string& tokenizer_path, int vocab_size);

    /**
     * @brief Encode text to tokens
     */
    std::vector<int> encode(const std::string& text, bool bos = true, bool eos = true);

    /**
     * @brief Decode single token to string
     */
    std::string decode(int token);

    /**
     * @brief Check if tokenizer is properly initialized
     */
    bool is_valid() const;

    /**
     * @brief Get vocabulary size
     */
    int get_vocab_size() const { return vocab_size; }

private:
    /**
     * @brief Binary search for token string
     */
    int str_lookup(const std::string& str);

    /**
     * @brief Internal encoding helper
     */
    void encode_internal(const std::string& text, std::vector<int>& tokens);
};

} // namespace qwen

#endif // QWEN_INFERENCE_TOKENIZER_H