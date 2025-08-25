#ifndef QWEN_INFERENCE_CONFIG_H
#define QWEN_INFERENCE_CONFIG_H

#include <cstdint>

namespace qwen {

/**
 * @brief Configuration parameters for the Transformer model
 */
class Config {
public:
    int32_t dim;         // transformer dimension
    int32_t hidden_dim;  // for ffn layers
    int32_t n_layers;    // number of layers
    int32_t n_heads;     // number of query heads
    int32_t n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int32_t vocab_size;  // vocabulary size, usually 4096 (byte-level)
    int32_t seq_len;     // max sequence length

    Config() = default;
    Config(int32_t dim, int32_t hidden_dim, int32_t n_layers, int32_t n_heads, 
           int32_t n_kv_heads, int32_t vocab_size, int32_t seq_len);

    // Validation methods
    bool is_valid() const;
    int32_t get_head_size() const { return dim / n_heads; }
    int32_t get_kv_dim() const { return (dim * n_kv_heads) / n_heads; }
    int32_t get_kv_mul() const { return n_heads / n_kv_heads; }
};

} // namespace qwen

#endif // QWEN_INFERENCE_CONFIG_H