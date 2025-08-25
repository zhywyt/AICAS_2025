#ifndef QWEN_INFERENCE_TRANSFORMER_WEIGHTS_H
#define QWEN_INFERENCE_TRANSFORMER_WEIGHTS_H

#include "config.h"
#include "memory_manager.h"

namespace qwen {

/**
 * @brief Container for all transformer model weights
 */
class TransformerWeights {
public:
    // Token embedding table
    float* token_embedding_table = nullptr; // (vocab_size, dim)
    
    // Weights for rmsnorms
    float* rms_att_weight = nullptr; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight = nullptr; // (layer, dim)
    
    // Weights for matmuls. note dim == n_heads * head_size
    float* wq = nullptr; // (layer, dim, n_heads * head_size)
    float* wk = nullptr; // (layer, dim, n_kv_heads * head_size)
    float* wv = nullptr; // (layer, dim, n_kv_heads * head_size)
    float* wo = nullptr; // (layer, n_heads * head_size, dim)
    
    // Bias terms
    float* bq = nullptr; // p->n_heads * head_size
    float* bk = nullptr; // p->n_kv_heads * head_size
    float* bv = nullptr; // p->n_kv_heads * head_size
    
    // Weights for ffn
    float* w1 = nullptr; // (layer, hidden_dim, dim)
    float* w2 = nullptr; // (layer, dim, hidden_dim)
    float* w3 = nullptr; // (layer, hidden_dim, dim)
    
    // Final rmsnorm
    float* rms_final_weight = nullptr; // (dim,)
    
    // Optional classifier weights for the logits, on the last layer
    float* wcls = nullptr;

    TransformerWeights() = default;
    ~TransformerWeights() = default;

    // Non-copyable
    TransformerWeights(const TransformerWeights&) = delete;
    TransformerWeights& operator=(const TransformerWeights&) = delete;

    // Movable
    TransformerWeights(TransformerWeights&&) = default;
    TransformerWeights& operator=(TransformerWeights&&) = default;

    /**
     * @brief Map weights from memory-mapped model file
     */
    void map_from_memory(const Config& config, float* ptr, bool shared_weights, bool has_bias);

    /**
     * @brief Validate that all required weights are initialized
     */
    bool is_valid(const Config& config, bool has_bias) const;
};

} // namespace qwen

#endif // QWEN_INFERENCE_TRANSFORMER_WEIGHTS_H