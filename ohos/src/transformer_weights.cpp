#include "transformer_weights.h"

namespace qwen {

void TransformerWeights::map_from_memory(const Config& config, float* ptr, bool shared_weights, bool has_bias) {
    int head_size = config.get_head_size();
    unsigned long long n_layers = config.n_layers;
    
    // Token embedding table
    token_embedding_table = ptr;
    ptr += config.vocab_size * config.dim;
    
    // RMS attention weights
    rms_att_weight = ptr;
    ptr += n_layers * config.dim;
    
    // Query weights
    wq = ptr;
    ptr += n_layers * config.dim * (config.n_heads * head_size);
    
    // Query bias (if present)
    if (has_bias) {
        bq = ptr;
        ptr += n_layers * (config.n_heads * head_size);
    }
    
    // Key weights
    wk = ptr;
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
    
    // Key bias (if present)
    if (has_bias) {
        bk = ptr;
        ptr += n_layers * (config.n_kv_heads * head_size);
    }
    
    // Value weights
    wv = ptr;
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
    
    // Value bias (if present)
    if (has_bias) {
        bv = ptr;
        ptr += n_layers * (config.n_kv_heads * head_size);
    }
    
    // Output weights
    wo = ptr;
    ptr += n_layers * (config.n_heads * head_size) * config.dim;
    
    // RMS FFN weights
    rms_ffn_weight = ptr;
    ptr += n_layers * config.dim;
    
    // FFN weights
    w1 = ptr;
    ptr += n_layers * config.hidden_dim * config.dim;
    
    w2 = ptr;
    ptr += n_layers * config.dim * config.hidden_dim;
    
    w3 = ptr;
    ptr += n_layers * config.hidden_dim * config.dim;
    
    // Final RMS norm
    rms_final_weight = ptr;
    ptr += config.dim;
    
    // Classifier weights (if not shared)
    if (!shared_weights) {
        wcls = ptr;
    } else {
        wcls = token_embedding_table;
    }
}

bool TransformerWeights::is_valid(const Config& config, bool has_bias) const {
    bool valid = token_embedding_table && rms_att_weight && wq && wk && wv && wo &&
                 rms_ffn_weight && w1 && w2 && w3 && rms_final_weight && wcls;
    
    if (has_bias) {
        valid = valid && bq && bk && bv;
    }
    
    return valid;
}

} // namespace qwen