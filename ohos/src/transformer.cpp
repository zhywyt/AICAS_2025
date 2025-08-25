#include "transformer.h"
#include "math_utils.h"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace qwen {

bool Transformer::load_from_checkpoint(const std::string& checkpoint_path) {
    return read_checkpoint(checkpoint_path);
}

bool Transformer::read_checkpoint(const std::string& checkpoint_path) {
    // Create memory map
    memory_map_ = MemoryManager::MemoryMap(checkpoint_path.c_str());
    if (!memory_map_.is_valid()) {
        return false;
    }
    
    // Read config from beginning of file
    const Config* config_ptr = static_cast<const Config*>(memory_map_.data());
    config = *config_ptr;
    
    // Validate config
    if (!config.is_valid()) {
        return false;
    }
    
    // Check for shared weights (negative vocab size is hacky way of signaling unshared weights)
    bool shared_weights = config.vocab_size > 0;
    config.vocab_size = abs(config.vocab_size);
    
    // Map weights from memory
    float* weights_ptr = static_cast<float*>(memory_map_.data()) + sizeof(Config) / sizeof(float);
    weights.map_from_memory(config, weights_ptr, shared_weights, true); // Assume bias is present
    
    // Initialize run state
    if (!state.initialize(config)) {
        return false;
    }
    
    return weights.is_valid(config, true);
}

void Transformer::forward(int token, int pos) {
    // Convenience variables
    Config* p = &config;
    TransformerWeights* w = &weights;
    RunState* s = &state;
    float* x = s->get_x();
    int dim = p->dim;
    int kv_dim = p->get_kv_dim();
    int kv_mul = p->get_kv_mul();
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    // Copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    MathUtils::copy_array(x, content_row, dim);
    
    // Forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        
        // Attention rmsnorm
        MathUtils::rmsnorm(s->get_xb(), x, w->rms_att_weight + l * dim, dim);
        
        // Key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->get_key_cache() + loff + pos * kv_dim;
        float* value_cache_row = s->get_value_cache() + loff + pos * kv_dim;
        
        // QKV matmuls for this position
        MathUtils::matmul(s->get_q(), s->get_xb(), w->wq + l * dim * dim, dim, dim);
        MathUtils::matmul(key_cache_row, s->get_xb(), w->wk + l * dim * kv_dim, dim, kv_dim);
        MathUtils::matmul(value_cache_row, s->get_xb(), w->wv + l * dim * kv_dim, dim, kv_dim);
        
        // Add bias if present
        if (w->bq) {
            for (int i = 0; i < dim; i++) {
                s->get_q()[i] += w->bq[l * dim + i];
            }
        }
        if (w->bk) {
            for (int i = 0; i < kv_dim; i++) {
                key_cache_row[i] += w->bk[l * kv_dim + i];
            }
        }
        if (w->bv) {
            for (int i = 0; i < kv_dim; i++) {
                value_cache_row[i] += w->bv[l * kv_dim + i];
            }
        }
        
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < p->dim; i += head_size) {
            for (int head_dim = i % head_size; head_dim < head_size / 2; head_dim++) {
                float freq = 1.0f / powf(1000000.0f, (float)(head_dim * 2) / (float)head_size);
                float val = pos * freq;
                float fci = sinf(val);
                float fcr = cosf(val);
                
                float q0 = s->get_q()[i + head_dim];
                float q1 = s->get_q()[i + head_dim + head_size / 2];
                s->get_q()[i + head_dim] = q0 * fcr - q1 * fci;
                s->get_q()[i + head_dim + head_size / 2] = q0 * fci + q1 * fcr;
                
                if (i < p->n_kv_heads * head_size) {
                    float k0 = key_cache_row[i + head_dim];
                    float k1 = key_cache_row[i + head_dim + head_size / 2];
                    key_cache_row[i + head_dim] = k0 * fcr - k1 * fci;
                    key_cache_row[i + head_dim + head_size / 2] = k0 * fci + k1 * fcr;
                }
            }
        }
        
        // Multihead attention. Iterate over all heads
        for (int h = 0; h < p->n_heads; h++) {
            // Get query vector for this head
            float* q = s->get_q() + h * head_size;
            // Attention scores for this head
            float* att = s->get_att() + h * p->seq_len;
            
            // Iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // Get key vector for this head and at this timestep
                float* k = s->get_key_cache() + loff + t * kv_dim + (h / kv_mul) * head_size;
                // Calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // Save the score to the attention buffer
                att[t] = score;
            }
            
            // Softmax the scores to get attention weights, from 0..pos inclusively
            MathUtils::softmax(att, pos + 1);
            
            // Weighted sum of the values, store back into xb
            float* xb = s->get_xb() + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // Get the value vector for this head and at this timestep
                float* v = s->get_value_cache() + loff + t * kv_dim + (h / kv_mul) * head_size;
                // Get the attention weight for this timestep
                float a = att[t];
                // Accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        
        // Final matmul to get the output of the attention
        MathUtils::matmul(s->get_xb2(), s->get_xb(), w->wo + l * dim * dim, dim, dim);
        
        // Residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->get_xb2()[i];
        }
        
        // FFN rmsnorm
        MathUtils::rmsnorm(s->get_xb(), x, w->rms_ffn_weight + l * dim, dim);
        
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // First calculate self.w1(x) and self.w3(x)
        MathUtils::matmul(s->get_hb(), s->get_xb(), w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        MathUtils::matmul(s->get_hb2(), s->get_xb(), w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        
        // SiLU non-linearity
        MathUtils::apply_silu(s->get_hb(), hidden_dim);
        
        // Elementwise multiply with w3(x)
        MathUtils::element_mul(s->get_hb(), s->get_hb2(), hidden_dim);
        
        // Final matmul to get the output of the FFN
        MathUtils::matmul(s->get_xb(), s->get_hb(), w->w2 + l * hidden_dim * dim, hidden_dim, dim);
        
        // Residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->get_xb()[i];
        }
    }
    
    // Final rmsnorm
    MathUtils::rmsnorm(x, x, w->rms_final_weight, dim);
    
    // Classifier into logits
    MathUtils::matmul(s->get_logits(), x, w->wcls, dim, config.vocab_size);
}

bool Transformer::is_initialized() const {
    return memory_map_.is_valid() && config.is_valid() && 
           weights.is_valid(config, true) && state.is_valid();
}

} // namespace qwen