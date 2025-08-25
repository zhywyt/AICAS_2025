#ifndef QWEN_INFERENCE_RUN_STATE_H
#define QWEN_INFERENCE_RUN_STATE_H

#include "config.h"
#include "memory_manager.h"
#include <memory>

namespace qwen {

/**
 * @brief Runtime state for transformer inference (the "wave" of activations)
 */
class RunState {
public:
    // Current wave of activations
    std::unique_ptr<float[]> x;      // activation at current time stamp (dim,)
    std::unique_ptr<float[]> xb;     // same, but inside a residual branch (dim,)
    std::unique_ptr<float[]> xb2;    // an additional buffer just for convenience (dim,)
    std::unique_ptr<float[]> hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    std::unique_ptr<float[]> q;      // query (dim,)
    std::unique_ptr<float[]> k;      // key (dim,)
    std::unique_ptr<float[]> v;      // value (dim,)
    std::unique_ptr<float[]> att;    // buffer for scores/attention values (n_heads, seq_len)
    std::unique_ptr<float[]> logits; // output logits
    
    // KV cache
    std::unique_ptr<float[]> key_cache;   // (layer, seq_len, dim)
    std::unique_ptr<float[]> value_cache; // (layer, seq_len, dim)

    RunState() = default;
    ~RunState() = default;

    // Non-copyable
    RunState(const RunState&) = delete;
    RunState& operator=(const RunState&) = delete;

    // Movable
    RunState(RunState&&) = default;
    RunState& operator=(RunState&&) = default;

    /**
     * @brief Initialize all buffers based on model configuration
     */
    bool initialize(const Config& config);

    /**
     * @brief Check if all buffers are properly allocated
     */
    bool is_valid() const;

    /**
     * @brief Get raw pointers for performance-critical operations
     */
    float* get_x() const { return x.get(); }
    float* get_xb() const { return xb.get(); }
    float* get_xb2() const { return xb2.get(); }
    float* get_hb() const { return hb.get(); }
    float* get_hb2() const { return hb2.get(); }
    float* get_q() const { return q.get(); }
    float* get_k() const { return k.get(); }
    float* get_v() const { return v.get(); }
    float* get_att() const { return att.get(); }
    float* get_logits() const { return logits.get(); }
    float* get_key_cache() const { return key_cache.get(); }
    float* get_value_cache() const { return value_cache.get(); }
};

} // namespace qwen

#endif // QWEN_INFERENCE_RUN_STATE_H