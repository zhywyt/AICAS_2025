#ifndef QWEN_INFERENCE_TRANSFORMER_H
#define QWEN_INFERENCE_TRANSFORMER_H

#include "config.h"
#include "transformer_weights.h"
#include "run_state.h"
#include "memory_manager.h"
#include <string>

namespace qwen {

/**
 * @brief Main transformer model class
 */
class Transformer {
public:
    Config config;
    TransformerWeights weights;
    RunState state;

    Transformer() = default;
    ~Transformer() = default;

    // Non-copyable
    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    // Movable
    Transformer(Transformer&&) = default;
    Transformer& operator=(Transformer&&) = default;

    /**
     * @brief Load model from checkpoint file
     */
    bool load_from_checkpoint(const std::string& checkpoint_path);

    /**
     * @brief Perform forward pass
     */
    void forward(int token, int pos);

    /**
     * @brief Check if model is properly initialized
     */
    bool is_initialized() const;

private:
    MemoryManager::MemoryMap memory_map_;
    
    /**
     * @brief Read model configuration and weights from checkpoint
     */
    bool read_checkpoint(const std::string& checkpoint_path);
};

} // namespace qwen

#endif // QWEN_INFERENCE_TRANSFORMER_H