#ifndef QWEN_INFERENCE_H
#define QWEN_INFERENCE_H

/**
 * @file qwen_inference.h
 * @brief Main header file for Qwen Inference Library
 * 
 * This library provides a C++ interface for running Qwen language model inference
 * on OpenHarmony platform. It includes all necessary components for loading models,
 * tokenizing text, and generating responses.
 */

// Core components
#include "config.h"
#include "memory_manager.h"
#include "transformer_weights.h"
#include "run_state.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "math_utils.h"
#include "chat.h"

namespace qwen {

/**
 * @brief Main inference engine class that orchestrates all components
 */
class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine() = default;

    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /**
     * @brief Initialize the inference engine with model files
     */
    bool initialize(const std::string& model_path,
                   const std::string& tokenizer_path,
                   float temperature = 0.0f,
                   float topp = 0.9f,
                   unsigned long long rng_seed = 0);

    /**
     * @brief Generate response for user prompt
     */
    std::string generate(const std::string& prompt,
                        const std::string& system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                        int max_tokens = 600);

    /**
     * @brief Get chat interface for conversational interactions
     */
    Chat& get_chat() { return chat_; }

    /**
     * @brief Check if engine is initialized
     */
    bool is_initialized() const;

    /**
     * @brief Get model configuration
     */
    const Config& get_config() const;

private:
    std::shared_ptr<Transformer> transformer_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<Sampler> sampler_;
    Chat chat_;
    bool initialized_ = false;
};

} // namespace qwen

#endif // QWEN_INFERENCE_H