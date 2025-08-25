#ifndef QWEN_INFERENCE_CHAT_H
#define QWEN_INFERENCE_CHAT_H

#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include <string>
#include <vector>
#include <functional>

namespace qwen {

/**
 * @brief Performance metrics for inference
 */
struct InferenceMetrics {
    float prefill_throughput = 0.0f;
    float decode_throughput = 0.0f;
    int total_tokens = 0;
    long long prefill_time_ms = 0;
    long long decode_time_ms = 0;
};

/**
 * @brief Chat interface for conversational AI
 */
class Chat {
public:
    // Callback function type for token output
    using TokenCallback = std::function<void(const std::string& token)>;
    
    Chat() = default;
    ~Chat() = default;

    // Non-copyable
    Chat(const Chat&) = delete;
    Chat& operator=(const Chat&) = delete;

    /**
     * @brief Initialize chat with model components
     */
    bool initialize(std::shared_ptr<Transformer> transformer,
                   std::shared_ptr<Tokenizer> tokenizer,
                   std::shared_ptr<Sampler> sampler);

    /**
     * @brief Start chat conversation
     */
    std::string chat(const std::string& user_prompt,
                    const std::string& system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                    int max_steps = 600,
                    TokenCallback callback = nullptr);

    /**
     * @brief Chat without encode/decode (for pre-tokenized input)
     */
    std::vector<int> chat_tokens(const std::vector<int>& prompt_tokens,
                                int max_steps = 600,
                                bool show_output = true);

    /**
     * @brief Get inference metrics from last chat
     */
    const InferenceMetrics& get_metrics() const { return metrics_; }

    /**
     * @brief Check if chat is properly initialized
     */
    bool is_initialized() const;

private:
    std::shared_ptr<Transformer> transformer_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<Sampler> sampler_;
    InferenceMetrics metrics_;

    /**
     * @brief Prepare prompt tokens for chat format
     */
    std::vector<int> prepare_prompt_tokens(const std::string& user_prompt,
                                          const std::string& system_prompt);

    /**
     * @brief Clean spaces for tokenization
     */
    std::string clean_prompt(const std::string& prompt);

    /**
     * @brief Get current time in milliseconds
     */
    long long get_time_ms();
};

} // namespace qwen

#endif // QWEN_INFERENCE_CHAT_H