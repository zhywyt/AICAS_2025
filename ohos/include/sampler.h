#ifndef QWEN_INFERENCE_SAMPLER_H
#define QWEN_INFERENCE_SAMPLER_H

#include <vector>
#include <memory>
#include <random>

namespace qwen {

/**
 * @brief Probability index structure for sampling
 */
struct ProbIndex {
    float prob;
    int index;
    
    ProbIndex() = default;
    ProbIndex(float p, int i) : prob(p), index(i) {}
};

/**
 * @brief Text generation sampler with temperature and top-p sampling
 */
class Sampler {
public:
    int vocab_size;
    float temperature;
    float topp;
    std::vector<ProbIndex> probindex; // buffer used in top-p sampling
    std::mt19937 rng; // random number generator

    Sampler() = default;
    ~Sampler() = default;

    // Non-copyable
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // Movable
    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;

    /**
     * @brief Initialize sampler with parameters
     */
    bool initialize(int vocab_size, float temperature, float topp, unsigned long long rng_seed);

    /**
     * @brief Sample next token from logits
     */
    int sample(float* logits);

    /**
     * @brief Check if sampler is properly initialized
     */
    bool is_valid() const;

private:
    /**
     * @brief Apply softmax to logits
     */
    void softmax(float* x, int size);

    /**
     * @brief Sample from probability distribution
     */
    int sample_mult(float* probabilities, int n);

    /**
     * @brief Top-p (nucleus) sampling
     */
    int sample_topp(float* probabilities, int n, float topp);
};

} // namespace qwen

#endif // QWEN_INFERENCE_SAMPLER_H