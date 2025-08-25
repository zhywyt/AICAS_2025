#ifndef QWEN_INFERENCE_MATH_UTILS_H
#define QWEN_INFERENCE_MATH_UTILS_H

#include <cmath>

namespace qwen {

/**
 * @brief Mathematical utility functions for transformer operations
 */
class MathUtils {
public:
    /**
     * @brief RMS normalization
     */
    static void rmsnorm(float* o, float* x, float* weight, int size);

    /**
     * @brief Softmax activation
     */
    static void softmax(float* x, int size);

    /**
     * @brief Matrix multiplication: xout = x * w
     */
    static void matmul(float* xout, float* x, float* w, int n, int d);

    /**
     * @brief SiLU (Swish) activation function
     */
    static inline float silu(float x) {
        return x / (1.0f + expf(-x));
    }

    /**
     * @brief Apply SiLU activation to array
     */
    static void apply_silu(float* x, int size);

    /**
     * @brief Element-wise multiplication
     */
    static void element_mul(float* a, float* b, int size);

    /**
     * @brief Copy array
     */
    static void copy_array(float* dst, const float* src, int size);
};

} // namespace qwen

#endif // QWEN_INFERENCE_MATH_UTILS_H