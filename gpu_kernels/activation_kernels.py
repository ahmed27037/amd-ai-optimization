"""
Activation Function Kernels

Optimized kernels for activation functions
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class OpenCLActivationKernel:
    """
    OpenCL-based activation function kernels
    """
    
    def __init__(self, activation_type: str = "relu"):
        """
        Initialize activation kernel
        
        Args:
            activation_type: Activation type ("relu", "gelu", "silu", "sigmoid", "tanh")
        """
        self.activation_type = activation_type
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply activation function
        
        Args:
            data: Input array
        
        Returns:
            Activated array
        """
        if self.activation_type == "relu":
            return np.maximum(0, data)
        elif self.activation_type == "gelu":
            # GELU approximation
            return data * 0.5 * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (data + 0.044715 * data ** 3)
            ))
        elif self.activation_type == "silu":
            return data * (1 / (1 + np.exp(-data)))
        elif self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-data))
        elif self.activation_type == "tanh":
            return np.tanh(data)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")
    
    def apply_gpu(self, data: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated activation (simulated)
        """
        logger.debug(f"GPU activation ({self.activation_type}) on shape {data.shape}")
        return self.apply(data)


# OpenCL activation kernel sources

RELU_KERNEL = """
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int N
) {
    int i = get_global_id(0);
    if (i < N) {
        output[i] = fmax(0.0f, input[i]);
    }
}
"""

GELU_KERNEL = """
#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEF 0.044715f

__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int N
) {
    int i = get_global_id(0);
    if (i < N) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        output[i] = 0.5f * x * (1.0f + tanh(tanh_arg));
    }
}
"""

SILU_KERNEL = """
__kernel void silu(
    __global const float* input,
    __global float* output,
    const int N
) {
    int i = get_global_id(0);
    if (i < N) {
        float x = input[i];
        output[i] = x / (1.0f + exp(-x));
    }
}
"""

