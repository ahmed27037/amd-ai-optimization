"""
ROCm ML Inference Optimization Framework

Provides optimized ML inference on AMD GPUs using ROCm, with custom kernels
and memory optimization techniques.
"""

from .inference_engine import InferenceEngine, OptimizedInferenceEngine
from .custom_kernels import MatrixMultiplyKernel, ConvolutionKernel, ActivationKernel

__all__ = [
    'InferenceEngine',
    'OptimizedInferenceEngine',
    'MatrixMultiplyKernel',
    'ConvolutionKernel',
    'ActivationKernel',
]

