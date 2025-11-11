"""
GPU Compute Kernel Library

Provides custom OpenCL and ROCm HIP kernels for common ML operations.
"""

from .opencl_kernels import OpenCLKernelManager
from .reduction_kernels import ReductionKernel
from .activation_kernels import OpenCLActivationKernel

__all__ = [
    'OpenCLKernelManager',
    'ReductionKernel',
    'OpenCLActivationKernel',
]

