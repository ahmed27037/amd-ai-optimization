"""
Reduction Kernels

Provides optimized reduction operations (sum, max, mean) using tree reduction
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ReductionKernel:
    """
    Tree reduction kernel for efficient reduction operations
    """
    
    def __init__(self, reduction_type: str = "sum"):
        """
        Initialize reduction kernel
        
        Args:
            reduction_type: Type of reduction ("sum", "max", "min", "mean")
        """
        self.reduction_type = reduction_type
    
    def reduce(self, data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        Perform reduction operation
        
        Args:
            data: Input array
            axis: Axis to reduce along (None for all axes)
        
        Returns:
            Reduced array
        """
        if self.reduction_type == "sum":
            return np.sum(data, axis=axis)
        elif self.reduction_type == "max":
            return np.max(data, axis=axis)
        elif self.reduction_type == "min":
            return np.min(data, axis=axis)
        elif self.reduction_type == "mean":
            return np.mean(data, axis=axis)
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction_type}")
    
    def reduce_gpu(self, data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """
        GPU-accelerated reduction (simulated)
        
        In production, this would use OpenCL/ROCm kernels with tree reduction
        """
        # For now, use CPU implementation
        # In production, would use optimized GPU kernel
        logger.debug(f"GPU reduction ({self.reduction_type}) on shape {data.shape}")
        return self.reduce(data, axis)


# OpenCL reduction kernel source

REDUCTION_KERNEL = """
__kernel void reduction_sum(
    __global const float* input,
    __global float* output,
    __local float* local_sum,
    const int N
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    local_sum[lid] = (gid < N) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        output[get_group_id(0)] = local_sum[0];
    }
}
"""

REDUCTION_MAX_KERNEL = """
__kernel void reduction_max(
    __global const float* input,
    __global float* output,
    __local float* local_max,
    const int N
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    local_max[lid] = (gid < N) ? input[gid] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        output[get_group_id(0)] = local_max[0];
    }
}
"""

