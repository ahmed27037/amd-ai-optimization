import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OpenCLKernelManager:
    def __init__(self):
        self.context = None
        self.queue = None
        self.program = None
        self.kernels = {}
        self._initialized = False
    
    def initialize(self, context=None, queue=None) -> bool:
        try:
            import pyopencl as cl
            
            if context is None:
                platforms = cl.get_platforms()
                if not platforms:
                    logger.warning("No OpenCL platforms found")
                    return False
                
                devices = platforms[0].get_devices(cl.device_type.GPU)
                if not devices:
                    devices = platforms[0].get_devices(cl.device_type.CPU)
                
                if not devices:
                    logger.warning("No OpenCL devices found")
                    return False
                
                self.context = cl.Context([devices[0]])
                self.queue = cl.CommandQueue(self.context)
            else:
                self.context = context
                self.queue = queue
            
            self._initialized = True
            logger.info("OpenCL kernel manager initialized")
            return True
            
        except ImportError:
            logger.warning("pyopencl not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenCL: {e}")
            return False
    
    def build_program(self, kernel_source: str, kernel_name: str) -> Optional[object]:
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            import pyopencl as cl
            
            self.program = cl.Program(self.context, kernel_source).build()
            kernel = getattr(self.program, kernel_name)
            self.kernels[kernel_name] = kernel
            logger.info(f"Built OpenCL kernel: {kernel_name}")
            return kernel
            
        except Exception as e:
            logger.error(f"Failed to build kernel {kernel_name}: {e}")
            return None
    
    def get_kernel(self, kernel_name: str) -> Optional[object]:
        return self.kernels.get(kernel_name)

MATRIX_MULTIPLY_KERNEL = """
__kernel void matrix_multiply(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

TILED_MATRIX_MULTIPLY_KERNEL = """
#define TILE_SIZE 32

__kernel void tiled_matrix_multiply(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int localRow = get_local_id(0);
        int localCol = get_local_id(1);
        int tiledRow = t * TILE_SIZE + localCol;
        int tiledCol = t * TILE_SIZE + localRow;
        
        if (row < M && tiledRow < K) {
            tileA[localRow][localCol] = A[row * K + tiledRow];
        } else {
            tileA[localRow][localCol] = 0.0f;
        }
        
        if (col < N && tiledCol < K) {
            tileB[localRow][localCol] = B[tiledCol * N + col];
        } else {
            tileB[localRow][localCol] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""

VECTOR_ADD_KERNEL = """
__kernel void vector_add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N
) {
    int i = get_global_id(0);
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
"""

