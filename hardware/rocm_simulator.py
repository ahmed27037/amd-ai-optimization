import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from .abstraction import HardwareBackend, BackendType
import logging

logger = logging.getLogger(__name__)


class ROCmSimulator(HardwareBackend):
    def __init__(self):
        super().__init__(BackendType.SIMULATED)
        self.device_memory: Dict[int, Dict[str, Any]] = {}
        self.memory_allocations: Dict[int, Any] = {}
        self.alloc_counter = 0
        
        self.compute_units = 120
        self.wavefront_size = 64
        self.memory_bandwidth_gbps = 1200
        self.peak_tflops_fp32 = 47.9
        self.peak_tflops_fp16 = 383.0
        self.total_memory_gb = 32
        self.memory_bandwidth_efficiency = 0.85
        
        logger.info("Initializing ROCm Simulator")
    
    def initialize(self) -> bool:
        self.is_available = True
        self.device_info = {
            'name': 'ROCm Simulator (MI100-like)',
            'compute_units': self.compute_units,
            'wavefront_size': self.wavefront_size,
            'memory_bandwidth_gbps': self.memory_bandwidth_gbps,
            'peak_tflops_fp32': self.peak_tflops_fp32,
            'peak_tflops_fp16': self.peak_tflops_fp16,
            'total_memory_gb': self.total_memory_gb,
        }
        logger.info("ROCm Simulator initialized successfully")
        return True
    
    def get_device_count(self) -> int:
        return 1
    
    def get_device_info(self, device_id: int = 0) -> Dict[str, Any]:
        return self.device_info.copy()
    
    def allocate_memory(self, size: int, device_id: int = 0) -> int:
        total_allocated = sum(alloc['size'] for alloc in self.memory_allocations.values())
        max_memory = self.total_memory_gb * 1024 * 1024 * 1024
        
        if total_allocated + size > max_memory:
            raise RuntimeError(f"Out of memory: {size} bytes requested, "
                             f"{max_memory - total_allocated} bytes available")
        
        self.alloc_counter += 1
        alloc_id = self.alloc_counter
        
        self.memory_allocations[alloc_id] = {
            'size': size,
            'data': None,
            'device_id': device_id,
        }
        
        logger.debug(f"Allocated {size} bytes on device {device_id} (alloc_id={alloc_id})")
        return alloc_id
    
    def free_memory(self, ptr: int) -> None:
        if ptr in self.memory_allocations:
            size = self.memory_allocations[ptr]['size']
            del self.memory_allocations[ptr]
            logger.debug(f"Freed {size} bytes (alloc_id={ptr})")
        else:
            logger.warning(f"Attempted to free invalid pointer: {ptr}")
    
    def copy_to_device(self, host_data: np.ndarray, device_ptr: int, size: Optional[int] = None) -> None:
        if device_ptr not in self.memory_allocations:
            raise ValueError(f"Invalid device pointer: {device_ptr}")
        
        if size is None:
            size = host_data.nbytes
        
        transfer_time = size / (self.memory_bandwidth_gbps * 1e9 * self.memory_bandwidth_efficiency)
        self.memory_allocations[device_ptr]['data'] = host_data.copy()
        
        logger.debug(f"Copied {size} bytes to device (simulated {transfer_time*1000:.3f}ms)")
    
    def copy_from_device(self, device_ptr: int, host_data: Optional[np.ndarray] = None, 
                        size: Optional[int] = None) -> np.ndarray:
        if device_ptr not in self.memory_allocations:
            raise ValueError(f"Invalid device pointer: {device_ptr}")
        
        device_data = self.memory_allocations[device_ptr]['data']
        if device_data is None:
            raise RuntimeError(f"Device pointer {device_ptr} has no data")
        
        if size is None:
            size = device_data.nbytes
        
        transfer_time = size / (self.memory_bandwidth_gbps * 1e9 * self.memory_bandwidth_efficiency)
        result = device_data.copy()
        
        logger.debug(f"Copied {size} bytes from device (simulated {transfer_time*1000:.3f}ms)")
        return result
    
    def execute_kernel(self, kernel: Any, args: tuple, work_size: Tuple[int, ...]) -> None:
        total_work_items = np.prod(work_size)
        operations_per_item = getattr(kernel, 'ops_per_item', 100)
        total_ops = total_work_items * operations_per_item
        execution_time = total_ops / (self.peak_tflops_fp32 * 1e12)
        
        logger.debug(f"Executed kernel: {total_work_items} work items, "
                    f"simulated {execution_time*1000:.3f}ms")
    
    def get_memory_usage(self) -> Dict[str, float]:
        total_allocated = sum(alloc['size'] for alloc in self.memory_allocations.values())
        max_memory = self.total_memory_gb * 1024 * 1024 * 1024
        
        return {
            'allocated_gb': total_allocated / (1024**3),
            'total_gb': self.total_memory_gb,
            'usage_percent': (total_allocated / max_memory) * 100,
            'allocations': len(self.memory_allocations),
        }

