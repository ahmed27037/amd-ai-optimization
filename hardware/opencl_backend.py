import numpy as np
from typing import Dict, Any, Optional, Tuple
from .abstraction import HardwareBackend, BackendType
import logging

logger = logging.getLogger(__name__)


class OpenCLBackend(HardwareBackend):
    def __init__(self):
        super().__init__(BackendType.OPENCL)
        self.context = None
        self.queue = None
        self.device = None
        self.platform = None
    
    def initialize(self) -> bool:
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            if not platforms:
                logger.warning("No OpenCL platforms found")
                return False
            
            self.platform = platforms[0]
            devices = self.platform.get_devices(cl.device_type.GPU)
            if not devices:
                devices = self.platform.get_devices(cl.device_type.CPU)
            
            if not devices:
                logger.warning("No OpenCL devices found")
                return False
            
            self.device = devices[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            self.device_info = {
                'name': self.device.name,
                'vendor': self.device.vendor,
                'max_compute_units': self.device.max_compute_units,
                'global_mem_size': self.device.global_mem_size,
                'max_work_group_size': self.device.max_work_group_size,
                'local_mem_size': self.device.local_mem_size,
            }
            
            self.is_available = True
            logger.info(f"OpenCL backend initialized: {self.device.name}")
            return True
            
        except ImportError:
            logger.warning("pyopencl not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenCL: {e}")
            return False
    
    def get_device_count(self) -> int:
        try:
            import pyopencl as cl
            devices = self.platform.get_devices(cl.device_type.GPU)
            if not devices:
                devices = self.platform.get_devices(cl.device_type.CPU)
            return len(devices) if devices else 0
        except:
            return 0
    
    def get_device_info(self, device_id: int = 0) -> Dict[str, Any]:
        return self.device_info.copy()
    
    def allocate_memory(self, size: int, device_id: int = 0) -> Any:
        import pyopencl as cl
        buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size)
        return buffer
    
    def free_memory(self, ptr: Any) -> None:
        if hasattr(ptr, 'release'):
            ptr.release()
    
    def copy_to_device(self, host_data: np.ndarray, device_ptr: Any, size: Optional[int] = None) -> None:
        import pyopencl as cl
        cl.enqueue_copy(self.queue, device_ptr, host_data)
        self.queue.finish()
    
    def copy_from_device(self, device_ptr: Any, host_data: Optional[np.ndarray] = None,
                        size: Optional[int] = None) -> np.ndarray:
        import pyopencl as cl
        if host_data is None:
            host_data = np.empty_like(device_ptr)
        cl.enqueue_copy(self.queue, host_data, device_ptr)
        self.queue.finish()
        return host_data
    
    def execute_kernel(self, kernel: Any, args: tuple, work_size: Tuple[int, ...]) -> None:
        global_size = work_size
        local_size = None
        
        kernel.set_args(*args)
        import pyopencl as cl
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
        self.queue.finish()

