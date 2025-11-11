from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BackendType(Enum):
    ROCM = "rocm"
    OPENCL = "opencl"
    CPU = "cpu"
    SIMULATED = "simulated"


class HardwareBackend(ABC):
    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type
        self.is_available = False
        self.device_info: Dict[str, Any] = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        pass
    
    @abstractmethod
    def get_device_count(self) -> int:
        pass
    
    @abstractmethod
    def get_device_info(self, device_id: int = 0) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def allocate_memory(self, size: int, device_id: int = 0) -> Any:
        pass
    
    @abstractmethod
    def free_memory(self, ptr: Any) -> None:
        pass
    
    @abstractmethod
    def copy_to_device(self, host_data: Any, device_ptr: Any, size: int) -> None:
        pass
    
    @abstractmethod
    def copy_from_device(self, device_ptr: Any, host_data: Any, size: int) -> None:
        pass
    
    @abstractmethod
    def execute_kernel(self, kernel: Any, args: tuple, work_size: tuple) -> None:
        pass


def detect_hardware() -> BackendType:
    try:
        import subprocess
        result = subprocess.run(['rocminfo'], capture_output=True, timeout=2)
        if result.returncode == 0:
            logger.info("ROCm detected")
            return BackendType.ROCM
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            logger.info(f"OpenCL detected: {len(platforms)} platform(s)")
            return BackendType.OPENCL
    except ImportError:
        pass
    
    logger.info("Falling back to CPU/simulated backend")
    return BackendType.SIMULATED


def get_backend(backend_type: Optional[BackendType] = None) -> HardwareBackend:
    if backend_type is None:
        backend_type = detect_hardware()
    
    if backend_type == BackendType.ROCM:
        try:
            from .rocm_backend import ROCmBackend
            backend = ROCmBackend()
            if backend.initialize():
                return backend
        except ImportError:
            logger.warning("ROCm backend requested but not available")
    
    if backend_type == BackendType.OPENCL:
        try:
            from .opencl_backend import OpenCLBackend
            backend = OpenCLBackend()
            if backend.initialize():
                return backend
        except Exception as e:
            logger.warning(f"OpenCL backend failed: {e}")
    
    from .rocm_simulator import ROCmSimulator
    backend = ROCmSimulator()
    backend.initialize()
    return backend

