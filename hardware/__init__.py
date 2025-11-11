"""
Hardware Abstraction Layer

Provides abstraction for AMD hardware with fallback simulation
when physical hardware is not available.
"""

from .abstraction import HardwareBackend, get_backend, detect_hardware
from .rocm_simulator import ROCmSimulator
from .opencl_backend import OpenCLBackend

__all__ = [
    'HardwareBackend',
    'get_backend',
    'detect_hardware',
    'ROCmSimulator',
    'OpenCLBackend',
]

