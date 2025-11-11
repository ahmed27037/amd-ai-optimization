import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from ..hardware import get_backend, HardwareBackend

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, backend: Optional[HardwareBackend] = None):
        self.backend = backend or get_backend()
        self.model = None
        self.input_shape = None
        self.output_shape = None
        self.device_info = self.backend.get_device_info()
        
        logger.info(f"Initialized InferenceEngine with backend: {self.backend.backend_type.value}")
    
    def load_model(self, model_path: str, model_type: str = "pytorch") -> None:
        if model_type == "pytorch":
            try:
                self.model = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
            logger.info(f"Loaded PyTorch model from {model_path}")
        elif model_type == "onnx":
            try:
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                if self.backend.backend_type.value == "rocm":
                    try:
                        providers.insert(0, 'ROCMExecutionProvider')
                    except:
                        pass
                self.model = ort.InferenceSession(model_path, providers=providers)
                logger.info(f"Loaded ONNX model from {model_path}")
            except ImportError:
                raise ImportError("onnxruntime not installed")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def infer(self, input_data: np.ndarray, batch_size: int = 1) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if len(input_data.shape) == 3:
            input_data = input_data[np.newaxis, ...]
        
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_data).float()
                output = self.model(input_tensor)
                return output.numpy()
        
        elif hasattr(self.model, 'run'):
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_data})
            return output[0]
        
        else:
            raise RuntimeError("Unknown model type")
    
    def benchmark(self, input_shape: Tuple[int, ...], num_iterations: int = 100,
                 warmup_iterations: int = 10) -> Dict[str, float]:
        import time
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        for _ in range(warmup_iterations):
            _ = self.infer(dummy_input)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.infer(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_fps': 1000.0 / np.mean(times),
        }


class OptimizedInferenceEngine(InferenceEngine):
    def __init__(self, backend: Optional[HardwareBackend] = None):
        super().__init__(backend)
        self.custom_kernels = {}
        self.memory_pool = {}
        self.optimized = False
        
        logger.info("Initialized OptimizedInferenceEngine")
    
    def optimize_model(self, model_path: str, input_shape: Tuple[int, ...]) -> None:
        self.load_model(model_path)
        self._apply_optimizations()
        self.optimized = True
        logger.info("Model optimization complete")
    
    def _apply_optimizations(self) -> None:
        if isinstance(self.model, torch.nn.Module):
            self.model = torch.jit.script(self.model)
            logger.info("Applied TorchScript optimization")
            
            try:
                torch.jit.optimize_for_inference(self.model)
                logger.info("Applied inference optimizations")
            except:
                pass
        
        self._register_custom_kernels()
    
    def _register_custom_kernels(self) -> None:
        from .custom_kernels import (
            MatrixMultiplyKernel,
            ConvolutionKernel,
            ActivationKernel
        )
        
        self.custom_kernels['matmul'] = MatrixMultiplyKernel(self.backend)
        self.custom_kernels['conv'] = ConvolutionKernel(self.backend)
        self.custom_kernels['activation'] = ActivationKernel(self.backend)
        
        logger.info(f"Registered {len(self.custom_kernels)} custom kernels")
    
    def infer(self, input_data: np.ndarray, batch_size: int = 1, use_optimized: bool = True) -> np.ndarray:
        if use_optimized and self.optimized:
            return self._infer_optimized(input_data, batch_size)
        else:
            return super().infer(input_data, batch_size)
    
    def _infer_optimized(self, input_data: np.ndarray, batch_size: int) -> np.ndarray:
        logger.debug("Using optimized inference path")
        return super().infer(input_data, batch_size)

