import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Quantizer:
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.scale = None
        self.zero_point = None
    
    def quantize(self, data: np.ndarray) -> Tuple[np.ndarray, float, int]:
        raise NotImplementedError
    
    def dequantize(self, quantized_data: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        return (quantized_data.astype(np.float32) - zero_point) * scale


class PTQQuantizer(Quantizer):
    def __init__(self, bits: int = 8, calibration_method: str = "minmax"):
        super().__init__(bits)
        self.calibration_method = calibration_method
        self.calibration_data = []
    
    def calibrate(self, calibration_data: list) -> Dict[str, Any]:
        self.calibration_data = calibration_data
        all_data = np.concatenate([d.flatten() for d in calibration_data])
        
        if self.calibration_method == "minmax":
            min_val = np.min(all_data)
            max_val = np.max(all_data)
        elif self.calibration_method == "percentile":
            min_val = np.percentile(all_data, 0.1)
            max_val = np.percentile(all_data, 99.9)
        else:
            min_val = np.min(all_data)
            max_val = np.max(all_data)
        
        qmin = 0
        qmax = (1 << self.bits) - 1
        
        self.scale = (max_val - min_val) / (qmax - qmin)
        self.zero_point = int(np.round(qmin - min_val / self.scale))
        self.zero_point = np.clip(self.zero_point, qmin, qmax)
        
        logger.info(f"Calibration complete: scale={self.scale:.6f}, zero_point={self.zero_point}")
        
        return {
            'scale': self.scale,
            'zero_point': self.zero_point,
            'min_val': min_val,
            'max_val': max_val,
        }
    
    def quantize(self, data: np.ndarray) -> Tuple[np.ndarray, float, int]:
        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Calibration required before quantization")
        
        qmin = 0
        qmax = (1 << self.bits) - 1
        
        quantized = np.round(data / self.scale + self.zero_point)
        quantized = np.clip(quantized, qmin, qmax).astype(np.int8 if self.bits == 8 else np.int16)
        
        return quantized, self.scale, self.zero_point
    
    def quantize_model(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        model.eval()
        
        if self.bits == 8:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            with torch.no_grad():
                _ = model(sample_input)
            
            torch.quantization.convert(model, inplace=True)
            logger.info("Model quantized to INT8")
        elif self.bits == 16:
            model = model.half()
            logger.info("Model converted to FP16")
        
        return model


class QATQuantizer(Quantizer):
    def __init__(self, bits: int = 8):
        super().__init__(bits)
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        logger.info("Model prepared for quantization-aware training")
        return model
    
    def convert_model(self, model: nn.Module) -> nn.Module:
        model.eval()
        torch.quantization.convert(model, inplace=True)
        logger.info("QAT model converted to quantized model")
        return model

