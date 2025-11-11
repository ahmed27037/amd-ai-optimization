"""
ROCm Inference Benchmarking Tool

Provides comprehensive benchmarking for ML inference on AMD hardware.
"""

import argparse
import json
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np
import torch
import torchvision.models as models

from .inference_engine import InferenceEngine, OptimizedInferenceEngine
from ..hardware import get_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_model(model_name: str, input_shape: tuple, num_iterations: int = 100,
                   warmup_iterations: int = 10, use_optimized: bool = True) -> Dict[str, Any]:
    """
    Benchmark a model
    
    Args:
        model_name: Name of the model to benchmark
        input_shape: Input shape (N, C, H, W)
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        use_optimized: Whether to use optimized inference engine
    
    Returns:
        Benchmark results dictionary
    """
    logger.info(f"Benchmarking {model_name} with input shape {input_shape}")
    
    # Get hardware backend
    backend = get_backend()
    device_info = backend.get_device_info()
    
    # Load model
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.eval()
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.eval()
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create inference engine
    # Use tempfile for cross-platform temporary file handling
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False, prefix=f'{model_name}_') as tmp_file:
        temp_path = tmp_file.name
    
    try:
        if use_optimized:
            engine = OptimizedInferenceEngine(backend)
            # Save model temporarily and load through engine
            torch.save(model, temp_path)
            engine.load_model(temp_path, "pytorch")
        else:
            engine = InferenceEngine(backend)
            torch.save(model, temp_path)
            engine.load_model(temp_path, "pytorch")
        
        # Run benchmark
        results = engine.benchmark(input_shape, num_iterations, warmup_iterations)
        
        # Add metadata
        results['model_name'] = model_name
        results['input_shape'] = input_shape
        results['backend_type'] = backend.backend_type.value
        results['device_info'] = device_info
        results['num_iterations'] = num_iterations
        
        return results
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description="Benchmark ML inference on AMD hardware")
    parser.add_argument("--model", type=str, default="resnet50",
                       choices=["resnet50", "resnet18", "mobilenet_v2"],
                       help="Model to benchmark")
    parser.add_argument("--input-shape", type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=("N", "C", "H", "W"),
                       help="Input shape (batch, channels, height, width)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--optimized", action="store_true", default=True,
                       help="Use optimized inference engine")
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline (non-optimized) benchmark")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    input_shape = tuple(args.input_shape)
    
    # Run optimized benchmark
    if args.optimized:
        logger.info("Running optimized benchmark...")
        results_optimized = benchmark_model(
            args.model, input_shape, args.iterations, args.warmup, use_optimized=True
        )
        print("\n" + "=" * 60)
        print("Optimized Benchmark Results")
        print("=" * 60)
        print(f"Model: {results_optimized['model_name']}")
        print(f"Backend: {results_optimized['backend_type']}")
        print(f"Mean Latency: {results_optimized['mean_ms']:.3f} ms")
        print(f"Std Dev: {results_optimized['std_ms']:.3f} ms")
        print(f"Min Latency: {results_optimized['min_ms']:.3f} ms")
        print(f"Max Latency: {results_optimized['max_ms']:.3f} ms")
        print(f"P50 Latency: {results_optimized['p50_ms']:.3f} ms")
        print(f"P95 Latency: {results_optimized['p95_ms']:.3f} ms")
        print(f"Throughput: {results_optimized['throughput_fps']:.2f} FPS")
        print("=" * 60)
    
    # Run baseline benchmark
    if args.baseline:
        logger.info("Running baseline benchmark...")
        results_baseline = benchmark_model(
            args.model, input_shape, args.iterations, args.warmup, use_optimized=False
        )
        print("\n" + "=" * 60)
        print("Baseline Benchmark Results")
        print("=" * 60)
        print(f"Model: {results_baseline['model_name']}")
        print(f"Backend: {results_baseline['backend_type']}")
        print(f"Mean Latency: {results_baseline['mean_ms']:.3f} ms")
        print(f"Throughput: {results_baseline['throughput_fps']:.2f} FPS")
        print("=" * 60)
        
        if args.optimized:
            speedup = results_baseline['mean_ms'] / results_optimized['mean_ms']
            print(f"\nSpeedup: {speedup:.2f}x")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            'optimized': results_optimized if args.optimized else None,
            'baseline': results_baseline if args.baseline else None,
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

