"""
Benchmark Suite

Provides comprehensive benchmarking framework.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite
    
    Provides automated benchmarking for models and operations
    """
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """
        Initialize benchmark suite
        
        Args:
            output_dir: Directory for benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = PerformanceProfiler()
        self.results = []
    
    def run_benchmark(self, name: str, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Run a benchmark
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark: {name}")
        
        # Warmup
        for _ in range(5):
            _ = func(*args, **kwargs)
        
        # Benchmark
        num_iterations = kwargs.pop('iterations', 100)
        durations = []
        
        for _ in range(num_iterations):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            durations.append(duration * 1000)  # Convert to ms
        
        durations = np.array(durations)
        
        benchmark_result = {
            'name': name,
            'mean_ms': float(np.mean(durations)),
            'std_ms': float(np.std(durations)),
            'min_ms': float(np.min(durations)),
            'max_ms': float(np.max(durations)),
            'p50_ms': float(np.percentile(durations, 50)),
            'p95_ms': float(np.percentile(durations, 95)),
            'p99_ms': float(np.percentile(durations, 99)),
            'iterations': num_iterations,
            'throughput_fps': 1000.0 / np.mean(durations),
        }
        
        self.results.append(benchmark_result)
        logger.info(f"Benchmark '{name}' complete: {benchmark_result['mean_ms']:.3f} ms")
        
        return benchmark_result
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """
        Save benchmark results to file
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self, filename: str = "benchmark_report.txt") -> None:
        """
        Generate text report from results
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Benchmark Report\n")
            f.write("=" * 60 + "\n\n")
            
            for result in self.results:
                f.write(f"Benchmark: {result['name']}\n")
                f.write(f"  Mean Latency: {result['mean_ms']:.3f} ms\n")
                f.write(f"  Std Dev: {result['std_ms']:.3f} ms\n")
                f.write(f"  Min: {result['min_ms']:.3f} ms\n")
                f.write(f"  Max: {result['max_ms']:.3f} ms\n")
                f.write(f"  P50: {result['p50_ms']:.3f} ms\n")
                f.write(f"  P95: {result['p95_ms']:.3f} ms\n")
                f.write(f"  Throughput: {result['throughput_fps']:.2f} FPS\n")
                f.write("\n")
        
        logger.info(f"Report saved to {output_path}")

