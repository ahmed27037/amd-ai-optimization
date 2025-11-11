import time
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def start(self, name: str) -> None:
        self.timings[name] = {'start': time.time()}
    
    def end(self, name: str) -> float:
        if name not in self.timings:
            logger.warning(f"Timing '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.timings[name]['start']
        self.timings[name]['duration'] = duration
        return duration
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'timings': self.timings,
            'memory': self.memory_usage,
        }


class PerformanceProfiler(Profiler):
    def __init__(self):
        super().__init__()
        self.measurements: Dict[str, List[float]] = {}
    
    def measure(self, name: str, func, *args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        if name not in self.measurements:
            self.measurements[name] = []
        
        self.measurements[name].append(duration * 1000)
        
        return result
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        if name not in self.measurements:
            return {}
        
        values = np.array(self.measurements[name])
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'count': len(values),
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        return {name: self.get_statistics(name) for name in self.measurements}

