"""
Memory monitoring utilities for the housing renting model solver.

This module provides tools for monitoring and managing memory usage during
intensive computation tasks, particularly useful for cluster environments
with strict memory limits.
"""

import psutil
import gc
import os
from typing import Optional
import warnings


def get_memory_usage() -> float:
    """
    Return current memory usage in GB.
    
    Returns:
        float: Current memory usage in gigabytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def get_available_memory() -> float:
    """
    Return available system memory in GB.
    
    Returns:
        float: Available memory in gigabytes.
    """
    return psutil.virtual_memory().available / 1024 / 1024 / 1024


def get_memory_percent() -> float:
    """
    Return current memory usage as percentage of total system memory.
    
    Returns:
        float: Memory usage percentage (0-100).
    """
    return psutil.virtual_memory().percent


def cleanup_if_needed(threshold_gb: float = 150.0) -> bool:
    """
    Trigger garbage collection if memory usage exceeds threshold.
    
    Args:
        threshold_gb (float): Memory threshold in GB. Default is 150 GB.
        
    Returns:
        bool: True if cleanup was triggered, False otherwise.
    """
    current_usage = get_memory_usage()
    if current_usage > threshold_gb:
        gc.collect()
        return True
    return False


def cleanup_if_percent_needed(threshold_percent: float = 80.0) -> bool:
    """
    Trigger garbage collection if memory usage exceeds percentage threshold.
    
    Args:
        threshold_percent (float): Memory threshold as percentage. Default is 80%.
        
    Returns:
        bool: True if cleanup was triggered, False otherwise.
    """
    current_percent = get_memory_percent()
    if current_percent > threshold_percent:
        gc.collect()
        return True
    return False


def log_memory_usage(label: str = "", verbose: bool = True) -> float:
    """
    Log current memory usage with optional label.
    
    Args:
        label (str): Optional label to include in the log message.
        verbose (bool): Whether to print the memory usage.
        
    Returns:
        float: Current memory usage in GB.
    """
    usage = get_memory_usage()
    available = get_available_memory()
    percent = get_memory_percent()
    
    if verbose:
        if label:
            print(f"Memory usage {label}: {usage:.2f} GB ({percent:.1f}% used, {available:.2f} GB available)")
        else:
            print(f"Memory usage: {usage:.2f} GB ({percent:.1f}% used, {available:.2f} GB available)")
    
    return usage


def check_memory_limit(limit_gb: float = 192.0, warning_threshold: float = 0.9) -> bool:
    """
    Check if memory usage is approaching a specified limit.
    
    Args:
        limit_gb (float): Memory limit in GB. Default is 192 GB (cluster limit).
        warning_threshold (float): Threshold as fraction of limit (0-1). Default is 0.9 (90%).
        
    Returns:
        bool: True if under the warning threshold, False if approaching limit.
    """
    current_usage = get_memory_usage()
    warning_level = limit_gb * warning_threshold
    
    if current_usage > warning_level:
        remaining = limit_gb - current_usage
        warnings.warn(
            f"Memory usage ({current_usage:.2f} GB) approaching limit ({limit_gb:.2f} GB). "
            f"Only {remaining:.2f} GB remaining.", 
            ResourceWarning
        )
        return False
    
    return True


def force_cleanup(aggressive: bool = False) -> tuple[float, float]:
    """
    Force garbage collection and return before/after memory usage.
    
    Args:
        aggressive (bool): If True, run multiple gc.collect() cycles.
        
    Returns:
        tuple[float, float]: Memory usage before and after cleanup in GB.
    """
    before = get_memory_usage()
    
    if aggressive:
        # Run multiple GC cycles for more thorough cleanup
        for _ in range(3):
            gc.collect()
    else:
        gc.collect()
    
    after = get_memory_usage()
    freed = before - after
    
    if freed > 0.1:  # Only log if significant memory was freed
        print(f"Memory cleanup: {before:.2f} GB → {after:.2f} GB (freed {freed:.2f} GB)")
    
    return before, after


class MemoryMonitor:
    """
    Context manager for monitoring memory usage during code execution.
    
    Example:
        with MemoryMonitor("Model solving") as monitor:
            # Your memory-intensive code here
            result = solve_model()
        
        print(f"Peak memory: {monitor.peak_usage:.2f} GB")
    """
    
    def __init__(self, label: str = "Operation", log_start: bool = True, log_end: bool = True):
        """
        Initialize memory monitor.
        
        Args:
            label (str): Label for the operation being monitored.
            log_start (bool): Whether to log memory at start.
            log_end (bool): Whether to log memory at end.
        """
        self.label = label
        self.log_start = log_start
        self.log_end = log_end
        self.start_usage: Optional[float] = None
        self.peak_usage: Optional[float] = None
        self.end_usage: Optional[float] = None
    
    def __enter__(self):
        self.start_usage = get_memory_usage()
        self.peak_usage = self.start_usage
        
        if self.log_start:
            log_memory_usage(f"at start of {self.label}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_usage = get_memory_usage()
        self.peak_usage = max(self.peak_usage or 0, self.end_usage)
        
        if self.log_end:
            delta = self.end_usage - (self.start_usage or 0)
            log_memory_usage(f"at end of {self.label} (Δ{delta:+.2f} GB)")
    
    def update_peak(self):
        """Update peak memory usage with current usage."""
        current = get_memory_usage()
        self.peak_usage = max(self.peak_usage or 0, current)
        return current


def memory_profile_function(func):
    """
    Decorator to profile memory usage of a function.
    
    Example:
        @memory_profile_function
        def expensive_computation():
            # Your code here
            pass
    """
    def wrapper(*args, **kwargs):
        with MemoryMonitor(f"{func.__name__}()") as monitor:
            result = func(*args, **kwargs)
        return result
    
    return wrapper


# Configuration for different environments
MEMORY_CONFIGS = {
    "cluster": {
        "limit_gb": 192.0,
        "warning_threshold": 0.85,
        "cleanup_threshold_gb": 150.0,
        "cleanup_threshold_percent": 80.0
    },
    "local": {
        "limit_gb": 32.0,
        "warning_threshold": 0.90,
        "cleanup_threshold_gb": 25.0,
        "cleanup_threshold_percent": 85.0
    },
    "development": {
        "limit_gb": 16.0,
        "warning_threshold": 0.80,
        "cleanup_threshold_gb": 12.0,
        "cleanup_threshold_percent": 75.0
    }
}


def get_memory_config(environment: str = "cluster") -> dict:
    """
    Get memory configuration for a specific environment.
    
    Args:
        environment (str): Environment name ("cluster", "local", or "development").
        
    Returns:
        dict: Memory configuration parameters.
    """
    return MEMORY_CONFIGS.get(environment, MEMORY_CONFIGS["cluster"]) 