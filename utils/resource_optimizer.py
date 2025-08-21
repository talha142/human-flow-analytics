# Author: Muneeb Ahmad | mpysolutions.com | fiverr.com/muneeb_ahmad_ch | github.com/Muneeb-Ahmad-Ch 
# © 2025 MPY Solutions. Developed by Muneeb Ahmad & Team. All rights reserved.
# Unauthorized use, distribution, or reproduction of this code is strictly prohibited and not permitted.
# The developer assumes no responsibility for any damages or losses that may result from the use of this code.
# Do not use this code for illegal or unethical activities.
# ==============================================================================
"""
Resource Optimization Utility for Human Flow Tracking
Automatically detects and configures system resources for optimal performance
"""

import os
import sys
import psutil
import torch
from pathlib import Path
import multiprocessing as mp


class ResourceOptimizer:
    """Optimize system resources for video processing"""
    
    def __init__(self, target_utilization=0.9):
        self.target_utilization = target_utilization
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Get comprehensive system information"""
        info = {
            'cpu_count': mp.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'has_cuda': torch.cuda.is_available(),
            'has_mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        # GPU information
        if info['has_cuda']:
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory_gb'] = []
            for i in range(info['gpu_count']):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info['gpu_memory_gb'].append(gpu_memory)
        else:
            info['gpu_count'] = 0
            info['gpu_memory_gb'] = []
            
        return info
    
    def get_optimal_settings(self):
        """Calculate optimal settings for 90% resource utilization"""
        settings = {
            'device': self._get_optimal_device(),
            'memory_limit_mb': self._get_optimal_memory_limit(),
            'batch_size': self._get_optimal_batch_size(),
            'num_workers': self._get_optimal_workers(),
            'model_recommendation': self._get_optimal_model(),
            'processing_mode': self._get_optimal_processing_mode(),
        }
        
        return settings
    
    def _get_optimal_device(self):
        """Select the best processing device"""
        if self.system_info['has_cuda'] and self.system_info['gpu_memory_gb']:
            # Use GPU with most memory
            best_gpu = max(range(len(self.system_info['gpu_memory_gb'])), 
                          key=lambda i: self.system_info['gpu_memory_gb'][i])
            return f"cuda:{best_gpu}"
        elif self.system_info['has_mps']:
            return "mps"
        else:
            return "cpu"
    
    def _get_optimal_memory_limit(self):
        """Calculate optimal memory limit for 90% utilization"""
        if self.system_info['has_cuda'] and self.system_info['gpu_memory_gb']:
            # Use 90% of GPU memory
            max_gpu_memory = max(self.system_info['gpu_memory_gb'])
            return int(max_gpu_memory * 1024 * self.target_utilization)  # Convert to MB
        else:
            # Use 90% of available system memory
            available_memory_gb = self.system_info['memory_available_gb']
            return int(available_memory_gb * 1024 * self.target_utilization)  # Convert to MB
    
    def _get_optimal_batch_size(self):
        """Calculate optimal batch size based on available memory"""
        memory_limit_mb = self._get_optimal_memory_limit()
        
        if memory_limit_mb >= 8192:  # 8GB+
            return 16
        elif memory_limit_mb >= 4096:  # 4GB+
            return 8
        elif memory_limit_mb >= 2048:  # 2GB+
            return 4
        elif memory_limit_mb >= 1024:  # 1GB+
            return 2
        else:
            return 1
    
    def _get_optimal_workers(self):
        """Calculate optimal number of worker threads"""
        # Use 90% of available CPU cores, but leave at least 1 core free
        optimal_workers = max(1, int(self.system_info['cpu_count'] * self.target_utilization))
        return min(optimal_workers, self.system_info['cpu_count'] - 1)
    
    def _get_optimal_model(self):
        """Recommend optimal model based on available resources"""
        memory_limit_mb = self._get_optimal_memory_limit()
        has_gpu = self.system_info['has_cuda'] or self.system_info['has_mps']
        
        if memory_limit_mb >= 6144 and has_gpu:  # 6GB+ with GPU
            return "yolo11l"  # Large model
        elif memory_limit_mb >= 4096 and has_gpu:  # 4GB+ with GPU
            return "yolo11m"  # Medium model
        elif memory_limit_mb >= 2048:  # 2GB+
            return "yolo11s"  # Small model
        else:
            return "yolo11n"  # Nano model
    
    def _get_optimal_processing_mode(self):
        """Recommend optimal processing mode"""
        has_gpu = self.system_info['has_cuda'] or self.system_info['has_mps']
        memory_limit_mb = self._get_optimal_memory_limit()
        
        if has_gpu and memory_limit_mb >= 4096:
            return "High Accuracy"  # Process every frame
        elif has_gpu and memory_limit_mb >= 2048:
            return "Normal"  # Process every 2nd frame
        else:
            return "Fast"  # Process every 5th frame
    
    def configure_torch_optimizations(self):
        """Apply PyTorch optimizations for maximum performance"""
        try:
            # Set number of threads for CPU operations
            torch.set_num_threads(self._get_optimal_workers())
            
            # Enable optimized attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Enable memory efficient attention
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Set CUDA optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
                
                # Set memory fraction to use 90% of GPU memory
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(self.target_utilization, i)
            
            print(f"[INFO] Applied PyTorch optimizations:")
            print(f"  - CPU threads: {torch.get_num_threads()}")
            print(f"  - CUDA benchmark: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'N/A'}")
            print(f"  - Memory fraction: {self.target_utilization}")
            
        except Exception as e:
            print(f"[WARN] Failed to apply some PyTorch optimizations: {e}")
    
    def set_environment_variables(self):
        """Set environment variables for optimal performance"""
        env_vars = {
            # OpenMP settings for CPU parallelization
            'OMP_NUM_THREADS': str(self._get_optimal_workers()),
            'MKL_NUM_THREADS': str(self._get_optimal_workers()),
            'NUMEXPR_NUM_THREADS': str(self._get_optimal_workers()),
            
            # CUDA settings
            'CUDA_LAUNCH_BLOCKING': '0',  # Async CUDA operations
            'CUDA_CACHE_DISABLE': '0',    # Enable CUDA caching
            
            # Memory settings
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
        print(f"[INFO] Set environment variables for optimization:")
        for key, value in env_vars.items():
            print(f"  - {key}={value}")
    
    def print_optimization_summary(self):
        """Print a summary of the optimization settings"""
        settings = self.get_optimal_settings()
        
        print("\n" + "="*60)
        print("RESOURCE OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"System Resources:")
        print(f"  - CPU Cores: {self.system_info['cpu_count']} ({self.system_info['cpu_count_physical']} physical)")
        print(f"  - Memory: {self.system_info['memory_total_gb']:.1f} GB total, {self.system_info['memory_available_gb']:.1f} GB available")
        print(f"  - GPU: {self.system_info['gpu_count']} devices, {self.system_info['gpu_memory_gb']} GB memory")
        
        print(f"\nOptimal Settings (90% utilization):")
        print(f"  - Device: {settings['device']}")
        print(f"  - Memory Limit: {settings['memory_limit_mb']} MB")
        print(f"  - Batch Size: {settings['batch_size']}")
        print(f"  - Workers: {settings['num_workers']}")
        print(f"  - Recommended Model: {settings['model_recommendation']}")
        print(f"  - Processing Mode: {settings['processing_mode']}")
        print("="*60)


def apply_global_optimizations():
    """Apply global optimizations for maximum performance"""
    optimizer = ResourceOptimizer(target_utilization=0.9)
    
    # Apply optimizations
    optimizer.set_environment_variables()
    optimizer.configure_torch_optimizations()
    optimizer.print_optimization_summary()
    
    return optimizer.get_optimal_settings()


if __name__ == "__main__":
    # Test the resource optimizer
    optimizer = ResourceOptimizer()
    settings = optimizer.get_optimal_settings()
    optimizer.print_optimization_summary()
