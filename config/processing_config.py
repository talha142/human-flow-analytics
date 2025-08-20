"""
Processing Configuration for Human Flow Tracking
Centralized configuration for memory optimization and model settings
"""

import os
from pathlib import Path


class ProcessingConfig:
    """Configuration class for video processing settings"""
    
    # Memory Management
    DEFAULT_MEMORY_LIMIT_MB = 1024
    MIN_MEMORY_LIMIT_MB = 256
    MAX_MEMORY_LIMIT_MB = 8192
    
    # Model Settings
    DEFAULT_MODEL = "yolo11n"
    MODEL_CACHE_DIR = "models"
    
    # Processing Settings
    DEFAULT_BATCH_SIZES = {
        "fast": 8,
        "balanced": 4,
        "accurate": 2,
    }
    
    # Device Settings
    AUTO_DEVICE_SELECTION = True
    PREFER_GPU = True
    
    # Memory Optimization Thresholds
    MEMORY_WARNING_THRESHOLD = 0.8  # Warn when 80% of limit is reached
    MEMORY_CLEANUP_THRESHOLD = 0.9  # Force cleanup when 90% is reached
    
    @classmethod
    def get_optimal_batch_size(cls, mode, available_memory_mb):
        """Calculate optimal batch size based on available memory"""
        base_batch_size = cls.DEFAULT_BATCH_SIZES.get(mode, 4)
        
        # Adjust based on available memory
        if available_memory_mb < 512:
            return max(1, base_batch_size // 4)
        elif available_memory_mb < 1024:
            return max(1, base_batch_size // 2)
        elif available_memory_mb > 2048:
            return min(16, base_batch_size * 2)
        else:
            return base_batch_size
    
    @classmethod
    def get_memory_limit(cls, user_limit=None):
        """Get appropriate memory limit with validation"""
        if user_limit is None:
            return cls.DEFAULT_MEMORY_LIMIT_MB
        
        return max(
            cls.MIN_MEMORY_LIMIT_MB,
            min(cls.MAX_MEMORY_LIMIT_MB, user_limit)
        )
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get full path for a model file"""
        model_dir = Path(cls.MODEL_CACHE_DIR)
        model_dir.mkdir(exist_ok=True)
        
        # Handle different model naming conventions
        if not model_name.endswith('.pt'):
            model_name += '.pt'
        
        return model_dir / model_name
    
    @classmethod
    def ensure_models_directory(cls):
        """Ensure the models directory exists"""
        model_dir = Path(cls.MODEL_CACHE_DIR)
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    @classmethod
    def estimate_video_memory_requirements(cls, width, height, fps, duration_seconds):
        """Estimate memory requirements for video processing"""
        # Rough estimation based on frame size and processing overhead
        frame_size_mb = (width * height * 3 * 4) / (1024 * 1024)  # RGB, float32
        frames_per_second = fps
        total_frames = duration_seconds * frames_per_second
        
        # Processing overhead (model inference, tracking, etc.)
        processing_overhead = 2.0
        
        # Estimate peak memory usage (batch processing)
        peak_memory_mb = frame_size_mb * 8 * processing_overhead  # 8 frame batch
        
        return {
            'frame_size_mb': frame_size_mb,
            'total_frames': total_frames,
            'peak_memory_mb': peak_memory_mb,
            'recommended_limit_mb': max(cls.MIN_MEMORY_LIMIT_MB, int(peak_memory_mb * 1.5))
        }


# Environment-specific configurations
def get_environment_config():
    """Get configuration based on current environment"""
    config = {
        'memory_limit_mb': ProcessingConfig.DEFAULT_MEMORY_LIMIT_MB,
        'model_cache_dir': ProcessingConfig.MODEL_CACHE_DIR,
        'default_model': ProcessingConfig.DEFAULT_MODEL,
    }
    
    # Check for environment variables
    if 'FLOW_MEMORY_LIMIT' in os.environ:
        try:
            config['memory_limit_mb'] = int(os.environ['FLOW_MEMORY_LIMIT'])
        except ValueError:
            pass
    
    if 'FLOW_MODEL_DIR' in os.environ:
        config['model_cache_dir'] = os.environ['FLOW_MODEL_DIR']
    
    if 'FLOW_DEFAULT_MODEL' in os.environ:
        config['default_model'] = os.environ['FLOW_DEFAULT_MODEL']
    
    return config


# Performance profiles for different hardware configurations
PERFORMANCE_PROFILES = {
    'low_end': {
        'memory_limit_mb': 512,
        'default_model': 'yolo11n',
        'batch_sizes': {'fast': 2, 'balanced': 1, 'accurate': 1},
        'resize_width': 480,
    },
    'mid_range': {
        'memory_limit_mb': 1024,
        'default_model': 'yolo11s',
        'batch_sizes': {'fast': 4, 'balanced': 2, 'accurate': 1},
        'resize_width': 640,
    },
    'high_end': {
        'memory_limit_mb': 2048,
        'default_model': 'yolo11m',
        'batch_sizes': {'fast': 8, 'balanced': 4, 'accurate': 2},
        'resize_width': None,  # Original size
    },
    'server': {
        'memory_limit_mb': 4096,
        'default_model': 'yolo11l',
        'batch_sizes': {'fast': 16, 'balanced': 8, 'accurate': 4},
        'resize_width': None,  # Original size
    }
}


def detect_hardware_profile():
    """Detect appropriate hardware profile based on system specs"""
    try:
        import psutil
        import torch
        
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check GPU availability
        has_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
        
        # Determine profile
        if memory_gb >= 16 and has_gpu:
            return 'high_end'
        elif memory_gb >= 8 and has_gpu:
            return 'mid_range'
        elif memory_gb >= 8:
            return 'mid_range'
        else:
            return 'low_end'
            
    except ImportError:
        # Fallback if psutil not available
        return 'mid_range'


def get_recommended_settings():
    """Get recommended settings based on detected hardware"""
    profile_name = detect_hardware_profile()
    profile = PERFORMANCE_PROFILES[profile_name]
    
    print(f"[INFO] Detected hardware profile: {profile_name}")
    print(f"[INFO] Recommended settings: {profile}")
    
    return profile_name, profile