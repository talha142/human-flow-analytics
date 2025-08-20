import torch
import os
import sys
import glob
from pathlib import Path
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore")
# Add frontend to path
frontend_path = os.path.join(os.path.dirname(__file__), 'frontend')
if frontend_path not in sys.path:
    sys.path.append(frontend_path)

try:
    from frontend.app import streamlit_main
except ImportError as e:
    print(f"Error importing streamlit app: {e}")
    sys.exit(1)


def print_system_info():
    """Print comprehensive system and environment information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Python info
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}") #type: ignore
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available")
    
    # MPS (Apple Silicon) support
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("MPS (Apple Silicon GPU) support enabled")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"System RAM: {memory.total / 1024**3:.1f} GB (Available: {memory.available / 1024**3:.1f} GB)")
    except ImportError:
        print("System memory info unavailable (install psutil for details)")
    
    print("=" * 60)


def list_available_models():
    """List available YOLO models"""
    try:
        from backend.video_processor import VideoProcessor
        VideoProcessor.list_available_models()
    except ImportError:
        print("Could not import VideoProcessor to list models")

def validate_environment():
    """Validate that all required components are available"""
    print("ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check required packages
    required_packages = [
        ('cv2', 'opencv-python'),
        ('pandas', 'pandas'),
        ('ultralytics', 'ultralytics'),
        ('deep_sort_realtime', 'deep-sort-realtime'),
        ('streamlit', 'streamlit'),
        ('matplotlib', 'matplotlib'),
        ('numpy', 'numpy'),
        ('torch', 'torch')
    ]
    
    print("Checking required packages...")
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            errors.append(f"Missing package: {package_name}")
            print(f"✗ {package_name} - MISSING")
    
    # Check directories first
    required_dirs = ["uploads", "outputs", "models"]
    print("\nChecking/creating directories...")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        try:
            dir_path.mkdir(exist_ok=True)
            print(f"✓ {dir_name}/")
        except Exception as e:
            errors.append(f"Cannot create directory {dir_name}: {e}")
            print(f"✗ {dir_name}/ - ERROR: {e}")
    
    # Move any misplaced model files
    print("\nChecking for misplaced model files...")
    try:
        
        # Look for YOLO models in current directory
        misplaced_models = []
        for pattern in ["yolo*.pt", "yolov*.pt"]:
            misplaced_models.extend(glob.glob(pattern))
        
        if misplaced_models:
            models_dir = Path("models")
            moved_count = 0
            for model_file in misplaced_models:
                source_path = Path(model_file)
                target_path = models_dir / source_path.name
                
                try:
                    if not target_path.exists():
                        source_path.rename(target_path)
                        print(f"📦 Moved {model_file} to models/")
                        moved_count += 1
                    else:
                        # Remove duplicate if target is larger/newer
                        if target_path.stat().st_size >= source_path.stat().st_size:
                            source_path.unlink()
                            print(f"🗑️  Removed duplicate {model_file}")
                except Exception as e:
                    print(f"⚠️  Could not move {model_file}: {e}")
            
            if moved_count > 0:
                print(f"✅ Moved {moved_count} model files to models/")
        else:
            print("✓ No misplaced model files found")
            
    except Exception as e:
        print(f"⚠️  Could not check for misplaced models: {e}")
    
    # Check model file
    model_path = Path("models/yolo11n.pt")
    print(f"\nChecking default model: {model_path}")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model file found ({size_mb:.1f} MB)")
    else:
        warnings.append(f"Default model not found: {model_path}")
        print(f"⚠️  Default model missing: {model_path}")
        print("   (Will be downloaded automatically on first use)")
        
    
    # Check write permissions
    test_files = [
        ("uploads", "test_upload.tmp"),
        ("outputs", "test_output.tmp")
    ]
    
    print("\nChecking write permissions...")
    for dir_name, test_file in test_files:
        test_path = Path(dir_name) / test_file
        try:
            test_path.write_text("test")
            test_path.unlink()  # Remove test file
            print(f"✓ {dir_name}/ writable")
        except Exception as e:
            warnings.append(f"Write permission issue in {dir_name}: {e}")
            print(f"⚠ {dir_name}/ - WARNING: {e}")
    
    # Check GPU compatibility
    print("\nGPU Compatibility:")
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            test_tensor = torch.randn(10, 10).cuda()
            torch.mm(test_tensor, test_tensor)
            print("✓ CUDA GPU functional")
        except Exception as e:
            warnings.append(f"CUDA GPU available but not functional: {e}")
            print(f"⚠ CUDA GPU issue: {e}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test MPS functionality
            test_tensor = torch.randn(10, 10).to('mps')
            torch.mm(test_tensor, test_tensor)
            print("✓ MPS (Apple GPU) functional")
        except Exception as e:
            warnings.append(f"MPS available but not functional: {e}")
            print(f"⚠ MPS issue: {e}")
    else:
        print("ℹ Using CPU (GPU acceleration not available)")
    
    print("=" * 60)
    
    # Summary
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  • {error}")
        print("\nPlease resolve these errors before running the application.")
        return False
    
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
        print("\nApplication can run but performance may be affected.")
    
    if not errors and not warnings:
        print("✅ All checks passed! Environment is ready.")
    
    return True


# def create_sample_config():
#     """Create a sample configuration file for reference"""
#     config_content = """# Human Flow Tracking Configuration
# # This file shows available configuration options

# # Processing modes
# MODES:
#   fast:
#     detection_interval: 8    # Process every 8th frame
#     confidence: 0.55        # Detection confidence threshold
#     max_age: 30            # Tracker max age
#     n_init: 2              # Frames to confirm track
  
#   balanced:
#     detection_interval: 5
#     confidence: 0.50
#     max_age: 50
#     n_init: 3
  
#   accurate:
#     detection_interval: 3
#     confidence: 0.45
#     max_age: 70
#     n_init: 3

# # Model settings
# MODEL_PATH: "models/yolo11n.pt"
# RESIZE_WIDTH: 640  # null for original size

# # Device settings (auto-detected by default)
# # DEVICE: "cuda:0"  # or "mps", "cpu", null for auto

# # Output settings
# OUTPUT_VIDEO_CODEC: "mp4v"
# DRAW_TRACKS: true
# DRAW_CONFIDENCE: false
# """
    
#     config_path = Path("config_sample.yaml")
#     if not config_path.exists():
#         try:
#             config_path.write_text(config_content)
#             print(f"✓ Created sample config: {config_path}")
#         except Exception as e:
#             print(f"⚠ Could not create sample config: {e}")


def main():
    """Main entry point"""
    print("🚀 Human Flow Tracking & Analytics")
    print("Starting application...\n")
    
    # Apply resource optimizations
    try:
        utils_path = os.path.join(os.path.dirname(__file__), 'utils')
        if utils_path not in sys.path:
            sys.path.append(utils_path)
        from resource_optimizer import apply_global_optimizations
        optimal_settings = apply_global_optimizations()
        print(f"✅ Applied 90% resource optimization")
    except Exception as e:
        print(f"⚠️  Resource optimization failed: {e}")
        print("Continuing with default settings...")
    
    # Print system information
    print_system_info()
    
    # List available models
    list_available_models()
    
    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Please fix the errors above.")
        sys.exit(1)
    
    # Create sample config for reference
    # create_sample_config()
    
    print("\n🌐 Starting Streamlit application...")
    print("📌 Access the app at: http://localhost:8501")
    print("📌 Use Ctrl+C to stop the application")
    print("\n" + "=" * 60)
    
    try:
        # Start the Streamlit app
        streamlit_main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()