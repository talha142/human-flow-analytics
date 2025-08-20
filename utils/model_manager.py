#!/usr/bin/env python3
"""
Model Management Utility for Human Flow Tracking
Helps download, manage, and validate YOLO models
"""

import os
import sys
from pathlib import Path
import argparse
from ultralytics import YOLO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from backend.video_processor import VideoProcessor
except ImportError as e:
    print(f"Error importing VideoProcessor: {e}")
    sys.exit(1)


def download_model(model_name, target_dir="models"):
    """Download a specific YOLO model"""
    if model_name not in VideoProcessor.SUPPORTED_MODELS:
        print(f"❌ Unsupported model: {model_name}")
        print("Available models:")
        VideoProcessor.list_available_models()
        return False
    
    model_info = VideoProcessor.SUPPORTED_MODELS[model_name]
    model_file = model_info["path"]
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    target_file = target_path / model_file
    
    # Check if already exists
    if target_file.exists():
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {target_file} ({size_mb:.1f} MB)")
        return True
    
    try:
        print(f"📥 Downloading {model_name} ({model_info['size']})...")
        
        # Change to target directory to ensure download goes there
        original_cwd = os.getcwd()
        os.chdir(target_path)
        
        try:
            # Download using ultralytics
            model = YOLO(model_file)
            
            # Verify the file was created in the target directory
            if (target_path / model_file).exists():
                size_mb = (target_path / model_file).stat().st_size / (1024 * 1024)
                print(f"✅ Download complete: {size_mb:.1f} MB")
                print(f"✅ Model saved to: {target_file}")
                return True
            else:
                print(f"❌ Download failed: {target_file} not found")
                return False
                
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            
        # Also check if model was downloaded to original directory and move it
        original_path = Path(original_cwd) / model_file
        if original_path.exists() and original_path != target_file:
            original_path.rename(target_file)
            print(f"✅ Moved model from {original_path} to: {target_file}")
            size_mb = target_file.stat().st_size / (1024 * 1024)
            print(f"✅ Download complete: {size_mb:.1f} MB")
            return True
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def validate_model(model_path):
    """Validate a YOLO model file"""
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print(f"🔍 Validating model: {model_path}")
        
        # Try to load the model
        model = YOLO(str(model_file))
        
        # Get model info
        size_mb = model_file.stat().st_size / (1024 * 1024)
        
        print(f"✅ Model validation successful:")
        print(f"   File: {model_file}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Type: {type(model.model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False


def list_local_models(search_dir="models"):
    """List locally available models"""
    search_path = Path(search_dir)
    
    if not search_path.exists():
        print(f"📁 Model directory not found: {search_dir}")
        return []
    
    model_files = list(search_path.glob("*.pt"))
    
    if not model_files:
        print(f"📁 No model files found in: {search_dir}")
        return []
    
    print(f"\n📁 Local Models in {search_dir}:")
    print("-" * 60)
    
    local_models = []
    for model_file in sorted(model_files):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        model_name = model_file.stem
        
        # Check if it's a known model
        is_supported = model_name in VideoProcessor.SUPPORTED_MODELS
        status = "✅ Supported" if is_supported else "⚠️  Unknown"
        
        print(f"{model_name:15} | {size_mb:8.1f} MB | {status}")
        local_models.append(str(model_file))
    
    print("-" * 60)
    return local_models


def cleanup_models(search_dir="models", keep_latest=True):
    """Clean up old or duplicate model files"""
    search_path = Path(search_dir)
    
    if not search_path.exists():
        print(f"📁 Model directory not found: {search_dir}")
        return
    
    model_files = list(search_path.glob("*.pt"))
    
    if not model_files:
        print(f"📁 No model files to clean up in: {search_dir}")
        return
    
    print(f"🧹 Cleaning up models in {search_dir}...")
    
    # Group by model type
    model_groups = {}
    for model_file in model_files:
        base_name = model_file.stem.split('_')[0]  # Remove version suffixes
        if base_name not in model_groups:
            model_groups[base_name] = []
        model_groups[base_name].append(model_file)
    
    removed_count = 0
    for base_name, files in model_groups.items():
        if len(files) > 1:
            # Sort by modification time, keep newest
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if keep_latest:
                files_to_remove = files[1:]  # Remove all but the newest
                print(f"Keeping latest: {files[0].name}")
            else:
                files_to_remove = files
            
            for file_to_remove in files_to_remove:
                try:
                    file_to_remove.unlink()
                    print(f"🗑️  Removed: {file_to_remove.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {file_to_remove.name}: {e}")
    
    print(f"✅ Cleanup complete. Removed {removed_count} files.")


def move_misplaced_models(source_dir=".", target_dir="models"):
    """Move any YOLO model files from source to target directory"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # Look for YOLO model files in source directory
    yolo_patterns = ["yolo*.pt", "yolov*.pt"]
    moved_count = 0
    
    for pattern in yolo_patterns:
        for model_file in source_path.glob(pattern):
            # Skip if it's already in the target directory
            if model_file.parent == target_path:
                continue
                
            target_file = target_path / model_file.name
            
            # Skip if target already exists and is newer/larger
            if target_file.exists():
                source_size = model_file.stat().st_size
                target_size = target_file.stat().st_size
                if target_size >= source_size:
                    print(f"🗑️  Removing duplicate: {model_file}")
                    model_file.unlink()
                    continue
            
            try:
                model_file.rename(target_file)
                print(f"📦 Moved model: {model_file.name} -> {target_dir}/")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to move {model_file.name}: {e}")
    
    if moved_count > 0:
        print(f"✅ Moved {moved_count} model files to {target_dir}/")
    else:
        print("ℹ️  No misplaced model files found")


def install_custom_model(model_path, target_dir="models/custom", model_name=None):
    """Install a custom model file to the models directory"""
    source_path = Path(model_path)
    
    if not source_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not source_path.suffix.lower() == '.pt':
        print(f"❌ Invalid model file format. Expected .pt, got {source_path.suffix}")
        return False
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Determine target filename
    if model_name:
        if not model_name.endswith('.pt'):
            model_name += '.pt'
        target_file = target_path / model_name
    else:
        target_file = target_path / source_path.name
    
    try:
        # Copy the model file
        import shutil
        shutil.copy2(source_path, target_file)
        
        # Get file size
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"✅ Custom model installed: {target_file}")
        print(f"📊 Size: {size_mb:.1f} MB")
        
        # Try to validate the model
        try:
            from ultralytics import YOLO
            model = YOLO(str(target_file))
            print(f"✅ Model validation passed")
            
            # Show model info
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                class_names = list(model.model.names.values())
                print(f"🏷️ Classes detected: {len(class_names)}")
                if len(class_names) <= 20:
                    print(f"   Classes: {', '.join(class_names)}")
                    
        except Exception as e:
            print(f"⚠️ Model validation failed: {e}")
            print("   Model file copied but validation failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install custom model: {e}")
        return False


def list_custom_models(custom_dir="models/custom"):
    """List installed custom models"""
    custom_path = Path(custom_dir)
    
    if not custom_path.exists():
        print(f"📁 Custom models directory not found: {custom_dir}")
        return []
    
    custom_models = list(custom_path.glob("*.pt"))
    
    if not custom_models:
        print(f"📁 No custom models found in: {custom_dir}")
        return []
    
    print(f"\n📁 Custom Models in {custom_dir}:")
    print("-" * 60)
    
    for model_file in sorted(custom_models):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        
        # Try to get model info
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_file))
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                class_count = len(model.model.names)
                status = f"✅ {class_count} classes"
            else:
                status = "✅ Valid"
        except Exception:
            status = "⚠️ Unknown"
        
        print(f"{model_file.name:25} | {size_mb:8.1f} MB | {status}")
    
    print("-" * 60)
    return [str(f) for f in custom_models]


def main():
    parser = argparse.ArgumentParser(description="YOLO Model Manager")
    parser.add_argument("command", choices=["list", "download", "validate", "cleanup", "list-local", "move", "install-custom", "list-custom"],
                       help="Command to execute")
    parser.add_argument("--model", "-m", help="Model name (for download/validate)")
    parser.add_argument("--path", "-p", help="Model path (for validate/install-custom)")
    parser.add_argument("--dir", "-d", default="models", help="Model directory")
    parser.add_argument("--source", "-s", default=".", help="Source directory (for move command)")
    parser.add_argument("--name", "-n", help="Custom name for installed model")
    parser.add_argument("--all", "-a", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("🔦 Available YOLO Models for Download:")
        VideoProcessor.list_available_models()
        
    elif args.command == "list-local":
        list_local_models(args.dir)
        
    elif args.command == "download":
        if args.all:
            print("📥 Downloading all supported models...")
            success_count = 0
            for model_name in VideoProcessor.SUPPORTED_MODELS.keys():
                if download_model(model_name, args.dir):
                    success_count += 1
            print(f"\n✅ Downloaded {success_count}/{len(VideoProcessor.SUPPORTED_MODELS)} models")
        elif args.model:
            download_model(args.model, args.dir)
        else:
            print("❌ Please specify --model or --all")
            
    elif args.command == "validate":
        if args.path:
            validate_model(args.path)
        elif args.model:
            model_path = Path(args.dir) / f"{args.model}.pt"
            validate_model(model_path)
        else:
            print("❌ Please specify --model or --path")
            
    elif args.command == "cleanup":
        cleanup_models(args.dir)
        
    elif args.command == "move":
        move_misplaced_models(args.source, args.dir)
        
    elif args.command == "install-custom":
        if args.path:
            install_custom_model(args.path, f"{args.dir}/custom", args.name)
        else:
            print("❌ Please specify --path to the custom model file")
            
    elif args.command == "list-custom":
        list_custom_models(f"{args.dir}/custom")


if __name__ == "__main__":
    main()