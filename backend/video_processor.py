import cv2
import pandas as pd
import time
from ultralytics import YOLO
import math
import torch
from typing import Optional
from pathlib import Path
import sys
import os
import psutil
from config.processing_config import ProcessingConfig, get_recommended_settings
from utils.resource_optimizer import ResourceOptimizer

# Add config and utils paths
config_path = os.path.join(os.path.dirname(__file__), '..', 'config')
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if config_path not in sys.path:
    sys.path.append(config_path)
if utils_path not in sys.path:
    sys.path.append(utils_path)




class VideoProcessor:
    """
    High-throughput video analytics with YOLOv8 built-in tracking.
    Modes:
      - "fast"      : max FPS, skip more frames
      - "balanced"  : middle ground  
      - "accurate"  : process more frames, better accuracy

    Backward-compat aliases supported:
      "Fast" -> fast, "Normal" -> balanced, "High Accuracy" -> accurate
    """

    MODES = {
        "fast": {
            "skip_interval": 5,        # process every 5th frame
            "conf": 0.55,
            "draw": True,
            "batch_size": 8,           # frames to process in batch
        },
        "balanced": {
            "skip_interval": 2,        # process every 2nd frame
            "conf": 0.5,
            "draw": True,
            "batch_size": 4,           # frames to process in batch
        },
        "accurate": {
            "skip_interval": 1,        # process every frame
            "conf": 0.45,
            "draw": True,
            "batch_size": 2,           # frames to process in batch
        },
    }

    # Supported model configurations
    SUPPORTED_MODELS = {
        "yolo11n": {"path": "yolo11n.pt", "size": "nano", "speed": "fastest"},
        "yolo11s": {"path": "yolo11s.pt", "size": "small", "speed": "fast"},
        "yolo11m": {"path": "yolo11m.pt", "size": "medium", "speed": "balanced"},
        "yolo11l": {"path": "yolo11l.pt", "size": "large", "speed": "accurate"},
        "yolo11x": {"path": "yolo11x.pt", "size": "extra-large", "speed": "most accurate"},
        "yolov8n": {"path": "yolov8n.pt", "size": "nano", "speed": "fastest"},
        "yolov8s": {"path": "yolov8s.pt", "size": "small", "speed": "fast"},
        "yolov8m": {"path": "yolov8m.pt", "size": "medium", "speed": "balanced"},
        "yolov8l": {"path": "yolov8l.pt", "size": "large", "speed": "accurate"},
        "yolov8x": {"path": "yolov8x.pt", "size": "extra-large", "speed": "most accurate"},
    }

    LEGACY_ALIASES = {
        "Fast": "fast",
        "Normal": "balanced", 
        "High Accuracy": "accurate"
    }

    def __init__(
        self,
        model_path=None,
        model_name="yolo11n",
        resize_width=None,
        device=None,
        default_mode="balanced",
        max_memory_mb=None,
        optimize_resources=True,
    ):
        # Resource optimization
        self.optimal_settings = None
        if optimize_resources and ResourceOptimizer:
            try:
                optimizer = ResourceOptimizer(target_utilization=0.9)
                self.optimal_settings = optimizer.get_optimal_settings()
                optimizer.configure_torch_optimizations()
                print("[INFO] Applied 90% resource optimization")
            except Exception as e:
                print(f"[WARN] Resource optimization failed: {e}")
        
        # Device selection with optimization
        if device:
            self.device = device
        elif self.optimal_settings:
            self.device = self.optimal_settings['device']
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[INFO] Using device: {self.device}")

        # Memory management with optimization
        if max_memory_mb is not None:
            self.max_memory_mb = max_memory_mb
        elif self.optimal_settings:
            self.max_memory_mb = self.optimal_settings['memory_limit_mb']
        else:
            self.max_memory_mb = ProcessingConfig.get_memory_limit(max_memory_mb)
        
        self.frame_buffer = []
        self.buffer_size = 0
        
        print(f"[INFO] Using memory limit: {self.max_memory_mb}MB")
        
        # Get recommended settings if available
        try:
            profile_name, profile_settings = get_recommended_settings()
            if max_memory_mb is None and not self.optimal_settings:  # Only use recommended if not explicitly set
                self.max_memory_mb = profile_settings['memory_limit_mb']
                print(f"[INFO] Using recommended memory limit: {self.max_memory_mb}MB")
        except Exception:
            pass  # Use default settings
        
        # Resolve model path with flexibility
        resolved_model_path = self._resolve_model_path(model_path, model_name)
        
        # Load YOLO model
        try:
            self.model = YOLO(resolved_model_path)
            self.model.to(self.device)
            print(f"[INFO] Loaded YOLO model from: {resolved_model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {resolved_model_path}: {e}")

        # Try model optimizations
        try:
            self.model.fuse()
        except Exception:
            print("[WARN] Model fusion failed, continuing without fusion")

        # Enable FP16 on CUDA
        self.fp16 = False
        if "cuda" in self.device:
            try:
                self.model.half()
                self.fp16 = True
                print("[INFO] Enabled FP16 inference")
            except Exception:
                print("[WARN] FP16 not available, using FP32")

        self.resize_width = resize_width
        self.global_id_map = {}
        self.next_global_id = 1
        
        # Initialize custom setting flags
        self._custom_conf_set = False
        self._custom_draw_set = False

        # Set default mode
        self.set_mode(default_mode)

    def _resolve_model_path(self, model_path, model_name):
        """Resolve model path with flexible fallback options"""
        # Ensure models directory exists
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # If explicit path provided, try it first (for custom models)
        if model_path:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                print(f"[INFO] Using custom model: {model_path}")
                return str(model_path_obj.absolute())
            else:
                print(f"[WARN] Specified model path not found: {model_path}")
                # For custom models, don't fallback - raise error
                if not model_name or model_name == "custom":
                    raise FileNotFoundError(f"Custom model file not found: {model_path}")
        
        # Handle custom model name without path
        if model_name == "custom":
            raise ValueError("Custom model selected but no model path provided")
        
        # Try model name in supported models
        if model_name in self.SUPPORTED_MODELS:
            model_file = self.SUPPORTED_MODELS[model_name]["path"]
            
            # Search in common locations
            search_paths = [
                models_dir / model_file,
                f"models/{model_file}",
                f"../models/{model_file}",
                f"./models/{model_file}",
            ]
            
            for path in search_paths:
                if Path(path).exists():
                    print(f"[INFO] Found model at: {path}")
                    return str(path)
            
            # If not found locally, download to models directory
            target_path = models_dir / model_file
            print(f"[INFO] Model not found locally, will download to: {target_path}")
            
            try:
                # Download using ultralytics, but ensure it goes to models folder
                temp_model = YOLO(model_file)  # This downloads to current dir
                
                # Move to models directory if it was downloaded to current dir
                current_dir_path = Path(model_file)
                if current_dir_path.exists() and current_dir_path != target_path:
                    current_dir_path.rename(target_path)
                    print(f"[INFO] Moved model to: {target_path}")
                
                return str(target_path)
                
            except Exception as e:
                print(f"[WARN] Failed to download model: {e}")
                # Return the target path anyway, let YOLO handle the error
                return str(target_path)
        
        # Fallback to default
        default_model_file = "yolo11n.pt"
        default_path = models_dir / default_model_file
        print(f"[WARN] Unknown model '{model_name}', using default: {default_path}")
        
        # Ensure default model exists
        if not default_path.exists():
            try:
                temp_model = YOLO(default_model_file)
                current_dir_path = Path(default_model_file)
                if current_dir_path.exists():
                    current_dir_path.rename(default_path)
                    print(f"[INFO] Downloaded default model to: {default_path}")
            except Exception as e:
                print(f"[WARN] Failed to download default model: {e}")
        
        return str(default_path)

    def _estimate_memory_usage(self, frame_shape, batch_size=1):
        """Estimate memory usage for frame processing"""
        height, width, channels = frame_shape
        # Rough estimate: frame size * batch size * processing overhead
        frame_mb = (height * width * channels * 4) / (1024 * 1024)  # 4 bytes per pixel (float32)
        return frame_mb * batch_size * 2  # 2x overhead for processing

    def _clear_memory_cache(self):
        """Clear memory caches to free up RAM"""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Clear frame buffer
        self.frame_buffer.clear()
        self.buffer_size = 0
        
        # Additional memory optimizations
        if torch.cuda.is_available():
            try:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                # Synchronize to ensure operations complete
                torch.cuda.synchronize()
            except Exception:
                pass
    
    def _monitor_resource_usage(self):
        """Monitor and adjust resource usage dynamically"""
        try:
            if ResourceOptimizer:
                
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                
                # If memory usage is too high, reduce batch size temporarily
                if memory_percent > 85 and self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    print(f"[INFO] High memory usage detected, reducing batch size to {self.batch_size}")
                
                # Check GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    if gpu_memory_percent > 85:
                        torch.cuda.empty_cache()
                        print("[INFO] High GPU memory usage detected, clearing cache")
                        
        except Exception as e:
            print(f"[WARN] Resource monitoring failed: {e}")

    def _should_process_batch(self, current_frame_size_mb):
        """Determine if we should process current batch based on memory constraints"""
        estimated_usage = self.buffer_size + current_frame_size_mb
        return estimated_usage >= self.max_memory_mb or len(self.frame_buffer) >= self.batch_size

    def _process_frame_batch(self, frames, metadata, unique_ids_per_minute, fps):
        """Process a batch of frames for memory efficiency"""
        if not frames:
            return []
            
        processed_frames = []
        
        try:
            # Process each frame in the batch
            for i, frame in enumerate(frames):
                frame_data = metadata[i]
                frame_idx = frame_data['frame_idx']
                current_minute = frame_data['current_minute']
                
                # Use YOLOv8's built-in tracking
                results = self.model.track(frame, conf=self.conf, persist=True, verbose=False, classes=[0])
                
                if current_minute not in unique_ids_per_minute:
                    unique_ids_per_minute[current_minute] = set()

                # Process results
                if len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    
                    if boxes is not None and len(boxes) > 0:
                        ids = getattr(boxes, "id", None)
                        
                        for j, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            if cls_id == 0:  # Only person class
                                if ids is not None and j < len(ids):
                                    local_id = int(ids[j])
                                    
                                    # Map to global unique ID
                                    if local_id not in self.global_id_map:
                                        self.global_id_map[local_id] = self.next_global_id
                                        self.next_global_id += 1
                                    
                                    global_id = self.global_id_map[local_id]
                                    unique_ids_per_minute[current_minute].add(global_id)
                                    
                                    # Draw if enabled
                                    if self.draw:
                                        self._draw_track(frame, box, global_id)
                                else:
                                    # Handle case where no ID is assigned
                                    if self.draw:
                                        self._draw_track(frame, box, "?")
                
                # Store the processed frame with drawings
                processed_frames.append((frame_data['frame_idx'], frame))
                                        
        except Exception as e:
            print(f"[WARN] Batch processing error: {e}")
            # Continue processing
        
        return processed_frames

    @classmethod
    def list_available_models(cls):
        """List all supported model configurations"""
        print("\nAvailable YOLO Models:")
        print("-" * 50)
        for name, info in cls.SUPPORTED_MODELS.items():
            print(f"{name:10} | {info['size']:12} | {info['speed']}")
        print("-" * 50)

    def set_mode(self, mode: str):
        """Set processing mode"""
        # Handle legacy aliases
        mode = self.LEGACY_ALIASES.get(mode, mode)
        if mode not in self.MODES:
            print(f"[WARN] Unknown mode '{mode}', falling back to 'balanced'.")
            mode = "balanced"
        
        self.mode = mode
        cfg = self.MODES[mode]
        
        # Set instance attributes from mode config
        self.skip_interval = cfg["skip_interval"]
        self.conf = cfg["conf"]
        self.draw = cfg["draw"]
        
        print(f"[INFO] Mode set to '{self.mode}': {cfg}")
        
        # Set batch size with optimization
        if self.optimal_settings:
            self.batch_size = self.optimal_settings['batch_size']
            print(f"[INFO] Using optimized batch size: {self.batch_size}")
        else:
            self.batch_size = cfg.get("batch_size", 1)

    def set_draw_enabled(self, enabled: bool):
        """Enable or disable drawing of bounding boxes"""
        self.draw = enabled
        self._custom_draw_set = True  # Flag to indicate custom setting
        print(f"[INFO] Drawing {'enabled' if enabled else 'disabled'}")

    def set_confidence(self, confidence: float):
        """Set detection confidence threshold"""
        if 0.0 <= confidence <= 1.0:
            self.conf = confidence
            self._custom_conf_set = True  # Flag to indicate custom setting
            print(f"[INFO] Confidence threshold set to {confidence}")
        else:
            print(f"[WARN] Invalid confidence {confidence}, must be between 0.0 and 1.0")

    def _draw_track(self, frame, box, global_id):
        """Draw bounding box and ID on frame"""
        try:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy[:4])
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width-1, x2))
            y2 = max(0, min(height-1, y2))
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text with background
            label = f"{global_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background rectangle
            text_y = max(text_height + 5, y1 - 5)
            cv2.rectangle(frame, (x1, text_y - text_height - 5), 
                         (x1 + text_width, text_y + 5), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, text_y), font, font_scale, (0, 0, 0), font_thickness)
            
        except Exception as e:
            print(f"[WARN] Failed to draw track {global_id}: {e}")

    def process_video(
        self,
        input_path,
        output_video_path,
        output_csv_path,
        mode: Optional[str] = None,
        speed_mode: Optional[str] = None,  # For backward compatibility
        progress_callback=None,
    ):
        """Process video with person tracking and counting"""
        
        # Handle backward compatibility
        if speed_mode and not mode:
            mode = speed_mode
        
        # Store current custom settings before potentially overwriting with mode
        custom_conf = getattr(self, 'conf', None)
        custom_draw = getattr(self, 'draw', None)
        
        # Set mode if provided
        if mode:
            self.set_mode(mode)
            
            # Restore custom settings if they were set
            if custom_conf is not None and hasattr(self, '_custom_conf_set'):
                self.conf = custom_conf
                print(f"[INFO] Restored custom confidence: {custom_conf}")
            if custom_draw is not None and hasattr(self, '_custom_draw_set'):
                self.draw = custom_draw
                print(f"[INFO] Restored custom draw setting: {custom_draw}")

        start_time = time.time()
        
        # Open video with fallback backends
        cap = None
        backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        
        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(input_path, backend)
                if cap.isOpened():
                    print(f"[INFO] Video opened with backend: {backend}")
                    break
                else:
                    if cap:
                        cap.release()
                    cap = None
            except Exception as e:
                print(f"[WARN] Backend {backend} failed: {e}")
                if cap:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Could not open video file with any backend: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_width <= 0 or original_height <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions")

        # Calculate output dimensions
        if self.resize_width and self.resize_width != original_width:
            ratio = self.resize_width / original_width
            output_width = self.resize_width
            output_height = max(1, int(original_height * ratio))
        else:
            output_width = original_width
            output_height = original_height

        print(f"[INFO] Video: {original_width}x{original_height} -> {output_width}x{output_height}, {fps} FPS, {total_frames} frames")

        # Initialize video writer with codec fallback
        out = None
        codecs_to_try = ['mp4v', 'XVID', 'MJPG', 'X264']
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
                if out.isOpened():
                    print(f"[INFO] Video writer initialized with codec: {codec}")
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                print(f"[WARN] Codec {codec} failed: {e}")
                if out:
                    out.release()
                    out = None
        
        if out is None:
            cap.release()
            raise RuntimeError(f"Could not initialize video writer with any codec: {output_video_path}")

        # Initialize tracking variables
        unique_ids_per_minute = {}
        frame_idx = 0
        
        # Reset global ID mapping for new video
        self.global_id_map.clear()
        self.next_global_id = 1
        
        # Initialize tracking variables
        self._clear_memory_cache()

        # Print current settings
        print(f"[INFO] Processing with: mode={self.mode}, conf={self.conf}, draw={self.draw}, skip_interval={self.skip_interval}")
        print(f"[INFO] Memory limit: {self.max_memory_mb}MB")
        print(f"[INFO] Video info: {total_frames} frames, {fps} FPS, estimated duration: {total_frames/fps:.1f}s")
        
        # Initialize progress tracking
        last_update = 0
        last_percent = -1
        
        # Initial progress callback
        if progress_callback:
            try:
                progress_callback(0)
            except Exception as e:
                print(f"[WARN] Initial progress callback failed: {e}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Resize frame if needed
                if self.resize_width and self.resize_width != original_width:
                    frame = cv2.resize(frame, (output_width, output_height))

                # Process frames according to skip interval
                if frame_idx % self.skip_interval == 0:
                    try:
                        # Use YOLOv8's built-in tracking
                        results = self.model.track(frame, conf=self.conf, persist=True, verbose=False, classes=[0])
                        
                        # Calculate current minute
                        current_minute = math.floor(frame_idx / (fps * 60))
                        
                        if current_minute not in unique_ids_per_minute:
                            unique_ids_per_minute[current_minute] = set()

                        # Process results and draw on the frame
                        if len(results) > 0:
                            r = results[0]
                            boxes = getattr(r, "boxes", None)
                            
                            if boxes is not None and len(boxes) > 0:
                                ids = getattr(boxes, "id", None)
                                
                                for i, box in enumerate(boxes):
                                    cls_id = int(box.cls[0])
                                    if cls_id == 0:  # Only person class
                                        if ids is not None and i < len(ids):
                                            local_id = int(ids[i])
                                            
                                            # Map to global unique ID
                                            if local_id not in self.global_id_map:
                                                self.global_id_map[local_id] = self.next_global_id
                                                self.next_global_id += 1
                                            
                                            global_id = self.global_id_map[local_id]
                                            unique_ids_per_minute[current_minute].add(global_id)
                                            
                                            # Draw if enabled (directly on the frame that will be written)
                                            if self.draw:
                                                self._draw_track(frame, box, global_id)
                                        else:
                                            # Handle case where no ID is assigned
                                            if self.draw:
                                                self._draw_track(frame, box, "?")
                                            
                    except Exception as e:
                        print(f"[WARN] Processing failed on frame {frame_idx}: {e}")
                        # Continue with unprocessed frame

                # Write frame to output (now with drawings if enabled)
                out.write(frame)

                # Periodic memory cleanup and optimization
                if frame_idx % 100 == 0:
                    self._clear_memory_cache()
                    
                    # Monitor and adjust resource usage
                    if frame_idx % 200 == 0:
                        self._monitor_resource_usage()
                    
                    # Apply additional optimizations every 500 frames
                    if self.optimal_settings and frame_idx % 500 == 0:
                        try:
                            # Optimize CUDA cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        except Exception:
                            pass

                # Update progress - simplified to avoid UI issues
                if progress_callback and total_frames > 0:
                    now = time.time()
                    current_percent = min(100, int((frame_idx / total_frames) * 100))
                    
                    # Update progress every 1 second or when percentage changes significantly
                    time_elapsed = now - last_update
                    percent_changed = abs(current_percent - last_percent) >= 1
                    
                    should_update = (
                        time_elapsed > 1.0 or  # Every 1 second
                        (percent_changed and time_elapsed > 0.2) or  # When percentage changes
                        frame_idx == 1 or  # First frame
                        current_percent >= 100  # Completion
                    )
                    
                    if should_update:
                        print(f"[INFO] Processing frames... {current_percent}% ({frame_idx}/{total_frames})")
                        
                        try:
                            progress_callback(current_percent)
                            last_percent = current_percent
                            last_update = now
                        except Exception as e:
                            print(f"[WARN] Progress callback failed: {e}")
                            # Continue processing even if progress callback fails


        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            raise
        finally:
            cap.release()
            out.release()
            
            # Final progress update
            if progress_callback:
                try:
                    progress_callback(100)
                except Exception as e:
                    print(f"[WARN] Final progress callback failed: {e}")

        # Generate CSV output
        try:
            summary_data = []
            for minute in sorted(unique_ids_per_minute.keys()):
                count = len(unique_ids_per_minute[minute])
                summary_data.append([minute, count])

            df = pd.DataFrame(summary_data, columns=["Minute", "Unique_People_Count"])
            df.to_csv(output_csv_path, index=False)
            
            total_unique = len(set().union(*unique_ids_per_minute.values()) if unique_ids_per_minute else set())
            print(f"[INFO] Processed {frame_idx} frames, found {total_unique} unique people across {len(unique_ids_per_minute)} minutes")
            
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
            # Create empty CSV as fallback
            pd.DataFrame(columns=["Minute", "Unique_People_Count"]).to_csv(output_csv_path, index=False)

        processing_time = round(time.time() - start_time, 2)
        return output_video_path, output_csv_path, processing_time