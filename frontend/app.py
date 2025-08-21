# Author: Muneeb Ahmad | mpysolutions.com | fiverr.com/muneeb_ahmad_ch | github.com/Muneeb-Ahmad-Ch 
# © 2025 MPY Solutions. Developed by Muneeb Ahmad & Team. All rights reserved.
# Unauthorized use, distribution, or reproduction of this code is strictly prohibited and not permitted.
# The developer assumes no responsibility for any damages or losses that may result from the use of this code.
# Do not use this code for illegal or unethical activities.
# ==============================================================================

import sys
import os
from pathlib import Path
import datetime as dt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.resource_optimizer import ResourceOptimizer

# Add parent directory to sys.path to import backend modules
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    from backend.video_processor import VideoProcessor
except ImportError as e:
    st.error(f"Failed to import VideoProcessor: {e}")
    st.stop()


def get_available_models():
    """Get all available models including custom ones"""
    model_options = {
        "yolo11n": "YOLO11 Nano (fastest, smallest)",
        "yolo11s": "YOLO11 Small (fast)",
        "yolo11m": "YOLO11 Medium (balanced)",
        "yolo11l": "YOLO11 Large (accurate)",
        "yolov8n": "YOLOv8 Nano (fastest, smallest)",
        "yolov8s": "YOLOv8 Small (fast)",
        "yolov8m": "YOLOv8 Medium (balanced)",
    }
    
    # Automatically detect and add existing custom models
    custom_models_dir = Path("models/custom")
    custom_models = {}
    if custom_models_dir.exists():
        for model_file in custom_models_dir.glob("*.pt"):
            try:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                model_key = f"custom_{model_file.stem}"
                model_options[model_key] = f"📁 {model_file.name} ({size_mb:.1f}MB)"
                custom_models[model_key] = model_file
            except Exception as e:
                print(f"[WARN] Could not read custom model {model_file}: {e}")
    
    # Add upload option at the end
    model_options["custom_upload"] = "📁 Upload New Custom Model (.pt file)"
    
    return model_options, custom_models


def streamlit_main(uploaded_folder="uploads", output_folder="outputs"):
    """Main Streamlit application for human flow tracking"""
    
    # Page config
    st.set_page_config(
        page_title="Human Flow Tracking & Analytics", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("👥 Human Flow Tracking & Analytics")
    # --- Footer Section ---
    with st.expander("ℹ️ About / Legal Notice", expanded=False):
        st.markdown("""
        #### 📌 About This Application
        **Project:** Human Flow Tracking & Analytics  
        **Developed by:** Muneeb Ahmad & MPY Solutions Team  
        **Company:** [MPY Solutions](https://mpysolutions.com)  
        **Contact:** info@mpysolutions.com  

        ---
        © 2025 MPY Solutions. All rights reserved.  

        **Notice:**  
        This software and its source code are the intellectual property of MPY Solutions.  
        Unauthorized use, modification, distribution, or reverse engineering is strictly prohibited.  

        **Special License Exception:**  
        Sebastián is granted the right to use this software for **personal** and **company purposes** without restriction.  

        **Disclaimer:**  
        This software is provided *“AS IS”* without any warranties.  
        MPY Solutions and its developers assume no liability for damages, 
        losses, or misuse of this software.  
        Use responsibly and only for lawful purposes.  
        """)
    st.markdown("Upload a video to track and count unique people per minute.")
    
    # Add session management
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Session", help="Clear all processing results and start fresh"):
            st.session_state.processing_results = {}
            # Clear uploaded files folder (optional)
            try:
                if Path(uploaded_folder).exists():
                    for file in Path(uploaded_folder).glob("*"):
                        file.unlink()
            except Exception:
                pass  # Ignore cleanup errors
            st.success("✅ Session cleared!")
            st.rerun()

    # Create directories
    try:
        Path(uploaded_folder).mkdir(parents=True, exist_ok=True)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create directories: {e}")
        return

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    # Mode selection (supporting both new and legacy names)
    mode_options = {
        "Fast": "Process every 5th frame (fastest)",
        "Normal": "Process every 2nd frame (balanced)", 
        "High Accuracy": "Process every frame (most accurate)"
    }
    
    speed_mode = st.sidebar.radio(
        "Processing Mode", 
        options=list(mode_options.keys()),
        index=2,  # Default to High Accuracy
        format_func=lambda x: f"{x} - {mode_options[x]}",
        help="Choose processing mode based on your speed vs accuracy needs"
    )

    # Resize options
    resize_options = [None, 480, 640, 960, 1280]
    resize_labels = ["Original", "480px", "640px", "960px", "1280px"]
    resize_width = st.sidebar.selectbox(
        "Resize Width", 
        resize_options, 
        index=2,  # Default to 640px
        format_func=lambda v: resize_labels[resize_options.index(v)] if v is not None else resize_labels[0],
        help="Smaller sizes process faster but may reduce accuracy"
    )

    # Device selection
    device_choice = st.sidebar.selectbox(
        "Processing Device", 
        ["Auto", "cuda:0", "mps", "cpu"], 
        index=0,
        help="Auto-detect is recommended"
    )
    device_override = None if device_choice == "Auto" else device_choice

    # Get all available models including custom ones
    model_options, custom_models = get_available_models()
    
    # Model selection with refresh option
    selected_model = st.sidebar.selectbox(
        "Detection Model",
        options=list(model_options.keys()),
        index=0,  # Default to yolo11n
        format_func=lambda x: model_options[x],
        help="Choose model based on speed vs accuracy needs"
    )
    
    if st.sidebar.button("🔄 Refresh Model List", help="Refresh model list", key="refresh_models", use_container_width=True):
        st.rerun()
    
    # Display currently selected model information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Currently Selected Model")
    
    # Show model information based on selection
    if selected_model.startswith("custom_") and selected_model != "custom_upload":
        if selected_model in custom_models:
            model_path = custom_models[selected_model]
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.sidebar.success(f"📁 **{model_path.name}**")
            st.sidebar.info(f"📊 Size: {size_mb:.1f} MB\n📂 Type: Custom Model")
            
            # Try to show additional model info
            try:
                from ultralytics import YOLO
                test_model = YOLO(str(model_path))
                if hasattr(test_model, 'model') and hasattr(test_model.model, 'names'):
                    class_count = len(test_model.model.names)
                    st.sidebar.info(f"🏷️ Classes: {class_count}")
            except Exception:
                pass
        else:
            st.sidebar.error("❌ Custom model not found")
    elif selected_model == "custom_upload":
        st.sidebar.warning("⚠️ Please upload a custom model")
    else:
        # Built-in model
        model_display = model_options.get(selected_model, selected_model)
        st.sidebar.success(f"🔦 **{model_display.split(' (')[0]}**")
        
        # Show model characteristics
        if "nano" in selected_model.lower():
            st.sidebar.info("⚡ Fastest processing, smallest size")
        elif "small" in selected_model.lower():
            st.sidebar.info("🚀 Fast processing, good accuracy")
        elif "medium" in selected_model.lower():
            st.sidebar.info("⚖️ Balanced speed and accuracy")
        elif "large" in selected_model.lower():
            st.sidebar.info("🎯 High accuracy, slower processing")
        
        st.sidebar.info("📂 Type: Built-in YOLO Model")
    
    # Show model status in main area
    if selected_model and selected_model != "custom_upload":
        if selected_model.startswith("custom_") and selected_model in custom_models:
            model_name = custom_models[selected_model].name
            st.info(f"🎯 **Currently Selected:** Custom Model - {model_name}")
        elif not selected_model.startswith("custom_"):
            model_name = model_options[selected_model].split(' (')[0]
            st.info(f"🔦 **Currently Selected:** {model_name}")
    else:
        st.warning("⚠️ **No model selected** - Please select or upload a model to continue")
    
    # Handle custom model selection
    custom_model_path = None
    
    # Check if a custom model is selected from existing models
    if selected_model.startswith("custom_") and selected_model != "custom_upload":
        if selected_model in custom_models:
            custom_model_path = custom_models[selected_model]
            st.sidebar.success(f"✅ Using custom model: {custom_model_path.name}")
            
            # Show model information
            try:
                size_mb = custom_model_path.stat().st_size / (1024 * 1024)
                st.sidebar.info(f"📊 Size: {size_mb:.1f} MB")
                
                # Try to get model info
                try:
                    from ultralytics import YOLO
                    test_model = YOLO(str(custom_model_path))
                    
                    if hasattr(test_model, 'model') and hasattr(test_model.model, 'names'):
                        class_names = list(test_model.model.names.values())
                        st.sidebar.info(f"🏷️ Classes: {len(class_names)}")
                        if len(class_names) <= 10:
                            st.sidebar.text("Classes: " + ", ".join(class_names))
                            
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load model info: {str(e)}")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error reading model file: {str(e)}")
                custom_model_path = None
    
    # Handle new custom model upload
    elif selected_model == "custom_upload":
        st.sidebar.markdown("### 📁 Upload New Custom Model")
        uploaded_model = st.sidebar.file_uploader(
            "Upload your custom .pt model file",
            type=["pt"],
            help="Upload a custom YOLO model file (.pt format)"
        )
        
        if uploaded_model is not None:
            # Save the uploaded model to models directory
            models_dir = Path("models/custom")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            custom_model_path = models_dir / uploaded_model.name
            
            # Check if model already exists
            if custom_model_path.exists():
                st.sidebar.warning(f"⚠️ Model {uploaded_model.name} already exists")
                if st.sidebar.button("🔄 Overwrite existing model"):
                    # Continue with upload
                    pass
                else:
                    st.sidebar.info("Using existing model or choose a different name")
                    custom_model_path = custom_model_path  # Use existing
            
            # Save the uploaded file
            try:
                with open(custom_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                # Get file size
                file_size_mb = custom_model_path.stat().st_size / (1024 * 1024)
                
                st.sidebar.success(f"✅ Model uploaded: {uploaded_model.name}")
                st.sidebar.info(f"📊 Size: {file_size_mb:.1f} MB")
                
                # Try to validate the model
                try:
                    from ultralytics import YOLO
                    test_model = YOLO(str(custom_model_path))
                    st.sidebar.success("✅ Model validation passed")
                    
                    # Show model info if available
                    if hasattr(test_model, 'model') and hasattr(test_model.model, 'names'):
                        class_names = list(test_model.model.names.values())
                        st.sidebar.info(f"🏷️ Classes: {len(class_names)}")
                        if len(class_names) <= 10:
                            st.sidebar.text("Classes: " + ", ".join(class_names))
                        
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Model validation failed: {str(e)}")
                    st.sidebar.info("Model may still work, but validation failed")
                
                # Suggest refreshing to see the model in dropdown
                st.sidebar.info("💡 Refresh the page to see the new model in the dropdown")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Failed to save model: {str(e)}")
                custom_model_path = None
        else:
            st.sidebar.info("👆 Please upload a .pt model file to use custom model")
    
    # Show custom models management section
    if custom_models:
        with st.sidebar.expander("🗂️ Manage Custom Models"):
            st.write(f"Found {len(custom_models)} custom model(s):")
            for model_key, model_path in custom_models.items():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {model_path.name}")
                    st.caption(f"Size: {size_mb:.1f} MB")
                with col2:
                    if st.button("🗑️", key=f"delete_{model_key}", help=f"Delete {model_path.name}"):
                        try:
                            model_path.unlink()
                            st.success(f"Deleted {model_path.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        confidence_threshold = st.slider(
            "Detection Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence for person detection"
        )
        
        hide_boxes = st.checkbox(
            "Hide tracking boxes", 
            value=False,
            help="Disable bounding box drawing for faster processing"
        )
        
        # Resource optimization toggle
        optimize_resources = st.checkbox(
            "🚀 Optimize for 90% Resource Usage",
            value=True,
            help="Automatically optimize settings to use 90% of available system resources"
        )
        
        if not optimize_resources:
            memory_limit = st.slider(
                "Memory Limit (MB)",
                min_value=256,
                max_value=8192,
                value=1024,
                step=256,
                help="Maximum memory usage for frame processing"
            )
        else:
            # Show estimated optimal settings
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
                
                optimizer = ResourceOptimizer(target_utilization=0.9)
                optimal_settings = optimizer.get_optimal_settings()
                
                st.info(f"🎯 Optimal Settings Detected:\n"
                       f"- Memory: {optimal_settings['memory_limit_mb']} MB\n"
                       f"- Device: {optimal_settings['device']}\n"
                       f"- Recommended Model: {optimal_settings['model_recommendation']}\n"
                       f"- Processing Mode: {optimal_settings['processing_mode']}")
                
                memory_limit = optimal_settings['memory_limit_mb']
                
                # Auto-select optimal model if resource optimization is enabled (only for built-in models)
                if not selected_model.startswith("custom_") and optimal_settings['model_recommendation'] in model_options:
                    selected_model = optimal_settings['model_recommendation']
                    
                # Auto-select optimal processing mode
                if optimal_settings['processing_mode'] in mode_options:
                    speed_mode = optimal_settings['processing_mode']
                    
            except Exception as e:
                st.warning(f"Could not detect optimal settings: {e}")
                memory_limit = 2048  # Default to higher limit

    # File upload
    st.header("📁 Video Upload")
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV"
    )
    


    
    if not uploaded_file:
        st.info("👆 Upload a video file to begin tracking people")
        
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Upload** your video file
            2. **Configure** processing settings in the sidebar  
            3. **Process** the video to track people
            4. **View** results with tracking visualization and analytics
            5. **Download** processed video and CSV data
            
            The system uses YOLOv8's built-in tracking for reliable person detection and tracking.
            """)
            
        with st.expander("🔧 Custom Models"):
            st.markdown("""
            ### Using Custom YOLO Models
            
            You can upload your own trained YOLO models (.pt files) for specialized detection:
            
            **Supported Model Types:**
            - YOLOv8/YOLOv11 models (.pt format)
            - Custom trained models
            - Fine-tuned models for specific scenarios
            
            **Requirements:**
            - Model must be in PyTorch (.pt) format
            - Should be compatible with Ultralytics YOLO
            - Person class should be included (class 0) for tracking
            
            **Tips:**
            - Larger models = better accuracy but slower processing
            - Test with small videos first
            - Custom models are saved in `models/custom/` directory
            """)
        return

    # Save uploaded file
    input_path = Path(uploaded_folder) / uploaded_file.name
    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        return

    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Display video info
    col1, col2 = st.columns([6, 1])
    with col1:
        try:
            st.video(str(input_path))
        except Exception as e:
            st.error(f"Failed to display video: {e}")
            st.info("The video file may be corrupted or in an unsupported format.")
    with col2:
        file_size = input_path.stat().st_size / (1024*1024)
        st.metric("File Size", f"{file_size:.1f} MB")

    # Initialize session state
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = {}
    
    # Clear invalid results from previous sessions
    if st.session_state.processing_results:
        video_path = st.session_state.processing_results.get("video_path")
        csv_path = st.session_state.processing_results.get("csv_path")
        
        # Check if files still exist
        if video_path and not Path(video_path).exists():
            st.session_state.processing_results = {}
            # Don't rerun here as it can interrupt processing

    # Generate output paths
    stem = Path(uploaded_file.name).stem
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = Path(output_folder) / f"processed_{stem}_{timestamp}.mp4"
    output_csv_path = Path(output_folder) / f"log_{stem}_{timestamp}.csv"

    # Processing section
    st.header("🚀 Processing")
    
    # Show current configuration summary
    with st.expander("📋 Current Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Model Settings**")
            if selected_model.startswith("custom_") and selected_model != "custom_upload":
                if selected_model in custom_models:
                    model_path = custom_models[selected_model]
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    st.write(f"📁 **{model_path.name}**")
                    st.write(f"Size: {size_mb:.1f} MB")
                    st.write("Type: Custom Model")
            elif selected_model != "custom_upload":
                model_name = model_options[selected_model].split(' (')[0]
                st.write(f"🔦 **{model_name}**")
                st.write("Type: Built-in Model")
            else:
                st.write("⚠️ No model selected")
        
        with col2:
            st.markdown("**⚙️ Processing Settings**")
            st.write(f"Mode: {speed_mode}")
            st.write(f"Device: {device_choice}")
            st.write(f"Confidence: {confidence_threshold}")
            if resize_width:
                st.write(f"Resize: {resize_width}px")
            else:
                st.write("Resize: Original")
        
        with col3:
            st.markdown("**🧠 Resource Settings**")
            if optimize_resources:
                st.write("🚀 90% Resource Optimization: ON")
            else:
                st.write("⚙️ Manual Settings")
            st.write(f"Memory Limit: {memory_limit} MB")
            st.write(f"Boxes: {'Hidden' if hide_boxes else 'Visible'}")
    
    # Processing button with model info
    model_info_text = ""
    if selected_model.startswith("custom_") and selected_model != "custom_upload":
        if selected_model in custom_models:
            model_info_text = f" with {custom_models[selected_model].name}"
    elif selected_model != "custom_upload":
        model_info_text = f" with {selected_model.upper()}"

    # Ensure progress_bar and status_text are always defined
    progress_bar = None
    status_text = None

    if st.button(f"▶️ Start Processing ({speed_mode}){model_info_text}", type="primary", use_container_width=True):
        
        try:
            # Check if custom model is selected but not available
            is_custom_model = selected_model.startswith("custom_") or selected_model == "custom_upload"
            if is_custom_model and custom_model_path is None:
                st.error("❌ Please select or upload a custom model file before processing")
                return
            
            # Initialize processor with flexible model selection and optimization
            model_display_name = model_options[selected_model]
            if is_custom_model and custom_model_path:
                model_display_name = f"Custom Model ({custom_model_path.name})"
                
            with st.spinner(f"Initializing video processor with {model_display_name}..."):
                if is_custom_model and custom_model_path:
                    processor = VideoProcessor(
                        model_path=str(custom_model_path),
                        resize_width=resize_width,
                        device=device_override,
                        max_memory_mb=memory_limit,
                        optimize_resources=optimize_resources,
                    )
                else:
                    processor = VideoProcessor(
                        model_name=selected_model,
                        resize_width=resize_width,
                        device=device_override,
                        max_memory_mb=memory_limit,
                        optimize_resources=optimize_resources,
                    )
                
                # Apply custom settings using proper methods
                processor.set_draw_enabled(not hide_boxes)  # Invert because checkbox is "hide"
                processor.set_confidence(confidence_threshold)
                
                
            # Progress tracking - simplified to avoid UI refresh issues
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Starting processing...")

            def progress_callback(percent: int):
                try:
                    # Ensure percent is valid
                    percent = min(100, max(0, int(percent)))
                    
                    # Update UI elements (no rerun calls to avoid refresh issues)
                    progress_bar.progress(percent / 100.0)
                    status_text.text(f"Processing frames... {percent}%")
                        
                except Exception as e:
                    print(f"[WARN] Progress callback error: {e}")
                    # Fallback to simple progress update
                    try:
                        progress_bar.progress(percent / 100.0)
                        status_text.text(f"Processing... {percent}%")
                    except:
                        pass  # Ignore UI update errors

            # Process video
            start_time = dt.datetime.now()
            
            # Add processing status indicator
            with st.spinner("Processing video... This may take several minutes for large files."):
                # Use the legacy parameter name for backward compatibility
                video_path, csv_path, processing_time = processor.process_video(
                    input_path=str(input_path),
                    output_video_path=str(output_video_path),
                    output_csv_path=str(output_csv_path),
                    speed_mode=speed_mode,  # Use legacy parameter
                    progress_callback=progress_callback,
                )
            
            # Store results
            model_info = selected_model
            if is_custom_model and custom_model_path:
                model_info = f"custom ({custom_model_path.name})"
                
            st.session_state.processing_results = {
                "video_path": video_path,
                "csv_path": csv_path,
                "processing_time": processing_time,
                "start_time": start_time,
                "mode": speed_mode,
                "settings": {
                    "model": model_info,
                    "custom_model_path": str(custom_model_path) if custom_model_path else None,
                    "resize_width": resize_width,
                    "device": device_override or "auto",
                    "confidence": confidence_threshold,
                    "hide_boxes": hide_boxes,
                    "memory_limit": memory_limit
                }
            }
            
            # Final UI updates
            progress_bar.progress(1.0)
            status_text.text(f"✅ Processing completed in {processing_time:.1f} seconds")
        except Exception as e:
            # Reset progress UI on error
            try:
                if progress_bar is not None:
                    progress_bar.progress(0)
                if status_text is not None:
                    status_text.text("❌ Processing failed")
            except Exception:
                pass  # Ignore UI update errors
            
            st.error(f"❌ Processing failed: {str(e)}")
            
            # Show helpful error messages for common issues
            error_str = str(e).lower()
            if "memory" in error_str or "cuda" in error_str:
                st.info("💡 Try reducing the memory limit or using a smaller model (e.g., yolo11n)")
            elif "codec" in error_str or "video" in error_str:
                st.info("💡 Try converting your video to MP4 format or reducing the resolution")
            elif "timeout" in error_str:
                st.info("💡 Large videos may take longer to process. Consider using 'Fast' mode or reducing video resolution")
            
            st.exception(e)
            return
                

    # Results section
    results = st.session_state.processing_results
    
    if results and results.get("video_path") and Path(results.get("video_path", "")).exists():
        st.header("📊 Results")
        
        # Processing info
        with st.expander("ℹ️ Processing Details"):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Processing Time", f"{results['processing_time']:.1f}s")
            with col2:
                st.metric("Mode", results['mode'])
            with col3:
                settings = results['settings']
                model_display = settings.get('model', 'yolo11n')
                if model_display.startswith('custom'):
                    st.metric("Model", "📁 Custom")
                else:
                    st.metric("Model", model_display.upper())
            with col4:
                st.metric("Device", settings['device'].upper())
            with col5:
                st.metric("Memory", f"{settings.get('memory_limit', 1024)}MB")
            
            # Show detailed model information
            if settings.get('custom_model_path'):
                st.info(f"🎯 **Custom Model Used:** {Path(settings['custom_model_path']).name}")
            else:
                model_name = settings.get('model', 'yolo11n')
                st.info(f"🔦 **Built-in Model Used:** {model_name.upper()}")

        # Processed video
        st.subheader("🎬 Processed Video")
        try:
            # Verify file exists before displaying
            video_file = Path(results["video_path"])
            if video_file.exists() and video_file.stat().st_size > 0:
                st.video(results["video_path"])
            else:
                st.error("❌ Processed video file not found or empty")
                st.session_state.processing_results = {}  # Clear invalid results
        except Exception as e:
            st.error(f"❌ Error displaying processed video: {e}")
            st.session_state.processing_results = {}  # Clear invalid results

        # Analytics
        if Path(results.get("csv_path", "")).exists():
            st.subheader("📈 Analytics Dashboard")
            
            try:
                df = pd.read_csv(results["csv_path"])
                
                if not df.empty and all(col in df.columns for col in ["Minute", "Unique_People_Count"]):
                    # Summary statistics
                    total_detections = df["Unique_People_Count"].sum()
                    max_people = df["Unique_People_Count"].max()
                    avg_people = df["Unique_People_Count"].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Detections", int(total_detections))
                    with col2:
                        st.metric("Peak per Minute", int(max_people))
                    with col3:
                        st.metric("Average per Minute", f"{avg_people:.1f}")
                    with col4:
                        st.metric("Duration", f"{len(df)} minutes")

                    # Chart controls
                    chart_col1, chart_col2 = st.columns([1, 4])
                    with chart_col1:
                        chart_type = st.radio("Chart Type", ["Line Chart", "Bar Chart"], index=0)
                        show_values = st.checkbox("Show Values", value=True)

                    # Create chart
                    with chart_col2:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        x = df["Minute"].astype(int)
                        y = df["Unique_People_Count"].astype(int)

                        if chart_type == "Line Chart":
                            ax.plot(x, y, marker="o", linewidth=2, markersize=6, color="blue")
                            ax.fill_between(x, y, alpha=0.3, color="blue")
                            
                            if show_values:
                                for i, val in enumerate(y):
                                    ax.annotate(str(val), (x.iloc[i], val), 
                                              textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                        else:
                            bars = ax.bar(x, y, color="skyblue", alpha=0.8, edgecolor="black", linewidth=0.5)
                            
                            if show_values:
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                           f'{int(height)}', ha='center', va='bottom', fontsize=9)

                        ax.set_xlabel("Minute", fontsize=12)
                        ax.set_ylabel("Unique People Count", fontsize=12)
                        ax.set_title("People Count Over Time", fontsize=14, fontweight='bold')
                        ax.grid(True, linestyle="--", alpha=0.7)
                        ax.set_xticks(sorted(x.unique()))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    # Raw data
                    with st.expander("📋 Raw Data"):
                        st.dataframe(df, use_container_width=True)
                        
                else:
                    st.warning("⚠️ CSV file is empty or missing required columns")
                    
            except Exception as e:
                st.error(f"❌ Failed to load analytics data: {e}")

        # Download section
        st.subheader("⬇️ Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            video_file = Path(results["video_path"])
            if video_file.exists() and video_file.stat().st_size > 0:
                try:
                    with open(results["video_path"], "rb") as f:
                        st.download_button(
                            label="📹 Download Processed Video",
                            data=f.read(),
                            file_name=video_file.name,
                            mime="video/mp4",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Failed to prepare video download: {e}")
            else:
                st.error("❌ Video file not available for download")
            
        with col2:
            csv_file = Path(results["csv_path"])
            if csv_file.exists() and csv_file.stat().st_size > 0:
                try:
                    with open(results["csv_path"], "rb") as f:
                        st.download_button(
                            label="📊 Download Analytics CSV",
                            data=f.read(),
                            file_name=csv_file.name,
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Failed to prepare CSV download: {e}")
            else:
                st.error("❌ CSV file not available for download")
        
       
                
    elif results:
        # Clear invalid results and show message
        st.warning("⚠️ Previous processing results are no longer available. Please process a new video.")
        if st.button("Clear Results", help="Clear stale processing results"):
            st.session_state.processing_results = {}
            st.rerun()


# if __name__ == "__main__":
#     streamlit_main()
