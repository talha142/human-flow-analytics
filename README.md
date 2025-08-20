
# Human Flow Tracking & Analytics

## 📖 Project Overview
Human Flow Tracking & Analytics is a video analytics platform designed to detect, track, and analyze human movement in videos.  
The system integrates **YOLOv8 for object detection**, **DeepSORT for multi-object tracking**, and **Streamlit for the user interface**.  

The application automatically detects people in uploaded videos, assigns unique IDs, and calculates the number of unique individuals per minute.  
It also generates visual analytics, processed video output, and downloadable CSV logs for further analysis.

---

## 🚀 Key Features
- 🎥 Upload video for processing  
- 👤 Human detection using **YOLOv8/YOLOv11** with flexible model selection
- 🔄 Built-in tracking with **YOLOv8 tracking**  
- ⏱️ Count **unique individuals per minute**  
- 📊 Real-time analytics dashboard with charts  
- 📂 Download processed video with bounding boxes and IDs  
- 📑 Export CSV log containing:
  - Frame number  
  - Minute  
  - People count  
  - Unique people count  
- 🧠 **Memory optimization** with batch processing
- 🎛️ **Flexible model selection** (nano to extra-large models)
- ⚡ **Hardware-aware configuration** with automatic performance profiling
- 🛠️ **Model management utilities** for downloading and validation
- 📁 **Custom model support** - Upload your own trained .pt models
- 🔧 **Model validation** - Automatic validation of uploaded models  

---

## ⚙️ Installation Guide

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/human-flow-tracking.git
   cd human-flow-tracking
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


Usage Instructions

Launch the Streamlit app:

python main.py
```

Or run Streamlit directly:
```bash
streamlit run frontend/app.py
```

### 🎛️ Configuration Options

From the browser interface:

1. **Upload a video file**
2. **Select processing mode**: Fast, Normal, or High Accuracy
3. **Choose YOLO model**: From nano (fastest) to extra-large (most accurate)
4. **Configure memory settings**: Set memory limit for optimal performance
5. **Adjust detection settings**: Confidence threshold, resize options
6. **Process the video**

After processing, you will get:
- 📈 A line chart showing people count per minute
- 🎬 A processed video with bounding boxes and IDs
- 📊 A downloadable CSV log
- 📋 Detailed processing statistics

### 🛠️ Model Management

Use the model manager utility:

```bash
# List available models
python utils/model_manager.py list

# Download specific model
python utils/model_manager.py download --model yolo11s

# Validate model file
python utils/model_manager.py validate --model yolo11n

# Clean up old models
python utils/model_manager.py cleanup

# Move misplaced models to correct directory
python utils/model_manager.py move

# Install custom model
python utils/model_manager.py install-custom --path /path/to/custom_model.pt

# List custom models
python utils/model_manager.py list-custom

# Quick organize script
python organize_models.py
```
## 🛠️ Technology Stack

- **Programming Language**: Python 3.8+
- **Frontend Framework**: Streamlit
- **Detection Models**: YOLOv8/YOLOv11 (nano to extra-large variants)
- **Tracking Algorithm**: YOLOv8 built-in tracking
- **Libraries Used**: Pandas, OpenCV, Matplotlib, PyTorch, Ultralytics
- **Memory Management**: Batch processing with configurable limits
- **Hardware Support**: CUDA, MPS (Apple Silicon), CPU fallback

## 📊 Performance Profiles

The system automatically detects your hardware and suggests optimal settings:

| Profile | Memory | GPU | Recommended Model | Batch Size |
|---------|--------|-----|-------------------|------------|
| Low-end | <8GB | No | yolo11n | 1-2 |
| Mid-range | 8-16GB | Optional | yolo11s | 2-4 |
| High-end | 16GB+ | Yes | yolo11m | 4-8 |
| Server | 32GB+ | Yes | yolo11l/x | 8-16 |

## 🔧 Advanced Configuration

### Environment Variables
```bash
export FLOW_MEMORY_LIMIT=2048    # Memory limit in MB
export FLOW_MODEL_DIR=./models   # Model cache directory
export FLOW_DEFAULT_MODEL=yolo11s # Default model name
```

### Memory Optimization
- **Batch Processing**: Processes multiple frames together for efficiency
- **Dynamic Memory Management**: Automatically clears cache when limits are reached
- **Hardware-aware Settings**: Adjusts batch sizes based on available memory
- **Model Caching**: Downloads and caches models locally for faster startup

## 📁 Custom Model Support

### Using Your Own YOLO Models

The system supports uploading and using custom trained YOLO models:

#### **Supported Formats**
- PyTorch (.pt) format
- YOLOv8/YOLOv11 compatible models
- Custom trained models
- Fine-tuned models for specific scenarios

#### **Upload Methods**

**Via Web Interface:**
1. Select "📁 Upload Custom Model" from the model dropdown
2. Upload your .pt file using the file uploader
3. System automatically validates the model
4. Start processing with your custom model

**Via Command Line:**
```bash
# Install custom model
python utils/model_manager.py install-custom --path /path/to/model.pt --name my_model

# List installed custom models
python utils/model_manager.py list-custom

# Validate custom model
python utils/model_manager.py validate --path models/custom/my_model.pt
```

#### **Requirements**
- Model must include person class (class 0) for tracking
- Compatible with Ultralytics YOLO framework
- Recommended: Test with small videos first

#### **Custom Model Directory Structure**
```
models/
├── custom/
│   ├── my_custom_model.pt
│   ├── specialized_model.pt
│   └── fine_tuned_model.pt
├── yolo11n.pt
└── yolo11s.pt
```
