
# Human Flow Tracking & Analytics

## 📖 Project Overview
Human Flow Tracking & Analytics is a video analytics platform designed to detect, track, and analyze human movement in videos.  
The system integrates **YOLOv8 for object detection**, **DeepSORT for multi-object tracking**, and **Streamlit for the user interface**.  

The application automatically detects people in uploaded videos, assigns unique IDs, and calculates the number of unique individuals per minute.  
It also generates visual analytics, processed video output, and downloadable CSV logs for further analysis.

---

## 🚀 Key Features
- 🎥 Upload video for processing  
- 👤 Human detection using **YOLOv8**  
- 🔄 Multi-object tracking with **DeepSORT**  
- ⏱️ Count **unique individuals per minute**  
- 📊 Real-time analytics dashboard with charts  
- 📂 Download processed video with bounding boxes and IDs  
- 📑 Export CSV log containing:
  - Frame number  
  - Minute  
  - People count  
  - Unique people count  

---

## ⚙️ Installation Guide

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/human-flow-tracking.git
   cd human-flow-tracking

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
Usage Instructions

Launch the Streamlit app:

streamlit run frontend/app.py


From the browser interface:

Upload a video file

(Optional) Upload a ground truth CSV for accuracy evaluation

Process the video

After processing, you will get:

 A line chart showing people count per minute

 A processed video with bounding boxes and IDs

 A downloadable CSV log

🛠️ Technology Stack

Programming Language: Python

Frontend Framework: Streamlit

Detection Model: YOLOv8

Tracking Algorithm: DeepSORT

Libraries Used: Pandas, OpenCV, Matplotlib
