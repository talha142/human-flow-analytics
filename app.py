import sys, os
# Add parent directory to sys.path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.video_processor import VideoProcessor  # Import custom VideoProcessor class

import streamlit as st  # Streamlit for frontend UI
import pandas as pd      # Pandas for data handling
import matplotlib.pyplot as plt  # Matplotlib for plotting charts

# === Folders ===
UPLOAD_FOLDER = "uploads"  # Folder to save uploaded videos
OUTPUT_FOLDER = "outputs"  # Folder to save processed videos and CSVs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set Streamlit page configuration
st.set_page_config(page_title="Human Flow Tracking & Analytics", layout="wide")
st.title("Human Flow Tracking & Analytics")  # Page title

# === Upload Video ===
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])  # File uploader widget
if uploaded_file:
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    # Save uploaded file to the uploads folder
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved: {uploaded_file.name}")
    st.video(video_path)  # Preview uploaded video

    # === Initialize Session State ===
    # Using session_state to persist output paths and processing time
    if "out_video" not in st.session_state:
        st.session_state["out_video"] = None
    if "out_csv" not in st.session_state:
        st.session_state["out_csv"] = None
    if "proc_time" not in st.session_state:
        st.session_state["proc_time"] = None

    # === Video Processing Button ===
    if st.button("Start Processing (High Accuracy)"):
        # Define paths for processed video and CSV log
        output_video_path = os.path.join(OUTPUT_FOLDER, f"processed_{uploaded_file.name}")
        output_csv_path = os.path.join(OUTPUT_FOLDER, f"log_{os.path.splitext(uploaded_file.name)[0]}.csv")

        processor = VideoProcessor(resize_width=640)  # Initialize VideoProcessor
        progress_bar = st.progress(0)  # Progress bar widget
        progress_text = st.empty()     # Placeholder for progress text
        progress_text.text("Processing video, please wait...")

        # Callback function to update progress bar during processing
        def progress_callback(percent):
            progress_bar.progress(percent)
            progress_text.text(f"Processing: {percent}%")

        # === Process Video ===
        # Process the video and get output paths and processing time
        out_video, out_csv, proc_time = processor.process_video(
            video_path, output_video_path, output_csv_path,
            speed_mode="High Accuracy",
            progress_callback=progress_callback
        )

        # === Save results to session_state ===
        st.session_state["out_video"] = out_video
        st.session_state["out_csv"] = out_csv
        st.session_state["proc_time"] = proc_time

        progress_bar.progress(100)
        progress_text.text(f"✅ Processing complete in {proc_time} sec")
        st.video(out_video)  # Show processed video

    # === Analytics & Line Chart ===
    # Display unique people detected per minute
    if st.session_state.get("out_csv") and os.path.exists(st.session_state["out_csv"]):
        df = pd.read_csv(st.session_state["out_csv"])
        st.subheader("Unique People Detected Per Minute")

        fig, ax = plt.subplots()
        ax.plot(df["Minute"], df["Unique_People_Count"], marker="o")  # Line chart
        ax.set_xlabel("Minute")
        ax.set_ylabel("Unique People Count")
        st.pyplot(fig)  # Render chart in Streamlit

    # === Download Buttons ===
    # Button to download processed video
    if st.session_state.get("out_video") and os.path.exists(st.session_state["out_video"]):
        with open(st.session_state["out_video"], "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name=os.path.basename(st.session_state["out_video"]),
                mime="video/mp4"
            )

    # Button to download CSV analytics
    if st.session_state.get("out_csv") and os.path.exists(st.session_state["out_csv"]):
        with open(st.session_state["out_csv"], "rb") as f:
            st.download_button(
                label="Download CSV Analytics",
                data=f,
                file_name=os.path.basename(st.session_state["out_csv"]),
                mime="text/csv"
            )
