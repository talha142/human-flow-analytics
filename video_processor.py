import cv2
import pandas as pd
import time
from ultralytics import YOLO  # YOLOv8 model
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracker
import math
import os


class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt", resize_width=None):
        self.model = YOLO(model_path)  # Load YOLO model
        self.tracker = DeepSort(max_age=70)  # Initialize DeepSORT tracker with max_age
        self.resize_width = resize_width  # Optional width resizing
        self.global_id_map = {}   # Map from local IDs to global unique IDs
        self.next_global_id = 1   # Next global ID counter

    def process_video(
        self, 
        input_path, 
        output_video_path, 
        output_csv_path, 
        speed_mode="High Accuracy", 
        progress_callback=None
    ):
        start_time = time.time()  # Start timer for processing
        cap = cv2.VideoCapture(input_path)  # Open input video
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)  # Get FPS
        width, height = int(cap.get(3)), int(cap.get(4))  # Get frame dimensions

        # Optional resizing for speed
        if self.resize_width:
            ratio = self.resize_width / width
            width, height = self.resize_width, int(height * ratio)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Determine frame skip based on speed mode
        skip_interval = 1
        if speed_mode == "Fast":
            skip_interval = 5
        elif speed_mode == "Normal":
            skip_interval = 2

        # Dictionary to store unique person IDs per minute
        unique_ids_per_minute = {}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while True:
            ret, frame = cap.read()  # Read frame
            if not ret:
                break

            frame_idx += 1

            # Skip frames for faster processing
            if frame_idx % skip_interval != 0:
                continue

            # Resize frame if required
            if self.resize_width:
                frame = cv2.resize(frame, (width, height))

            # YOLO detection + DeepSORT tracking
            results = self.model.track(frame, conf=0.5, persist=True)
            current_minute = math.floor(frame_idx / (fps * 60))  # Minute index

            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                ids = getattr(r.boxes, "id", None) if boxes else None

                if boxes is not None:
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        if cls_id == 0:  # Only consider person class
                            if ids is not None:
                                local_id = int(ids[i])

                                # Assign global unique ID
                                if local_id not in self.global_id_map:
                                    self.global_id_map[local_id] = self.next_global_id
                                    self.next_global_id += 1
                                global_id = self.global_id_map[local_id]

                                # Track unique IDs per minute
                                if current_minute not in unique_ids_per_minute:
                                    unique_ids_per_minute[current_minute] = set()
                                unique_ids_per_minute[current_minute].add(global_id)

                                # Draw bounding box and ID on frame
                                xyxy = box.xyxy[0].tolist()
                                x1, y1, x2, y2 = map(int, xyxy[:4])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    frame, 
                                    f"Person {global_id}", 
                                    (x1, max(15, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, 
                                    (0, 255, 0), 
                                    1
                                )

            out.write(frame)  # Write frame to output video

            # Update progress bar if callback provided
            if progress_callback and total_frames > 0:
                percent = int(frame_idx / total_frames * 100)
                progress_callback(min(100, percent))

        cap.release()  # Release video capture
        out.release()  # Release video writer

        # ---------------------------
        # SAVE MINUTE-WISE CSV ONLY
        # ---------------------------
        summary_log = []
        for minute, ids in unique_ids_per_minute.items():
            summary_log.append([minute, len(ids)])  # [Minute, Unique Count]

        df = pd.DataFrame(summary_log, columns=["Minute", "Unique_People_Count"])
        df.to_csv(output_csv_path, index=False)  # Save CSV

        processing_time = round(time.time() - start_time, 2)  # Total processing time
        return output_video_path, output_csv_path, processing_time  # Return paths and time
