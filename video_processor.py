import cv2
import pandas as pd
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import os

class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt", resize_width=None, roi=None, conf_threshold=0.6):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=150, max_cosine_distance=0.4)
        self.resize_width = resize_width
        self.roi = roi  # ROI = (x1, y1, x2, y2) rectangle
        self.conf_threshold = conf_threshold

    def _is_inside_roi(self, x1, y1, x2, y2):
        """Check if bounding box is inside ROI"""
        if self.roi is None:
            return True
        rx1, ry1, rx2, ry2 = self.roi
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

    def process_video(self, input_path, output_video_path, output_csv_path, speed_mode="High Accuracy", progress_callback=None):
        start_time = time.time()
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width, height = int(cap.get(3)), int(cap.get(4))

        # Optional resizing for speed
        if self.resize_width:
            ratio = self.resize_width / width
            width, height = self.resize_width, int(height * ratio)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Speed mode frame skip
        skip_interval = 1
        if speed_mode == "Fast":
            skip_interval = 5
        elif speed_mode == "Normal":
            skip_interval = 2

        unique_ids_per_minute = {}
        log_data = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames for speed
            if frame_idx % skip_interval != 0:
                continue

            # Resize frame if needed
            if self.resize_width:
                frame = cv2.resize(frame, (width, height))

            # YOLO detection
            results = self.model(frame, conf=self.conf_threshold)
            people_dets = []

            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id == 0 and conf >= self.conf_threshold:  # person class
                            xyxy = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = map(int, xyxy[:4])

                            if self._is_inside_roi(x1, y1, x2, y2):
                                people_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

            # DeepSORT tracking
            tracks = self.tracker.update_tracks(people_dets, frame=frame)
            people_count = 0
            current_minute = math.floor(frame_idx / (fps * 60))

            if len(tracks) > 0:
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    people_count += 1

                    if current_minute not in unique_ids_per_minute:
                        unique_ids_per_minute[current_minute] = set()
                    unique_ids_per_minute[current_minute].add(track_id)

                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save per-frame log
            log_data.append([
                frame_idx,
                current_minute,
                people_count,
                len(unique_ids_per_minute.get(current_minute, set()))
            ])

            out.write(frame)

            # Update progress
            if progress_callback and total_frames > 0:
                percent = int(frame_idx / total_frames * 100)
                progress_callback(min(100, percent))

        cap.release()
        out.release()

        # Save CSV
        df = pd.DataFrame(log_data, columns=["Frame", "Minute", "People_Count", "Unique_People_Count"])
        df.to_csv(output_csv_path, index=False)

        processing_time = round(time.time() - start_time, 2)
        return output_video_path, output_csv_path, processing_time
