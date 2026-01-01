"""
YOLOv8 People Detection and IoU-Based Tracking System with Streamlit Frontend
This script provides a web interface for YOLOv8-based people detection and tracking.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from ultralytics import YOLO
import plotly.graph_objects as go
import pandas as pd


class IoUTracker:
    """Simple IoU-based tracker for object tracking across frames"""
    
    def __init__(self, iou_threshold=0.3, max_frames_to_skip=30):
        self.iou_threshold = iou_threshold
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
    def compute_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """
        Update tracks with new detections
        detections: list of [x1, y1, x2, y2, confidence]
        returns: list of (track_id, box) tuples
        """
        self.frame_count += 1
        
        # Update frames_since_update for all tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['frames_since_update'] += 1
            
            # Remove tracks that haven't been updated for too long
            if self.tracks[track_id]['frames_since_update'] > self.max_frames_to_skip:
                del self.tracks[track_id]
        
        if len(detections) == 0:
            return [(tid, t['box']) for tid, t in self.tracks.items()]
        
        # Match detections to existing tracks using IoU
        matched_tracks = set()
        matched_detections = set()
        results = []
        
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                    
                iou = self.compute_iou(track['box'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_iou >= self.iou_threshold:
                # Update existing track
                self.tracks[track_id]['box'] = detections[best_det_idx][:4]
                self.tracks[track_id]['frames_since_update'] = 0
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
                results.append((track_id, detections[best_det_idx][:4]))
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'box': det[:4],
                    'frames_since_update': 0
                }
                results.append((new_id, det[:4]))
        
        return results


@st.cache_resource
def load_model():
    """Load YOLOv8n model (cached)"""
    return YOLO('yolov8n.pt')


def process_video(video_path, model, iou_thresh, conf_thresh, max_skip, progress_bar, status_text):
    """Process video with YOLOv8 and IoU tracking"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return None, None, None
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = IoUTracker(iou_threshold=iou_thresh, max_frames_to_skip=max_skip)
    
    # Statistics
    unique_people = set()
    frame_num = 0
    people_per_frame = []
    unique_per_frame = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Run YOLOv8 detection
        results = model(frame, verbose=False)
        
        # Filter for person class only (class 0 in COCO)
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls == 0:  # Person class
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    if conf > conf_thresh:
                        detections.append([box[0], box[1], box[2], box[3], conf])
        
        # Update tracker
        tracked_objects = tracker.update(detections)
        
        # Draw tracked objects
        for track_id, box in tracked_objects:
            unique_people.add(track_id)
            
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID label
            label = f"ID: {track_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add info overlay
        info_text = f"Frame: {frame_num}/{total_frames} | People: {len(tracked_objects)} | Unique: {len(unique_people)}"
        cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Store statistics
        people_per_frame.append(len(tracked_objects))
        unique_per_frame.append(len(unique_people))
        
        # Update progress
        progress = frame_num / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: {frame_num}/{total_frames} frames | Unique people: {len(unique_people)}")
    
    cap.release()
    out.release()
    
    return output_path, len(unique_people), (people_per_frame, unique_per_frame, total_frames)


def main():
    """Main Streamlit app"""
    
    # Page config
    st.set_page_config(
        page_title="YOLOv8 People Tracker",
        page_icon="🎥",
        layout="wide"
    )
    
    # Title and description
    st.title("🎥 YOLOv8 People Detection & IoU Tracking")
    st.markdown("""
    Upload a video to detect and track people using YOLOv8n and IoU-based tracking.
    The system will count unique individuals across the entire video.
    """)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.3, 
        step=0.05,
        help="Higher values require more overlap to match tracks"
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence for person detection"
    )
    
    max_frames_skip = st.sidebar.slider(
        "Max Frames to Skip", 
        min_value=5, 
        max_value=60, 
        value=30, 
        step=5,
        help="How long to keep tracks without detection"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Features:**
    - YOLOv8n person detection
    - IoU-based tracking
    - Unique person counting
    - Real-time visualization
    - Analytics & statistics
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Upload")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video containing people"
        )
    
    with col2:
        st.subheader("📊 Quick Stats")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            unique_metric = st.empty()
            unique_metric.metric("Unique People", "—")
        with metric_col2:
            status_metric = st.empty()
            status_metric.metric("Status", "Waiting")
    
    # Process video
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display original video info
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Show original video
        with st.expander("🎬 View Original Video"):
            st.video(video_path)
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            status_metric.metric("Status", "Processing")
            
            # Load model
            with st.spinner("Loading YOLOv8n model..."):
                model = load_model()
            
            st.success("✅ Model loaded successfully!")
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            start_time = time.time()
            
            output_path, unique_count, stats = process_video(
                video_path, 
                model, 
                iou_threshold, 
                conf_threshold, 
                max_frames_skip,
                progress_bar,
                status_text
            )
            
            processing_time = time.time() - start_time
            
            if output_path:
                # Update metrics
                unique_metric.metric("Unique People", unique_count)
                status_metric.metric("Status", "Complete ✅")
                
                # Show results
                st.success(f"✅ Processing complete in {processing_time:.2f} seconds!")
                
                # Results tabs
                tab1, tab2, tab3 = st.tabs(["📹 Output Video", "📊 Analytics", "📈 Charts"])
                
                with tab1:
                    st.subheader("Tracked Video Output")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Tracked Video",
                            data=f.read(),
                            file_name="tracked_output.mp4",
                            mime="video/mp4"
                        )
                
                with tab2:
                    st.subheader("Detection Statistics")
                    
                    people_per_frame, unique_per_frame, total_frames = stats
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Unique People", unique_count)
                    with col_b:
                        st.metric("Total Frames", total_frames)
                    with col_c:
                        avg_people = np.mean(people_per_frame)
                        st.metric("Avg People/Frame", f"{avg_people:.1f}")
                    
                    col_d, col_e, col_f = st.columns(3)
                    with col_d:
                        max_people = max(people_per_frame)
                        st.metric("Peak People", max_people)
                    with col_e:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with col_f:
                        fps_processed = total_frames / processing_time
                        st.metric("Processing FPS", f"{fps_processed:.1f}")
                
                with tab3:
                    st.subheader("Tracking Analytics")
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Frame': range(1, len(people_per_frame) + 1),
                        'People in Frame': people_per_frame,
                        'Cumulative Unique': unique_per_frame
                    })
                    
                    # Plotly chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Frame'], 
                        y=df['People in Frame'],
                        mode='lines',
                        name='People per Frame',
                        line=dict(color='#2ecc71', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['Frame'], 
                        y=df['Cumulative Unique'],
                        mode='lines',
                        name='Cumulative Unique People',
                        line=dict(color='#3498db', width=2)
                    ))
                    
                    fig.update_layout(
                        title="People Detection Over Time",
                        xaxis_title="Frame Number",
                        yaxis_title="Count",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw data
                    with st.expander("📋 View Raw Data"):
                        st.dataframe(df, use_container_width=True)
    
    else:
        st.info("👆 Please upload a video file to get started")
        
        # Example section
        st.markdown("---")
        st.subheader("📝 How to Use")
        st.markdown("""
        1. **Upload Video**: Choose a video file containing people
        2. **Adjust Settings**: Use the sidebar to fine-tune detection parameters
        3. **Process**: Click "Start Processing" to run detection and tracking
        4. **View Results**: Check the output video and analytics
        5. **Download**: Save the tracked video for later use
        
        **Tips:**
        - Higher IoU threshold = stricter track matching
        - Higher confidence threshold = fewer false positives
        - Increase max frames to skip for crowded scenes
        """)


if __name__ == "__main__":
    main()