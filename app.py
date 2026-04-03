"""
AI-Based Security System - Streamlit Cloud Compatible
Uses browser camera instead of server-side webcam
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
from config import UNKNOWN_FACE_THRESHOLD, KNOWN_PERSON_COLOR, UNKNOWN_PERSON_COLOR
from utils import FaceEncoder, FaceDetector, create_known_faces_directory
from alert_system import AlertSystem
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Security System",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.face_encoder = None
    st.session_state.face_detector = None
    st.session_state.alert_system = None
    st.session_state.detection_history = []
    st.session_state.last_alert_time = 0

# Title
st.title("🛡️ AI-Based Security System")
st.markdown("Real-time face detection and recognition using your browser's camera")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Known faces status
    st.subheader("📁 Known Faces Database")
    create_known_faces_directory()
    
    known_faces_path = Path("known_faces")
    if known_faces_path.exists():
        persons = [d.name for d in known_faces_path.iterdir() if d.is_dir()]
        if persons:
            st.success(f"✅ Loaded {len(persons)} known person(s): {', '.join(persons)}")
        else:
            st.warning("⚠️ No known faces found. Add images to 'known_faces/' directory")
    
    # Threshold setting
    threshold = st.slider(
        "Face Recognition Threshold",
        min_value=0.3,
        max_value=0.8,
        value=UNKNOWN_FACE_THRESHOLD,
        step=0.01,
        help="Lower = stricter matching, Higher = more permissive"
    )
    
    # Alert cooldown
    cooldown = st.number_input("Alert Cooldown (seconds)", min_value=1, max_value=30, value=5)
    
    # Reset button
    if st.button("🔄 Reset Session"):
        st.session_state.detection_history = []
        st.session_state.last_alert_time = 0
        st.rerun()

# Initialize system
@st.cache_resource
def initialize_system():
    """Initialize face recognition components (cached for performance)"""
    face_encoder = FaceEncoder()
    face_detector = FaceDetector()
    alert_system = AlertSystem()
    
    # Load known faces
    face_encoder.load_known_faces()
    
    return face_encoder, face_detector, alert_system

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Camera input from browser
    camera_image = st.camera_input("Position your face in frame", key="security_camera")
    
    if camera_image is not None:
        # Convert uploaded image to OpenCV format
        bytes_data = camera_image.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Initialize system if not already
            if not st.session_state.system_initialized:
                face_encoder, face_detector, alert_system = initialize_system()
                st.session_state.face_encoder = face_encoder
                st.session_state.face_detector = face_detector
                st.session_state.alert_system = alert_system
                st.session_state.system_initialized = True
            
            # Process frame for face detection
            face_locations = st.session_state.face_detector.detect_faces(frame)
            frame_display = frame.copy()
            detections_this_frame = []
            
            if face_locations:
                face_encodings = st.session_state.face_detector.get_face_encodings(frame, face_locations)
                
                for face_location, face_encoding in zip(face_locations, face_encodings):
                    name, distance, is_known = st.session_state.face_encoder.recognize_face(
                        face_encoding, threshold=threshold
                    )
                    
                    color = KNOWN_PERSON_COLOR if is_known else UNKNOWN_PERSON_COLOR
                    label = f"{name} ({distance:.2f})" if is_known else f"UNKNOWN ({distance:.2f})"
                    
                    # Draw bounding box
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame_display, (left, top), (right, bottom), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame_display, (left, top - label_size[1] - 10), 
                                 (left + label_size[0], top), color, -1)
                    cv2.putText(frame_display, label, (left, top - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Handle unknown person alert
                    current_time = time.time()
                    if not is_known and (current_time - st.session_state.last_alert_time) > cooldown:
                        st.session_state.alert_system.trigger_alert(name, distance)
                        st.session_state.last_alert_time = current_time
                        st.warning(f"🚨 INTRUSION ALERT: Unknown person detected! (Confidence: {distance:.2f})")
                    
                    # Log detection
                    detections_this_frame.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "name": name,
                        "confidence": distance,
                        "status": "KNOWN" if is_known else "UNKNOWN"
                    })
            
            # Update detection history
            if detections_this_frame:
                st.session_state.detection_history.extend(detections_this_frame)
                # Keep last 100 records
                if len(st.session_state.detection_history) > 100:
                    st.session_state.detection_history = st.session_state.detection_history[-100:]
            
            # Display processed frame
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Processed Feed", use_container_width=True)

with col2:
    st.subheader("📊 Live Statistics")
    
    # Stats cards
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        total_detections = len(df)
        unknown_count = len(df[df['status'] == 'UNKNOWN'])
        known_count = len(df[df['status'] == 'KNOWN'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Detections", total_detections)
            st.metric("Known Persons", known_count)
        with col_b:
            st.metric("Unknown Persons", unknown_count)
            st.metric("Alert Status", "🔴 Active" if unknown_count > 0 else "🟢 Idle")
        
        # Recent detections table
        st.subheader("📋 Recent Detections")
        recent_df = df.tail(10).sort_values('timestamp', ascending=False)
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No detections yet. Position yourself in front of the camera.")
    
    # Instructions
    st.markdown("---")
    st.subheader("📖 Instructions")
    st.markdown("""
    1. **Add known faces** to the `known_faces/` directory
       - Create folders: `known_faces/PersonName/`
       - Add 2-5 clear face photos per person
    2. **Position yourself** in front of the camera
    3. **Known persons** will show with 🟢 GREEN boxes
    4. **Unknown persons** will show with 🔴 RED boxes and trigger alerts
    """)

# Footer
st.markdown("---")
st.caption("🛡️ AI Security System | Face Recognition | Real-time Detection")
