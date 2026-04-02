import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import requests
import numpy as np

# API URL setup
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Security Dashboard", layout="wide")
st.title("🛡️ AI Security System - Live WebRTC Dashboard")

# --- FACE RECOGNITION CALLBACK ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Yahan frame return ho raha hai jo browser mein dikhega
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed (WebRTC)")
    webrtc_streamer(
        key="security-camera",
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 System Control & Stats")
    
    if st.button("🔄 Sync Known Faces"):
        try:
            res = requests.post(f"{API_URL}/reload_faces", timeout=5)
            if res.status_code == 200:
                st.success("Database Updated!")
            else:
                st.error("Backend Error")
        except:
            st.warning("Backend API is not reachable.")

    st.markdown("---")
    stats_placeholder = st.empty()
    try:
        r = requests.get(f"{API_URL}/stats", timeout=1)
        if r.status_code == 200:
            stats_placeholder.json(r.json())
        else:
            stats_placeholder.info("Waiting for statistics...")
    except:
        stats_placeholder.error("Cannot connect to Backend API.")

st.sidebar.markdown("### Deployment Info")
st.sidebar.info("Teacher Demo: Use the Start button on the camera feed for live streaming.")
