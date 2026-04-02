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

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

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
        st.success("Sync command sent!")

    st.markdown("---")
    st.info("Teacher Demo: Use Start button to activate camera.")
