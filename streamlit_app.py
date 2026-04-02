import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np

# IMPORTANT: remove localhost backend dependency

# API_URL removed for cloud deployment

st.set_page_config(page_title="AI Security Dashboard", layout="wide")

st.title("🛡️ AI Security System - Live WebRTC Dashboard")

def video_frame_callback(frame):
img = frame.to_ndarray(format="bgr24")

# Basic processing (safe for deployment)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
st.info("Backend API disabled for cloud deployment")

st.markdown("---")
st.write("System Status: Running")
st.write("Camera: Active")

st.sidebar.markdown("### Deployment Info")
st.sidebar.info("Click START on camera for live streaming.")
