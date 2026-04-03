class Config:
    # Face Detection Settings
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 5
    FACE_DETECTION_MIN_SIZE = (30, 30)
    
    # Paths
    KNOWN_FACES_PATH = "known_faces"
    ALERT_LOG_PATH = "security_alerts.log"
    
    # Alert Settings
    ALERT_COOLDOWN_SECONDS = 5
    
    # Display Settings
    DISPLAY_FACE_RECTANGLE = True
    DISPLAY_FACE_NAME = True
    DISPLAY_CONFIDENCE = True
"""
Configuration settings for AI Security System
Optimized for Streamlit Cloud deployment
"""

import os
from pathlib import Path

# ==================== PATH CONFIGURATION ====================
BASE_DIR = Path(__file__).parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
ALERT_LOG_FILE = BASE_DIR / "security_alerts.log"

# ==================== FACE DETECTION SETTINGS ====================
# Face detection model: 'hog' (faster) or 'cnn' (more accurate, but slower)
# For cloud deployment, 'hog' is recommended
FACE_DETECTION_MODEL = "hog"

# Face recognition threshold (lower = stricter matching)
# Range: 0.4 to 0.8
UNKNOWN_FACE_THRESHOLD = 0.6

# Number of times to upsample the image for better detection
# Higher = better for small faces, but slower
UPSAMPLE_TIMES = 1

# ==================== DISPLAY SETTINGS ====================
# Colors (BGR format for OpenCV)
KNOWN_PERSON_COLOR = (0, 255, 0)      # Green for known persons
UNKNOWN_PERSON_COLOR = (0, 0, 255)    # Red for unknown persons
TEXT_COLOR = (255, 255, 255)           # White text

# Font settings
FONT_SIZE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# ==================== ALERT SETTINGS ====================
# Minimum time between alerts for the same unknown person (seconds)
ALERT_TRIGGER_COOLDOWN = 5

# Maximum number of alerts to keep in memory
MAX_ALERT_HISTORY = 100

# ==================== PERFORMANCE SETTINGS ====================
# Maximum number of faces to detect per frame
MAX_FACES_PER_FRAME = 5

# Image quality for encoding (1-100)
ENCODING_QUALITY = 90

# ==================== STREAMLIT SPECIFIC ====================
# Maximum image size for processing (width, height)
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 720

# Cache duration for face encodings (seconds)
ENCODING_CACHE_DURATION = 3600

# ==================== LOGGING SETTINGS ====================
LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create necessary directories
def ensure_directories():
    """Ensure all required directories exist"""
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    
    # Create a sample person directory if none exists
    if not any(KNOWN_FACES_DIR.iterdir()):
        sample_dir = KNOWN_FACES_DIR / "Sample_Person"
        sample_dir.mkdir(exist_ok=True)
        print("📁 Created sample directory: known_faces/Sample_Person/")
        print("   Add face images there for recognition")

# Run directory check
ensure_directories()
