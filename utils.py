import cv2
import numpy as np
import os
from pathlib import Path

class FaceDetector:
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from the known_faces directory"""
        known_faces_path = Path("known_faces")
        if known_faces_path.exists():
            for person_dir in known_faces_path.iterdir():
                if person_dir.is_dir():
                    person_name = person_dir.name
                    for img_path in person_dir.glob("*.jpg"):
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Store face encoding (simplified - just store path for demo)
                            self.known_names.append(person_name)
    
    def detect_faces(self, image):
        """Detect faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def recognize_faces(self, image, faces):
        """Recognize detected faces (simplified version)"""
        recognitions = []
        for (x, y, w, h) in faces:
            # For demo, just mark as "Unknown"
            recognitions.append({
                "bbox": (x, y, w, h),
                "name": "Unknown",
                "confidence": 0.0
            })
        return recognitions
