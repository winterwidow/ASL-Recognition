import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the hand landmark model once
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


def extract_hand_landmarks(image):
    """Extract 21 hand landmarks from an image using MediaPipe Tasks API."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        coords = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
        return np.array(coords).flatten()
    return None
