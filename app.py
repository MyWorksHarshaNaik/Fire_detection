import cv2
import numpy as np
import streamlit as st

# Define color ranges for fire and smoke detection
fire_lower_range = np.array([0, 50, 50])  # Adjusted lower range for fire detection
fire_upper_range = np.array([35, 255, 255])  # Adjusted upper range for fire detection

smoke_lower_range = np.array([0, 0, 50])  # Adjusted lower range for smoke detection
smoke_upper_range = np.array([180, 50, 200])  # Adjusted upper range for smoke detection

# Function to detect fire and smoke in an image
def detect_fire_and_smoke(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for fire and smoke detection
    fire_mask = cv2.inRange(hsv, fire_lower_range, fire_upper_range)
    smoke_mask = cv2.inRange(hsv, smoke_lower_range, smoke_upper_range)
    
    # Fire detection contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Only draw contours for significant areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Smoke detection contours
    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Only draw contours for significant areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image

# Streamlit application
st.title("Fire and Smoke Detection")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        detected_image = detect_fire_and_smoke(image)
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Fire and Smoke")
    else:
        st.error("Error reading the uploaded image.")
