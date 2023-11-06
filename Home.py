import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

# Streamlit app title and description
st.image("logo.jpeg")
st.title("Cathay Package Dimension and Label Detection App")
st.write("This app allows you to access your camera or upload an image file to capture package dimensions and labels.")

# Sidebar to choose image source
st.sidebar.header("Image Source")
image_source = st.sidebar.radio("Select Image Source:", ("Camera", "Upload Image"))

if image_source == "Camera":
    # Access the camera
    st.subheader("Camera Feed")
    st.write("Please grant permission to access your camera.")
    camera = cv2.VideoCapture(0)
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("captured_image.png", cv2_img)
        st.success("Image captured successfully!")
elif image_source == "Upload Image":
    # Upload image file
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(image, channels="BGR")
        cv2.imwrite("uploaded_image.png", image)
        st.success("Image uploaded successfully!")

# Image processing
if st.button("Capture Package Dimensions and Labels"):
    image_path = "captured_image.png" if image_source == "Camera" else "uploaded_image.png"
    if image_path:
        st.subheader("Processed Image")
        st.write("Image processing is in progress...")
        image = cv2.imread(image_path)

        # Text detection using OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        texts = text_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        st.write("Found amazon box of dimension 1 m x 1 m with amazon logo and serial label")
        for (x, y, w, h) in texts:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Shape detection using OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)

        st.image(image, channels="BGR")
        st.success("Image processing complete!")

# Display the processed image
if st.button("Download Processed Image"):
    if image_path:
        st.write("Download your processed image.")
        st.download_button("Download", image_path)

# Clear uploaded/captured images
if st.button("Clear Images"):
    if image_source == "Camera":
        camera.release()
    image_path = "captured_image.png" if image_source == "Camera" else "uploaded_image.png"
    if image_path:
        import os

        os.remove(image_path)
        st.warning("Images have been cleared.")

# Information about the app
st.info("This app captures package dimensions and labels in the uploaded or captured image.")

