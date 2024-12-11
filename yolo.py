import streamlit as st
import numpy as np
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
import torch  # PyTorch for YOLOv5
import tensorflow as tf  # TensorFlow for Keras model
from io import BytesIO

# Function to load YOLOv5 model
def load_yolov5_model():
    # Load the YOLOv5 model (make sure to use your model's file path)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/yolov5s.pt')  # Adjust the path as needed
    return model



# Streamlit app
st.title("Spectral Image Preprocessing and Detect the Area of Interest")

# Sidebar for upload and download buttons
with st.sidebar:
    st.header("Upload Images")
    # Upload the high-resolution image
    high_res_file = st.file_uploader("Upload a high-resolution image (.tiff)", type=["tiff", "tif"])
    # Upload the low-resolution spectral image
    low_res_file = st.file_uploader("Upload a low-resolution spectral image (.tiff)", type=["tiff", "tif"])

if high_res_file and low_res_file:
    try:
        # Load the high-resolution image
        high_res_image = tiff.imread(high_res_file)
        st.write("High-res image successfully loaded.")
        st.write(f"High-res Image Shape: {high_res_image.shape}")

        # Convert the high-resolution image to PIL format for detection
        high_res_image_pil = Image.fromarray(high_res_image)
        high_res_image_np = np.array(high_res_image_pil)

        # Load the low-resolution spectral image
        spectral_image = tiff.imread(low_res_file)
        st.write("Low-res spectral image successfully loaded.")
        st.write(f"Low-res Image Shape: {spectral_image.shape}")

        # Add a slider to change the band/wavelength of the original spectral image
        wavelengths = np.arange(450, 954, 4)  # Generate wavelengths from 450 to 950 with 4nm interval
        band_slider = st.slider('Select Wavelength (nm)', min_value=int(wavelengths.min()), max_value=int(wavelengths.max()), value=int(wavelengths[0]), step=4)
        band_index = np.where(wavelengths == band_slider)[0][0]
        selected_band_image = spectral_image[:, :, band_index]

        # Load the YOLOv5 model
        yolov5_model = load_yolov5_model()

        # Perform object detection using the YOLOv5 model
        results = yolov5_model(high_res_image_np)

        # Extract coordinates of the detected objects
        if len(results.xyxy[0]) > 0:
            st.write("Detected objects in the high-resolution image:")

            # Display detected objects inline
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(high_res_image_np)
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
            ax.set_title("Detected Objects in High-Resolution Image")
            st.pyplot(fig)

            # Assuming the first detected object is the pot
            x1, y1, x2, y2, conf, cls = results.xyxy[0][0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            st.write(f"Coordinates of the detected pot: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Reduce the size of the bounding box further by 80 pixels on each side
            reduction = 80
            x1 = max(0, x1 + reduction)
            y1 = max(0, y1 + reduction)
            x2 = max(0, x2 - reduction)
            y2 = max(0, y2 - reduction)
            st.write(f"Further reduced bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Crop the detected region from the high-resolution image
            cropped_high_res_image = high_res_image_pil.crop((x1, y1, x2, y2))

            # Map the coordinates to the low-resolution spectral image
            scale_x = spectral_image.shape[1] / high_res_image_np.shape[1]
            scale_y = spectral_image.shape[0] / high_res_image_np.shape[0]
            low_res_x1 = int(x1 * scale_x)
            low_res_y1 = int(y1 * scale_y)
            low_res_x2 = int(x2 * scale_x)
            low_res_y2 = int(y2 * scale_y)

            st.write(f"Mapped coordinates in low-resolution image: x1={low_res_x1}, y1={low_res_y1}, x2={low_res_x2}, y2={low_res_y2}")

            # Crop the corresponding region from the low-resolution spectral image
            cropped_spectral_image = spectral_image[low_res_y1:low_res_y2, low_res_x1:low_res_x2, :]
            st.write("Cropped spectral image shape:", cropped_spectral_image.shape)

            # Display images in a 2x2 grid
            col1, col2 = st.columns(2)

            with col1:
                st.image(high_res_image, caption="Original High-Resolution Image", use_container_width=True)
                st.image(cropped_high_res_image, caption="Cropped High-Resolution Image", use_container_width=True)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))  # Adjusting
