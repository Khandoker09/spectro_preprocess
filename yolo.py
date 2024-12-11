import streamlit as st
import numpy as np
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
import torch
import urllib.request
from io import BytesIO

# Function to download YOLOv5 model from the provided URL
def download_yolov5_model(model_url, model_path):
    urllib.request.urlretrieve(model_url, model_path)
    st.write(f"Model downloaded successfully to {model_path}")

# Function to load the YOLOv5 model
def load_yolo_model():
    model_url = "https://raw.githubusercontent.com/Khandoker09/spectro_preprocess/main/yolov5s.pt"  # URL to raw model file
    model_path = "yolov5s.pt"  # Local path to save the model
    download_yolov5_model(model_url, model_path)  # Download the model
    model = torch.load(model_path)  # Load the model
    model.eval()  # Set the model to evaluation mode
    return model

# Streamlit app
st.title("Spectral image preprocess and detect the area of interest")

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
        model = load_yolo_model()

        # Perform object detection with the high-resolution image
        results = model(high_res_image_np)

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
                fig, ax = plt.subplots(figsize=(10, 10))  # Adjusting figure size for better view
                cax = ax.imshow(selected_band_image, cmap='viridis', aspect='auto')
                fig.colorbar(cax)
                ax.set_title(f"Original Spectral Image (Wavelength {band_slider} nm)")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 10))  # Adjusting figure size for better view
                cax = ax.imshow(cropped_spectral_image[:, :, 0], cmap='viridis', aspect='auto')
                fig.colorbar(cax)
                ax.set_title("Heatmap of the Cropped Spectral Image (First Band)")
                st.pyplot(fig)

            with st.sidebar:
                st.header("Download Images")

                if 'cropped_high_res_image' in locals() and 'cropped_spectral_image' in locals():
                    # Prepare and download the cropped high-resolution image
                    buffer_high_res = BytesIO()
                    cropped_high_res_image.save(buffer_high_res, format="TIFF")
                    buffer_high_res.seek(0)

                    # Prepare and download the cropped spectral image
                    buffer_spectral = BytesIO()
                    tiff.imwrite(buffer_spectral, cropped_spectral_image)
                    buffer_spectral.seek(0)

                    # Enforce uniform button sizes using columns (Streamlit native layout)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Cropped High-Res Image",
                            data=buffer_high_res,
                            file_name="cropped_high_res_image.tiff",
                        )
                    with col2:
                        st.download_button(
                            label="Download Cropped Spectral Image",
                            data=buffer_spectral,
                            file_name="cropped_spectral_image.tiff",
                        )
                else:
                    st.warning("Images have not been processed yet. Upload and process the images first.")

    except Exception as e:
        st.error(f"Error processing the images: {e}")
