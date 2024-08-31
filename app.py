# Python In-built packages
from pathlib import Path
import PIL
import cv2

# External packages
import streamlit as st
import numpy as np

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Coffee Leaf Classification and Disease Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define global color mappings for classes
cleaf_colors = {
    0: (0, 255, 0),    # Green for 'arabica'
    1: (0, 255, 255),  # Yellow for 'liberica'
    2: (255, 0, 0)     # Blue for 'robusta'
}

cdisease_colors = {
    0: (255, 165, 0),  # Orange for 'brown_eye_spot'
    1: (255, 0, 255),  # Magenta for 'leaf_miner'
    2: (0, 0, 255),    # Red for 'leaf_rust'
    3: (128, 0, 128)   # Purple for 'red_spider_mite'
}

# Main page heading
st.title("Coffee Leaf Classification and Disease Detection")

# Sidebar
st.sidebar.header("DL Model Config")

# Model Selection
detection_model_choice = st.sidebar.radio(
    "Select Detection Model", ['Disease Detection', 'Leaf Detection', 'Both Models'])

confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Model
if detection_model_choice == 'Disease Detection':
    model_path = Path(settings.DISEASE_DETECTION_MODEL)
    model = helper.load_model(model_path)
elif detection_model_choice == 'Leaf Detection':
    model_path = Path(settings.LEAF_DETECTION_MODEL)
    model = helper.load_model(model_path)
elif detection_model_choice == 'Both Models':
    model_disease = helper.load_model(Path(settings.DISEASE_DETECTION_MODEL))
    model_leaf = helper.load_model(Path(settings.LEAF_DETECTION_MODEL))

# Load Pre-trained ML Model
try:
    if detection_model_choice != 'Both Models':
        model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Commenting out Image/Video Config
# st.sidebar.header("Image/Video Config")
# source_radio = st.sidebar.radio(
#     "Select Source", settings.SOURCES_LIST)

source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

def draw_bounding_boxes(image, boxes, labels, colors):
    """Function to draw bounding boxes with labels and confidence on the image."""
    res_image = np.array(image)
    height, width, _ = res_image.shape  # Get image dimensions

    # Extract the coordinates, classes, and confidence levels from the 'Boxes' object
    box_coordinates = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    for idx, (x1, y1, x2, y2) in enumerate(box_coordinates):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label_idx = int(class_ids[idx])
        confidence = confidences[idx]
        label = f"{labels[label_idx]}: {confidence:.2f}"

        # Determine color based on class
        color = colors[label_idx]

        # Adjust font scale and thickness based on image size
        font_scale = max(0.6, min(width, height) / 600)
        font_thickness = max(2, min(width, height) // 250)
        box_thickness = max(3, min(width, height) // 200)

        # Draw the bounding box
        cv2.rectangle(res_image, (x1, y1), (x2, y2), color=color, thickness=box_thickness)
        cv2.putText(res_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return res_image


col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image",
                     use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                     use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image',
                 use_column_width=True)
    else:
        if st.sidebar.button('Detect Objects'):
            if detection_model_choice == 'Both Models':
                # Use both models for detection
                res_disease = model_disease.predict(uploaded_image, conf=confidence)
                res_leaf = model_leaf.predict(uploaded_image, conf=confidence)

                # Extract boxes and labels for both models
                disease_boxes = res_disease[0].boxes
                leaf_boxes = res_leaf[0].boxes

                # Merge the labels by converting them to dictionaries and concatenating
                combined_labels = {**res_disease[0].names, **res_leaf[0].names}

                # Create a combined image with both model detections
                res_combined = np.array(uploaded_image)

                # Draw disease boxes
                res_combined = draw_bounding_boxes(res_combined, disease_boxes, res_disease[0].names, cdisease_colors)

                # Draw leaf boxes
                res_combined = draw_bounding_boxes(res_combined, leaf_boxes, res_leaf[0].names, cleaf_colors)

                st.image(res_combined, caption='Combined Detected Image', use_column_width=True)

                with st.expander("Combined Detection Results"):
                    # Iterate over each box separately
                    st.write("Disease Detection Results:")
                    for box in disease_boxes:
                        st.write(box)

                    st.write("Leaf Detection Results:")
                    for box in leaf_boxes:
                        st.write(box)

            else:
                # Single model prediction
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                labels = res[0].names

                # Choose the appropriate color map based on the model
                if detection_model_choice == 'Disease Detection':
                    colors = cdisease_colors
                else:
                    colors = cleaf_colors

                # Draw the bounding boxes
                res_plotted = draw_bounding_boxes(uploaded_image, boxes, labels, colors)
                
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

