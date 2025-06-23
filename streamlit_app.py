"""
Streamlit App for Space Station Object Detection
This app loads a trained YOLOv8 model and allows users to upload images for inference.
"""

import streamlit as st
import os
from pathlib import Path
import cv2
import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import glob

# Set page config
st.set_page_config(
    page_title="Space Station Object Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0066cc;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #0066cc;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0066cc;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Find best model path
def find_model_path():
    """Find the best trained model in common locations."""
    model_paths = [
        Path("runs/exp_space/weights/best.pt"),
        Path("runs/train/weights/best.pt"),
        Path("weights/best.pt"),
        Path("yolov8s.pt")  # Fallback to pretrained model
    ]
    
    for path in model_paths:
        if path.exists():
            return str(path)
    
    # If no model found, return None
    return None

# Load model
@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load class names from config
def load_class_names():
    """Load class names from config file."""
    config_path = Path("yolo_params.yaml")
    
    # Default class names
    default_names = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
    
    if not config_path.exists():
        return default_names
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'names' in config and config['names']:
            return config['names']
    except Exception as e:
        st.sidebar.warning(f"Error loading class names: {e}")
    
    return default_names

# Process image
def process_image(image, model, conf_threshold=0.25):
    """Process an image through the model and return results."""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Make a copy of the original for side-by-side comparison
    original_img = img_array.copy()
    
    # Run inference
    results = model.predict(img_array, conf=conf_threshold)
    result = results[0]
    
    # Plot the image with detections
    plotted_img = result.plot()
    
    # Convert back to PIL
    return Image.fromarray(original_img), Image.fromarray(plotted_img), result

# Main app layout
def main():
    st.markdown("<h1 class='main-header'>Space Station Object Detection</h1>", unsafe_allow_html=True)
    
    # Sidebar content
    st.sidebar.title("🚀 Model Controls")
    
    # Find and load model
    model_path = find_model_path()
    if not model_path:
        st.error("No model found! Please train a model first or place a model in the 'weights' directory.")
        st.stop()
    
    # Display model info
    st.sidebar.info(f"Using model: {os.path.basename(model_path)}")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        st.stop()
    
    # Load class names
    class_names = load_class_names()
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    # About section
    with st.sidebar.expander("About"):
        st.write("""
        This app uses a YOLOv8 model trained on space station objects.
        Upload an image to detect:
        - Fire Extinguishers
        - Tool Boxes
        - Oxygen Tanks
        
        This model can be updated with new synthetic data to improve performance.
        """)
    
    # Falcon update plan
    with st.sidebar.expander("Falcon Update Plan"):
        st.write("""
        ### Falcon Model Update Process
        
        1. **Generate Synthetic Data**:
           - Create simulated images of space station interiors
           - Include objects in varied lighting and angles
        
        2. **Semi-Supervised Learning**:
           - Use current model to label new images
           - Experts review and correct labels
        
        3. **Continuous Training**:
           - Fine-tune model with new data
           - Validate on real mission images
           - Deploy updated model to stations
        """)
    
    # Main content area - image upload
    st.markdown("<div class='highlight'>Upload an image to detect objects in space station environments.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Create two columns for results
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Display progress
        with st.spinner("Processing image..."):
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image
            original_img, processed_img, result = process_image(image, model, conf_threshold)
            
            # Show images side by side
            with col1:
                st.markdown("<h3 class='sub-header'>Original Image</h3>", unsafe_allow_html=True)
                st.image(original_img, caption="Original Image", use_column_width=True)
            
            with col2:
                st.markdown("<h3 class='sub-header'>Detection Result</h3>", unsafe_allow_html=True)
                st.image(processed_img, caption="Processed Image", use_column_width=True)
        
        # Show detection details
        st.markdown("<h3 class='sub-header'>Detections</h3>", unsafe_allow_html=True)
        
        boxes = result.boxes
        if len(boxes) == 0:
            st.info("No objects detected in this image.")
        else:
            # Create detection table
            detections_data = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detections_data.append({
                    "Class": class_name,
                    "Confidence": f"{conf:.2f}",
                    "Position": f"({x1}, {y1}) to ({x2}, {y2})"
                })
            
            st.table(pd.DataFrame(detections_data))
            
            # Display summary stats
            st.markdown("<h3 class='sub-header'>Summary</h3>", unsafe_allow_html=True)
            
            # Create columns for metrics display
            metric_cols = st.columns(len(class_names) + 1)
            
            # Calculate class counts
            class_counts = {}
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Display total count
            with metric_cols[0]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{len(boxes)}</div>"
                    f"<div class='metric-label'>Total Objects</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            # Display individual class counts
            for i, class_name in enumerate(class_names, 1):
                if i < len(metric_cols):
                    with metric_cols[i]:
                        count = class_counts.get(class_name, 0)
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div class='metric-value'>{count}</div>"
                            f"<div class='metric-label'>{class_name}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
    else:
        # Example images section
        st.subheader("Example Images")
        st.write("Don't have an image? Try one of these examples:")
        
        # Find example images
        example_dir = Path("data/test/images")
        if example_dir.exists():
            example_images = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
            if example_images:
                # Display 3 example images in a row
                example_cols = st.columns(3)
                for i, img_path in enumerate(example_images[:3]):
                    with example_cols[i % 3]:
                        st.image(str(img_path), width=150, caption=f"Example {i+1}")
                        if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                            # Process the example image
                            image = Image.open(img_path)
                            original_img, processed_img, result = process_image(image, model, conf_threshold)
                            
                            # Show images side by side
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(original_img, caption="Original Image", use_column_width=True)
                            with col2:
                                st.image(processed_img, caption="Detection Result", use_column_width=True)
        else:
            st.info("Example images not available. Please upload an image to get started.")
    
    # Footer
    st.markdown(
        "<div class='footer'>Space Station Object Detection Dashboard | Powered by YOLOv8 | © 2025</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
