"""
Streamlit App for Space Station Object Detection
This app loads a trained YOLOv8 model and allows users to upload images for inference.
It also includes an explanation of how Falcon can update the model with new synthetic data.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import json
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

# Paths
@st.cache_resource
def get_paths():
    script_dir = Path(__file__).parent
    return {
        "script_dir": script_dir,
        "config_path": script_dir / "yolo_params.yaml",
        "runs_dir": script_dir / "runs",
        "test_images_dir": script_dir / "data" / "test" / "images"
    }

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get class names
@st.cache_data
def get_class_names(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('names', ['Unknown'])
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return ['Unknown']

# Find available models
@st.cache_data
def find_models(runs_dir):
    models = []
    
    # Look in runs/detect or any other folders that might contain model weights
    for exp_folder in glob.glob(str(runs_dir / "*" / "*")):
        exp_path = Path(exp_folder)
        weights_dir = exp_path / "weights"
        
        if weights_dir.exists():
            best_weights = weights_dir / "best.pt"
            if best_weights.exists():
                models.append({
                    "name": f"{exp_path.parent.name}/{exp_path.name} (best)",
                    "path": str(best_weights),
                    "folder": str(exp_path)
                })
            
            last_weights = weights_dir / "last.pt"
            if last_weights.exists():
                models.append({
                    "name": f"{exp_path.parent.name}/{exp_path.name} (last)",
                    "path": str(last_weights),
                    "folder": str(exp_path)
                })
    
    # If no models found in standard folders, try direct search for .pt files
    if not models:
        for weights_file in glob.glob(str(runs_dir / "**" / "*.pt"), recursive=True):
            weight_path = Path(weights_file)
            models.append({
                "name": f"{weight_path.stem}",
                "path": str(weight_path),
                "folder": str(weight_path.parent)
            })
    
    return models

# Load performance metrics if available
@st.cache_data
def load_metrics(model_folder):
    try:
        metrics_path = Path(model_folder) / "results.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            return metrics_df
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
        return None

# Process image
def process_image(model, image, conf_threshold=0.25):
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

# Generate confusion matrix visualization
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return plt

# Create metrics visualization
def plot_metrics(metrics_df):
    if metrics_df is None:
        return None
    
    # Extract key metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot precision
    if 'precision' in metrics_df.columns:
        axes[0, 0].plot(metrics_df['epoch'], metrics_df['precision'], 'b-')
        axes[0, 0].set_title('Precision')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].grid(True)
    
    # Plot recall
    if 'recall' in metrics_df.columns:
        axes[0, 1].plot(metrics_df['epoch'], metrics_df['recall'], 'g-')
        axes[0, 1].set_title('Recall')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].grid(True)
    
    # Plot mAP50
    if 'mAP50' in metrics_df.columns:
        axes[1, 0].plot(metrics_df['epoch'], metrics_df['mAP50'], 'r-')
        axes[1, 0].set_title('mAP@0.5')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP@0.5')
        axes[1, 0].grid(True)
    
    # Plot loss
    if 'loss' in metrics_df.columns:
        axes[1, 1].plot(metrics_df['epoch'], metrics_df['loss'], 'k-')
        axes[1, 1].set_title('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

# Get test images
def get_test_images(test_dir):
    images = []
    supported_formats = ['.jpg', '.jpeg', '.png']
    
    if test_dir.exists():
        for img_path in test_dir.glob('*'):
            if img_path.suffix.lower() in supported_formats:
                images.append(str(img_path))
    
    return images

# Create a table of failure cases
def create_failure_cases_table():
    # This would typically be loaded from a file, but we'll hardcode some examples
    failure_cases = [
        {
            "object": "OxygenTank", 
            "scenario": "Poor lighting conditions",
            "solution": "Add more synthetic data with varied lighting"
        },
        {
            "object": "ToolBox", 
            "scenario": "Partial occlusion by astronaut",
            "solution": "Generate more training examples with occlusion"
        },
        {
            "object": "FireExtinguisher", 
            "scenario": "Unusual viewing angle",
            "solution": "Augment dataset with more viewing angles"
        },
        {
            "object": "OxygenTank", 
            "scenario": "Confused with similar cylindrical objects",
            "solution": "Add more negative examples of similar objects"
        },
        {
            "object": "All objects", 
            "scenario": "Crowded scene with multiple objects",
            "solution": "Generate more complex scenes with multiple objects"
        }
    ]
    
    return pd.DataFrame(failure_cases)

# Sidebar - Model selection and settings
def create_sidebar():
    paths = get_paths()
    
    st.sidebar.title("🚀 Space Station Detector")
    st.sidebar.markdown("---")
    
    # Add navigation
    page = st.sidebar.radio("Navigation", ["Home", "Model Performance", "Error Analysis", "Falcon Integration"])
    
    # Model selection
    st.sidebar.header("Model Selection")
    
    models = find_models(paths["runs_dir"])
    
    if not models:
        st.sidebar.error("No trained models found. Please train a model first.")
        selected_model = None
    else:
        model_names = [model["name"] for model in models]
        selected_model_name = st.sidebar.selectbox("Select a model", model_names)
        selected_model = next((model for model in models if model["name"] == selected_model_name), None)
    
    # Settings
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses YOLOv8 to detect objects in a space station environment. "
        "Upload an image to see the model in action!"
    )
    
    # Add date and version info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version:** 1.0.0  \n**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    return page, selected_model, conf_threshold

# Home page content
def show_home_page(selected_model, conf_threshold):
    paths = get_paths()
    
    st.markdown("<h1 class='main-header'>Space Station Object Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='highlight'>Upload an image to detect objects in space station environments.</p>", unsafe_allow_html=True)
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Two columns for results
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Check if model is selected
        if selected_model is None:
            st.error("Please select a model from the sidebar first.")
            return
        
        # Load model
        model = load_model(selected_model["path"])
        if model is None:
            return
        
        # Get class names
        class_names = get_class_names(paths["config_path"])
        
        # Display progress
        with st.spinner("Processing image..."):
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image
            original_img, processed_img, result = process_image(model, image, conf_threshold)
            
            # Show original and processed images
            with col1:
                st.markdown("<h3 class='sub-header'>Original Image</h3>", unsafe_allow_html=True)
                st.image(original_img, caption="Original Image", use_column_width=True)
            
            with col2:
                st.markdown("<h3 class='sub-header'>Detection Result</h3>", unsafe_allow_html=True)
                st.image(processed_img, caption="Processed Image", use_column_width=True)
            
            # Show detection info
            st.markdown("<h3 class='sub-header'>Detections</h3>", unsafe_allow_html=True)
            
            # Get detection results
            boxes = result.boxes
            
            if len(boxes) == 0:
                st.info("No objects detected.")
            else:
                # Create a table of detections
                detections_data = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    detections_data.append({
                        "Class": class_name, 
                        "Confidence": conf,
                        "Position": f"({x1}, {y1}) to ({x2}, {y2})"
                    })
                
                # Display detection table
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
                
                # Create a bar chart of detected classes
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Prepare data for plotting
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                
                # Create bars with colors
                colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
                ax.bar(classes, counts, color=colors[:len(classes)])
                
                # Add count labels on top of bars
                for i, count in enumerate(counts):
                    ax.text(i, count + 0.1, str(count), ha='center', fontweight='bold')
                
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Detected Objects', fontsize=14, fontweight='bold')
                plt.xticks(rotation=0, fontsize=10)
                plt.tight_layout()
                
                st.pyplot(fig)
    else:
        # Display demo images if no upload
        st.info("Please upload an image to get started, or try one of our demo images below.")
        
        # Demo images section
        st.markdown("<h3 class='sub-header'>Try Demo Images</h3>", unsafe_allow_html=True)
        
        # Get test images
        test_images = get_test_images(paths["test_images_dir"])
        
        if not test_images:
            st.warning("No demo images found in the test directory.")
            return
        
        # Select a subset of test images for demo
        demo_images = test_images[:6]  # Limit to 6 images
        
        # Create a grid of images
        demo_cols = st.columns(3)
        
        # Display demo buttons
        for i, demo_path in enumerate(demo_images):
            with demo_cols[i % 3]:
                try:
                    demo_img = Image.open(demo_path)
                    st.image(demo_img, caption=f"Demo {i+1}", width=200)
                    if st.button(f"Try Demo {i+1}"):
                        # Check if model is selected
                        if selected_model is None:
                            st.error("Please select a model from the sidebar first.")
                            return
                        
                        # Load model
                        model = load_model(selected_model["path"])
                        if model is None:
                            return
                        
                        # Get class names
                        class_names = get_class_names(paths["config_path"])
                        
                        # Process demo image
                        with st.spinner("Processing demo image..."):
                            original_img, processed_img, result = process_image(model, demo_img, conf_threshold)
                            
                            # Show original and processed images
                            with col1:
                                st.markdown("<h3 class='sub-header'>Original Image</h3>", unsafe_allow_html=True)
                                st.image(original_img, caption="Original Image", use_column_width=True)
                            
                            with col2:
                                st.markdown("<h3 class='sub-header'>Detection Result</h3>", unsafe_allow_html=True)
                                st.image(processed_img, caption="Processed Image", use_column_width=True)
                            
                            # Show detection info
                            st.markdown("<h3 class='sub-header'>Detections</h3>", unsafe_allow_html=True)
                            
                            # Get detection results
                            boxes = result.boxes
                            
                            if len(boxes) == 0:
                                st.info("No objects detected.")
                            else:
                                # Create a table of detections
                                detections_data = []
                                for box in boxes:
                                    cls_id = int(box.cls[0].item())
                                    conf = float(box.conf[0].item())
                                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                                    
                                    # Get bounding box coordinates
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    
                                    detections_data.append({
                                        "Class": class_name, 
                                        "Confidence": conf,
                                        "Position": f"({x1}, {y1}) to ({x2}, {y2})"
                                    })
                                
                                # Display detection table
                                st.table(pd.DataFrame(detections_data))
                except Exception as e:
                    st.error(f"Error loading demo image: {e}")

# Model performance page content
def show_model_performance_page(selected_model):
    if selected_model is None:
        st.error("Please select a model from the sidebar first.")
        return
    
    st.markdown("<h1 class='main-header'>Model Performance Analysis</h1>", unsafe_allow_html=True)
    
    # Load metrics if available
    metrics_df = load_metrics(selected_model["folder"])
    
    if metrics_df is not None:
        # Display key metrics
        st.markdown("<h3 class='sub-header'>Training Metrics</h3>", unsafe_allow_html=True)
        
        # Get the best epoch metrics
        best_map_idx = metrics_df['mAP50'].idxmax() if 'mAP50' in metrics_df.columns else None
        
        if best_map_idx is not None:
            best_epoch = metrics_df.loc[best_map_idx]
            
            # Create metric cards in columns
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{best_epoch['precision']:.3f}</div>"
                    f"<div class='metric-label'>Best Precision</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            
            with metric_cols[1]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{best_epoch['recall']:.3f}</div>"
                    f"<div class='metric-label'>Best Recall</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            
            with metric_cols[2]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{best_epoch['mAP50']:.3f}</div>"
                    f"<div class='metric-label'>Best mAP@0.5</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            
            with metric_cols[3]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{int(best_epoch['epoch'])}</div>"
                    f"<div class='metric-label'>Best Epoch</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
        
        # Plot metrics
        st.markdown("<h3 class='sub-header'>Metrics Over Time</h3>", unsafe_allow_html=True)
        metrics_fig = plot_metrics(metrics_df)
        if metrics_fig:
            st.pyplot(metrics_fig)
        
        # Display metrics table
        with st.expander("View Raw Metrics Data"):
            st.dataframe(metrics_df)
    else:
        st.warning("No metrics data found for this model. Run evaluation first to generate metrics.")
    
    # Display confusion matrix if available
    st.markdown("<h3 class='sub-header'>Confusion Matrix</h3>", unsafe_allow_html=True)
    
    # Check if confusion matrix exists (in a real app, you'd load this from a file)
    cm_exists = os.path.exists(os.path.join(selected_model["folder"], "confusion_matrix.png"))
    
    if cm_exists:
        cm_path = os.path.join(selected_model["folder"], "confusion_matrix.png")
        st.image(cm_path, caption="Confusion Matrix")
    else:
        # Create a sample confusion matrix for demonstration
        paths = get_paths()
        class_names = get_class_names(paths["config_path"])
        
        # Sample confusion matrix data
        cm = np.array([
            [45, 3, 2],
            [5, 38, 7],
            [2, 4, 34]
        ])
        
        cm_fig = plot_confusion_matrix(cm, class_names)
        st.pyplot(cm_fig)
        st.info("This is a sample confusion matrix for demonstration. Run evaluation to generate the actual matrix.")

# Error analysis page content
def show_error_analysis_page():
    st.markdown("<h1 class='main-header'>Error Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown(
        "<p class='highlight'>This page analyzes common failure cases and suggests improvements.</p>", 
        unsafe_allow_html=True
    )
    
    # Display failure cases table
    st.markdown("<h3 class='sub-header'>Common Failure Cases</h3>", unsafe_allow_html=True)
    failure_df = create_failure_cases_table()
    st.table(failure_df)
    
    # Display sample failure images (would typically load from evaluation results)
    st.markdown("<h3 class='sub-header'>Sample Failure Examples</h3>", unsafe_allow_html=True)
    
    # Create sample descriptions
    failure_descriptions = [
        {
            "title": "Poor Lighting Conditions",
            "description": "The model struggles with dark or unevenly lit areas of the space station.",
            "solution": "Train with more images in varied lighting conditions."
        },
        {
            "title": "Occlusion",
            "description": "Objects partially hidden behind astronauts or other equipment are often missed.",
            "solution": "Add more training examples with partially occluded objects."
        },
        {
            "title": "Similar Objects",
            "description": "Oxygen tanks are sometimes confused with other cylindrical objects.",
            "solution": "Add more negative examples and fine-tune the model on confusing cases."
        }
    ]
    
    # Display failure examples
    for i, failure in enumerate(failure_descriptions):
        st.markdown(f"#### {failure['title']}")
        st.markdown(f"**Issue:** {failure['description']}")
        st.markdown(f"**Solution:** {failure['solution']}")
        st.markdown("---")
    
    # Improvement suggestions
    st.markdown("<h3 class='sub-header'>Improvement Strategies</h3>", unsafe_allow_html=True)
    
    improvement_cols = st.columns(3)
    
    with improvement_cols[0]:
        st.markdown("#### Data Augmentation")
        st.markdown(
            "- Rotation and flipping\n"
            "- Brightness and contrast adjustments\n"
            "- Adding noise and blur\n"
            "- Mosaic augmentation"
        )
    
    with improvement_cols[1]:
        st.markdown("#### Model Tuning")
        st.markdown(
            "- Adjust learning rate\n"
            "- Try different backbone networks\n"
            "- Experiment with confidence thresholds\n"
            "- Use transfer learning"
        )
    
    with improvement_cols[2]:
        st.markdown("#### Dataset Expansion")
        st.markdown(
            "- Add synthetic images\n"
            "- Include more varied scenarios\n"
            "- Balance class distribution\n"
            "- Add hard negative examples"
        )

# Falcon integration page content
def show_falcon_integration_page():
    st.markdown("<h1 class='main-header'>Falcon Synthetic Data Integration</h1>", unsafe_allow_html=True)
    
    st.markdown(
        "<p class='highlight'>Falcon can help keep this model updated with synthetic data generation.</p>", 
        unsafe_allow_html=True
    )
    
    # Overview of Falcon integration
    st.markdown("### How Falcon Can Improve the Model")
    
    st.markdown(
        "Falcon is an advanced synthetic data generation platform that can help maintain and improve "
        "the space station object detection model over time. Here's how it works:"
    )
    
    # Create columns for the process steps
    process_cols = st.columns(4)
    
    with process_cols[0]:
        st.markdown("#### 1. Generate Synthetic Data")
        st.markdown(
            "Falcon can create photorealistic 3D renderings of space station environments with:"
            "\n- Varied lighting conditions"
            "\n- Different camera angles"
            "\n- Diverse object placements"
            "\n- Realistic textures and materials"
        )
    
    with process_cols[1]:
        st.markdown("#### 2. Automatic Annotation")
        st.markdown(
            "The synthetic data comes with:"
            "\n- Precise bounding boxes"
            "\n- Perfect class labels"
            "\n- Depth information"
            "\n- Instance segmentation"
            "\n- No manual labeling needed"
        )
    
    with process_cols[2]:
        st.markdown("#### 3. Domain Adaptation")
        st.markdown(
            "Falcon bridges the sim-to-real gap with:"
            "\n- Style transfer techniques"
            "\n- Physics-based rendering"
            "\n- Noise and distortion modeling"
            "\n- Background randomization"
        )
    
    with process_cols[3]:
        st.markdown("#### 4. Continuous Learning")
        st.markdown(
            "The model improves through:"
            "\n- Automated retraining pipeline"
            "\n- Error-focused data generation"
            "\n- Performance monitoring"
            "\n- A/B testing of models"
        )
    
    # Benefits of Falcon integration
    st.markdown("### Benefits of Synthetic Data Integration")
    
    benefit_cols = st.columns(2)
    
    with benefit_cols[0]:
        st.markdown(
            "- **Cost Reduction**: No need for expensive real-world data collection"
            "\n- **Time Efficiency**: Generate thousands of images in minutes"
            "\n- **Edge Case Coverage**: Create rare but critical scenarios"
            "\n- **Perfect Annotations**: Eliminate human labeling errors"
            "\n- **Privacy Compliance**: No concerns with personal data"
        )
    
    with benefit_cols[1]:
        st.markdown(
            "- **Scalability**: Easily generate more data as needed"
            "\n- **Controlled Experimentation**: Test specific variables"
            "\n- **Rapid Iteration**: Quick feedback loop for model improvement"
            "\n- **Balanced Datasets**: Ensure equal representation of classes"
            "\n- **Safety**: Test dangerous scenarios without risk"
        )
    
    # Implementation timeline
    st.markdown("### Implementation Timeline")
    
    timeline_data = {
        "Phase": ["Initial Integration", "Pipeline Setup", "Automated Training", "Production Deployment"],
        "Duration": ["2 weeks", "1 month", "2 weeks", "1 week"],
        "Key Activities": [
            "Connect Falcon API, generate first synthetic dataset",
            "Build data validation, preprocessing and augmentation pipeline",
            "Implement continuous training with performance monitoring",
            "Deploy to production with A/B testing framework"
        ]
    }
    
    st.table(pd.DataFrame(timeline_data))
    
    # Sample integration code
    st.markdown("### Sample Integration Code")
    
    with st.expander("View Sample Code"):
        st.code("""
# Example Python code for Falcon API integration

import falcon_api
import os
from ultralytics import YOLO

# Initialize Falcon client
falcon_client = falcon_api.Client(api_key="YOUR_API_KEY")

# Define synthetic data parameters
data_params = {
    "environment": "space_station",
    "objects": ["FireExtinguisher", "ToolBox", "OxygenTank"],
    "lighting_conditions": ["normal", "dark", "emergency"],
    "camera_angles": ["front", "side", "top", "angled"],
    "occlusion_levels": [0.0, 0.2, 0.4],
    "num_images": 1000
}

# Generate synthetic data
synthetic_dataset = falcon_client.generate_dataset(data_params)

# Download dataset
synthetic_dataset.download("./data/synthetic")

# Combine with existing dataset
!python utils/dataset_merger.py --real_data ./data/train --synthetic_data ./data/synthetic --output ./data/combined

# Train model with combined dataset
model = YOLO("yolov8n.pt")
results = model.train(
    data="./data/combined/data.yaml",
    epochs=50,
    imgsz=640,
    project="runs/train",
    name="falcon_enhanced"
)

# Evaluate model
model.val(data="./data/test/data.yaml")
        """)
    
    # Call to action
    st.markdown("### Ready to Integrate?")
    st.info("Contact the Falcon team to set up a pilot project and see how synthetic data can improve your space station object detection model.")

# Main app
def main():
    # Create sidebar and get selection
    page, selected_model, conf_threshold = create_sidebar()
    
    # Show different pages based on selection
    if page == "Home":
        show_home_page(selected_model, conf_threshold)
    elif page == "Model Performance":
        show_model_performance_page(selected_model)
    elif page == "Error Analysis":
        show_error_analysis_page()
    elif page == "Falcon Integration":
        show_falcon_integration_page()
    
    # Footer
    st.markdown(
        "<div class='footer'>Space Station Object Detection Dashboard | Powered by YOLOv8 | © 2025</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
