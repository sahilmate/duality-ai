#!/usr/bin/env python3
# filepath: HackByte_Dataset/predict.py
"""
YOLOv8 Prediction and Evaluation Script for Space Station Object Detection
This script loads a trained model and evaluates it on the test set.
"""
import os
from pathlib import Path
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
from ultralytics import YOLO


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def predict_and_save(model, image_path, output_path, output_path_txt, class_names):
    """Perform prediction on an image and save results."""
    # Perform prediction with 0.5 confidence threshold
    results = model.predict(image_path, conf=0.25)
    result = results[0]

    # Get original image for overlay
    img_orig = cv2.imread(str(image_path))

    # Draw boxes on the image with class names
    pred_img = result.plot(labels=True, conf=True)

    # Save the result
    cv2.imwrite(str(output_path), pred_img)

    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id, confidence and bounding box coordinates
            cls_id = int(box.cls)
            conf = float(box.conf)
            x_center, y_center, width, height = box.xywh[0].tolist()

            # Write bbox information
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            f.write(f"{cls_id} {conf:.4f} {x_center} {y_center} {width} {height} # {class_name}\n")

    return result


def create_report_folder(base_dir, run_name=None):
    """Create a folder for storing evaluation results."""
    if run_name is None:
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report_dir = base_dir / "reports" / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (report_dir / "images").mkdir(exist_ok=True)
    (report_dir / "labels").mkdir(exist_ok=True)
    (report_dir / "plots").mkdir(exist_ok=True)

    return report_dir


def save_confusion_matrix(conf_matrix, class_names, output_path):
    """Save confusion matrix as an image."""
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')

    # Add class names as labels
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(class_names)), class_names)

    # Add numeric values in cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, f"{conf_matrix[i, j]:.0f}",
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)
    plt.close()


def analyze_failures(results, class_names, output_dir):
    """Analyze failure cases and save examples."""
    failure_dir = output_dir / "failure_cases"
    failure_dir.mkdir(exist_ok=True)

    # Categories of failures
    low_conf_dir = failure_dir / "low_confidence"
    misclass_dir = failure_dir / "misclassifications"
    missed_dir = failure_dir / "missed_detections"

    low_conf_dir.mkdir(exist_ok=True)
    misclass_dir.mkdir(exist_ok=True)
    missed_dir.mkdir(exist_ok=True)

    # Add analysis text
    with open(failure_dir / "failure_analysis.md", 'w') as f:
        f.write("# Failure Analysis\n\n")
        f.write("## Categories of Failures\n\n")
        f.write("1. **Low Confidence Detections**: Detections with confidence < 0.7\n")
        f.write("2. **Misclassifications**: Incorrect class predictions\n")
        f.write("3. **Missed Detections**: Ground truth objects not detected\n\n")

        f.write("## Common Causes\n\n")
        f.write("- Poor lighting conditions\n")
        f.write("- Partial occlusion\n")
        f.write("- Unusual angles\n")
        f.write("- Similar-looking objects (especially ToolBox vs FireExtinguisher)\n")

    # Return directories for use in main
    return {
        "low_conf": low_conf_dir,
        "misclass": misclass_dir,
        "missed": missed_dir
    }


if __name__ == '__main__':
    # Setup paths
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load configuration
    config_path = this_dir / 'yolo_params.yaml'
    config = load_config(config_path)

    # Get class names
    class_names = config.get('names', ['Unknown'])

    # Check for test images directory
    if 'test' in config and config['test'] is not None:
        images_dir = Path(config['test']) / 'images'
    else:
        print("No test field found in yolo_params.yaml, using data/test/images")
        images_dir = this_dir / 'data' / 'test' / 'images'

    # Verify images directory exists and contains images
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        exit()

    # Find the best model from training runs
    model_path = this_dir / "runs" / "exp_space" / "weights" / "best.pt"

    if not model_path.exists():
        print(f"Best model weights not found at {model_path}")
        # Try last.pt as fallback
        model_path = this_dir / "runs" / "exp_space" / "weights" / "last.pt"
        if not model_path.exists():
            print("No model weights found. Please train a model first.")
            exit()
        print(f"Using last checkpoint instead: {model_path}")

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Create report directory
    report_dir = create_report_folder(this_dir, "eval_exp_space")
    print(f"Saving evaluation results to: {report_dir}")

    # Prediction and evaluation
    print("Running predictions on test images...")
    images_output_dir = report_dir / 'images'
    labels_output_dir = report_dir / 'labels'
    plots_output_dir = report_dir / 'plots'

    # Create directories
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the model and config for reference
    shutil.copy(model_path, report_dir / model_path.name)
    shutil.copy(config_path, report_dir / config_path.name)

    # Run the model validation on the test set
    val_results = model.val(data=config_path, split="test")

    # Save confusion matrix
    conf_matrix_path = plots_output_dir / "confusion_matrix.png"
    if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
        conf_matrix = val_results.confusion_matrix.matrix
        save_confusion_matrix(conf_matrix, class_names, conf_matrix_path)

    # Save other validation results - FIXED TO HANDLE NUMPY ARRAYS
    with open(report_dir / "validation_results.txt", "w") as f:
        # Handle the case where metrics might be numpy arrays
        try:
            map50 = val_results.box.map50
            map_val = val_results.box.map
            precision = val_results.box.p
            recall = val_results.box.r

            # Convert numpy arrays to floats if needed
            if hasattr(map50, 'item'):
                map50 = map50.item()
            if hasattr(map_val, 'item'):
                map_val = map_val.item()

            # For precision and recall which might be arrays, use mean
            if hasattr(precision, 'mean'):
                precision = precision.mean()
            if hasattr(recall, 'mean'):
                recall = recall.mean()

            f.write(f"mAP50: {map50:.4f}\n")
            f.write(f"mAP50-95: {map_val:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")

            # Print summary to console
            print("\n" + "=" * 50)
            print("EVALUATION METRICS SUMMARY")
            print("=" * 50)
            print(f"mAP50: {map50:.4f}")
            print(f"mAP50-95: {map_val:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print("=" * 50)
        except Exception as e:
            f.write(f"Error extracting metrics: {str(e)}\n")
            f.write("See validation output above for metrics.\n")

    # Print per-class metrics if available
    try:
        # Check if class metrics are available
        if hasattr(val_results, 'class_metrics'):
            print("\nPER-CLASS METRICS:")
            print("-" * 50)
            
            # Get class metrics
            for i, class_name in enumerate(class_names):
                class_map50 = val_results.class_metrics[i].get('map50', 0)
                class_precision = val_results.class_metrics[i].get('precision', 0)
                class_recall = val_results.class_metrics[i].get('recall', 0)
                
                print(f"{class_name}: mAP50={class_map50:.4f}, Precision={class_precision:.4f}, Recall={class_recall:.4f}")
            print("-" * 50)
    except Exception as e:
        print(f"Note: Could not extract per-class metrics: {e}")

    # Iterate through test images and make predictions
    all_results = []
    image_count = 0
    for img_path in images_dir.glob('*.*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        # Output paths
        img_output_path = images_output_dir / f"{img_path.stem}_pred{img_path.suffix}"
        txt_output_path = labels_output_dir / f"{img_path.stem}_pred.txt"

        # Make prediction and save
        result = predict_and_save(model, img_path, img_output_path, txt_output_path, class_names)
        all_results.append(result)
        image_count += 1
        
        # Print progress every 50 images
        if image_count % 50 == 0:
            print(f"Processed {image_count} images...")

    print(f"Predictions completed. {len(all_results)} images processed.")
    print(f"Results saved to {report_dir}")

    # Create failure analysis folders
    failure_dirs = analyze_failures(all_results, class_names, report_dir)

    print("Evaluation complete!")