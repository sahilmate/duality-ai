#!/usr/bin/env python3
# filepath: HackByte_Dataset/visualize.py
"""
YOLOv8 Visualization Script for Space Station Object Detection
This script creates visualizations from YOLOv8 training and evaluation results.
"""

import os
import cv2
import numpy as np
# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
import glob
import pandas as pd
from ultralytics import YOLO

# Main function
def main():
    # Set up paths
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load configuration
    config_path = this_dir / 'yolo_params.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Get class names
    class_names = config.get('names', ['Unknown'])

    # Create visualization output directory
    viz_dir = this_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # 1. Create dataset distribution visualization
    print("Creating dataset distribution visualization...")
    create_dataset_distribution(this_dir, class_names, viz_dir)

    # 2. Create model performance visualizations
    print("Creating model performance visualizations...")
    create_performance_viz(this_dir, class_names, viz_dir)

    # 3. Create sample predictions visualization
    print("Creating sample predictions visualization...")
    create_prediction_samples(this_dir, viz_dir)

    # 4. Create confusion matrix visualization
    print("Creating confusion matrix visualization...")
    create_confusion_matrix_viz(this_dir, class_names, viz_dir)
    
    # 5. Create precision-recall curves
    print("Creating precision-recall curves...")
    create_pr_curves(this_dir, class_names, viz_dir)
    
    # 6. Create before-after comparison
    print("Creating before-after comparison...")
    create_before_after_comparison(this_dir, viz_dir)

    print(f"All visualizations saved to {viz_dir}")

def create_dataset_distribution(this_dir, class_names, output_dir):
    """Create visualizations of dataset distribution."""
    # Count objects in each split
    train_counts = count_objects_by_class(this_dir / "data" / "train" / "labels", len(class_names))
    val_counts = count_objects_by_class(this_dir / "data" / "val" / "labels", len(class_names))
    test_counts = count_objects_by_class(this_dir / "data" / "test" / "labels", len(class_names))

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.2
    x = np.arange(len(class_names))

    ax.bar(x - width, train_counts, width, label='Train')
    ax.bar(x, val_counts, width, label='Validation')
    ax.bar(x + width, test_counts, width, label='Test')

    ax.set_ylabel('Number of Objects')
    ax.set_title('Object Distribution by Dataset Split')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_distribution.png", dpi=300)
    plt.close()

    # Create pie chart of overall distribution
    total_counts = [t + v + te for t, v, te in zip(train_counts, val_counts, test_counts)]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(total_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Overall Class Distribution')

    plt.savefig(output_dir / "class_distribution_pie.png", dpi=300)
    plt.close()

def count_objects_by_class(labels_dir, num_classes):
    """Count objects by class in a directory of label files."""
    counts = [0] * num_classes

    if not labels_dir.exists():
        return counts

    for label_file in labels_dir.glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if 0 <= class_id < num_classes:
                            counts[class_id] += 1
        except Exception as e:
            print(f"Error processing {label_file}: {e}")

    return counts

def create_performance_viz(this_dir, class_names, output_dir):
    """Create visualizations of model performance."""
    # Check for results.csv in the training directory
    results_path = list(this_dir.glob('runs/exp_space/results.csv'))

    if not results_path:
        results_path = list(this_dir.glob('runs/**/results.csv'))

    if not results_path:
        print("No results.csv found. Skipping performance visualization.")
        return

    results_path = results_path[0]

    # Read results
    try:
        results_df = pd.read_csv(results_path)

        # 1. Create training curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot mAP
        if 'mAP50' in results_df.columns:
            ax1.plot(results_df['epoch'], results_df['mAP50'], 'b-', label='mAP@0.5')
        if 'mAP50-95' in results_df.columns:
            ax1.plot(results_df['epoch'], results_df['mAP50-95'], 'r-', label='mAP@0.5:0.95')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.set_title('Mean Average Precision')
        ax1.legend()
        ax1.grid(True)

        # Plot precision and recall
        if 'precision' in results_df.columns:
            ax2.plot(results_df['epoch'], results_df['precision'], 'g-', label='Precision')
        if 'recall' in results_df.columns:
            ax2.plot(results_df['epoch'], results_df['recall'], 'm-', label='Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Precision and Recall')
        ax2.legend()
        ax2.grid(True)

        # Plot losses
        if 'box_loss' in results_df.columns:
            ax3.plot(results_df['epoch'], results_df['box_loss'], 'b-', label='Box Loss')
        if 'cls_loss' in results_df.columns:
            ax3.plot(results_df['epoch'], results_df['cls_loss'], 'r-', label='Class Loss')
        if 'dfl_loss' in results_df.columns:
            ax3.plot(results_df['epoch'], results_df['dfl_loss'], 'g-', label='DFL Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Losses')
        ax3.legend()
        ax3.grid(True)

        # Plot learning rate
        if 'lr0' in results_df.columns:
            ax4.plot(results_df['epoch'], results_df['lr0'], 'k-')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=300)
        plt.close()

        # 2. Create final metrics summary
        final_metrics = results_df.iloc[-1]

        # Create a bar chart of final metrics
        metrics_to_plot = ['precision', 'recall', 'mAP50', 'mAP50-95']
        available_metrics = [m for m in metrics_to_plot if m in final_metrics]

        if available_metrics:
            values = [final_metrics[m] for m in available_metrics]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(available_metrics, values)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Metric Value')
            ax.set_title('Final Model Performance Metrics')
            ax.grid(axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / "final_metrics.png", dpi=300)
            plt.close()

    except Exception as e:
        print(f"Error creating performance visualizations: {e}")

def create_prediction_samples(this_dir, output_dir):
    """Create visualization of sample predictions."""
    # Look for prediction images
    pred_dirs = [
        this_dir / "reports" / "eval_exp_space" / "images",
        this_dir / "runs" / "detect" / "predict" / "images",
        this_dir / "runs" / "exp_space" / "val"
    ]

    pred_images = []
    for pred_dir in pred_dirs:
        if pred_dir.exists():
            pred_images.extend(list(pred_dir.glob('*.jpg')))
            pred_images.extend(list(pred_dir.glob('*.png')))

    if not pred_images:
        print("No prediction images found. Skipping sample visualization.")
        return

    # Select up to 16 random images
    import random
    sample_images = random.sample(pred_images, min(16, len(pred_images)))

    # Create a grid of images
    rows = min(4, len(sample_images))
    cols = min(4, (len(sample_images) + rows - 1) // rows)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    # Flatten axes array for easy indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, img_path in enumerate(sample_images):
        if i < len(axes):
            try:
                img = plt.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Sample {i+1}")
                axes[i].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # Hide unused subplots
    for i in range(len(sample_images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_samples.png", dpi=300)
    plt.close()

def create_confusion_matrix_viz(this_dir, class_names, output_dir):
    """Create visualization of the confusion matrix."""
    # Look for confusion matrix
    confusion_matrix_paths = [
        this_dir / "reports" / "eval_exp_space" / "plots" / "confusion_matrix.png",
        this_dir / "runs" / "exp_space" / "confusion_matrix.png",
        this_dir / "runs" / "detect" / "val" / "confusion_matrix.png"
    ]

    for cm_path in confusion_matrix_paths:
        if cm_path.exists():
            # Copy the confusion matrix to the visualization directory
            shutil.copy(cm_path, output_dir / "confusion_matrix.png")
            print(f"Copied confusion matrix from {cm_path}")
            return

    # If no confusion matrix is found, try to create one
    print("No existing confusion matrix found. Attempting to create one...")
    
    # Try to find validation results with confusion matrix
    try:
        model_path = list(this_dir.glob('runs/exp_space/weights/best.pt'))
        if model_path:
            model = YOLO(str(model_path[0]))
            val_results = model.val(data=str(this_dir / 'yolo_params.yaml'), split="test")
            
            if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
                conf_matrix = val_results.confusion_matrix.matrix
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar()
                
                # Add labels
                plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
                plt.yticks(np.arange(len(class_names)), class_names)
                
                # Add numbers to cells
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        plt.text(j, i, str(int(conf_matrix[i, j])),
                                ha="center", va="center", 
                                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
                
                plt.tight_layout()
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.title('Confusion Matrix')
                
                plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
                plt.close()
                print("Created confusion matrix from validation results")
                return
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    
    print("Could not create confusion matrix. Please run prediction.py first.")

def create_pr_curves(this_dir, class_names, output_dir):
    """Create precision-recall curves for each class."""
    # Check if PR curve data is available
    try:
        # Look for PR curve data in common locations
        pr_data_paths = [
            this_dir / "reports" / "eval_exp_space" / "PR_data.json",
            this_dir / "runs" / "exp_space" / "PR_data.json",
            this_dir / "runs" / "detect" / "val" / "PR_data.json"
        ]
        
        pr_data = None
        for path in pr_data_paths:
            if path.exists():
                with open(path, 'r') as f:
                    pr_data = json.load(f)
                print(f"Loaded PR data from {path}")
                break
        
        if pr_data is None:
            # If no PR data found, try to create it by running validation
            print("No PR curve data found. Attempting to create it...")
            model_path = list(this_dir.glob('runs/exp_space/weights/best.pt'))
            if model_path:
                model = YOLO(str(model_path[0]))
                model.val(data=str(this_dir / 'yolo_params.yaml'), split="test", save_json=True)
                
                # Check if PR data was created
                for path in pr_data_paths:
                    if path.exists():
                        with open(path, 'r') as f:
                            pr_data = json.load(f)
                        print(f"Created and loaded PR data from {path}")
                        break
        
        if pr_data is None:
            # If still no PR data, create a placeholder
            print("Could not create PR data. Creating placeholder PR curves...")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create random PR curves for demonstration
            for i, class_name in enumerate(class_names):
                precision = np.linspace(1, 0.5, 11) + np.random.normal(0, 0.05, 11)
                precision = np.clip(precision, 0, 1)
                recall = np.linspace(0, 1, 11)
                ax.plot(recall, precision, 'o-', label=f"{class_name}")
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Placeholder Precision-Recall Curves\n(Actual data not available)')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / "pr_curves_placeholder.png", dpi=300)
            plt.close()
            return
        
        # Create PR curves from the data
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Process PR data
        for i, class_name in enumerate(class_names):
            if str(i) in pr_data:
                class_data = pr_data[str(i)]
                precision = class_data.get('precision', [])
                recall = class_data.get('recall', [])
                
                if precision and recall:
                    ax.plot(recall, precision, 'o-', label=f"{class_name}")
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves by Class')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curves.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating PR curves: {e}")
        # Create placeholder
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "PR curves could not be created", 
                ha='center', va='center', fontsize=12)
        ax.set_title("Error Creating PR Curves")
        plt.savefig(output_dir / "pr_curves_error.png", dpi=300)
        plt.close()

def create_before_after_comparison(this_dir, output_dir):
    """Create before-after comparison of predictions."""
    # Look for prediction images
    pred_dir = this_dir / "reports" / "eval_exp_space" / "images"
    if not pred_dir.exists():
        print("No prediction images found. Skipping before-after comparison.")
        return
    
    # Find some prediction images
    pred_images = list(pred_dir.glob("*_pred.png"))
    if not pred_images:
        pred_images = list(pred_dir.glob("*_pred.jpg"))
    
    if not pred_images:
        print("No prediction images found. Skipping before-after comparison.")
        return
    
    # Choose up to 5 images for comparison
    import random
    sample_images = random.sample(pred_images, min(5, len(pred_images)))
    
    # For each prediction image, find the original
    comparison_pairs = []
    for pred_img in sample_images:
        # Extract the original image name (remove _pred suffix)
        orig_name = pred_img.stem.replace("_pred", "")
        
        # Look for original in test/images
        test_img_dir = this_dir / "data" / "test" / "images"
        orig_candidates = list(test_img_dir.glob(f"{orig_name}.*"))
        
        if orig_candidates:
            comparison_pairs.append((orig_candidates[0], pred_img))
    
    if not comparison_pairs:
        print("Could not find matching original images. Skipping before-after comparison.")
        return
    
    # Create a grid of before-after comparisons
    n_pairs = len(comparison_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs))
    
    # Handle case of single pair
    if n_pairs == 1:
        axes = [axes]
    
    for i, (orig_img, pred_img) in enumerate(comparison_pairs):
        try:
            # Load images
            orig = plt.imread(orig_img)
            pred = plt.imread(pred_img)
            
            # Display original
            axes[i][0].imshow(orig)
            axes[i][0].set_title(f"Original")
            axes[i][0].axis('off')
            
            # Display prediction
            axes[i][1].imshow(pred)
            axes[i][1].set_title(f"Detection")
            axes[i][1].axis('off')
        except Exception as e:
            print(f"Error processing image pair {orig_img} and {pred_img}: {e}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "before_after_comparison.png", dpi=300)
    plt.close()
    print("Created before-after comparison visualization")

if __name__ == "__main__":
    main()