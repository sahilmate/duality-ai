#!/usr/bin/env python3
# filepath: HackByte_Dataset/train.py
"""
YOLOv8 Training Script for Space Station Object Detection
This script reads configuration from yolo_params.yaml and trains a YOLOv8 model.
"""
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_metrics_summary(results):
    """Print summary metrics from training results."""
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - METRICS SUMMARY")
    print("=" * 50)
    
    # Extract metrics, handling the case where they might be different types
    try:
        # Try to access box metrics
        map50 = getattr(results, 'box', {}).get('map50', 0)
        map = getattr(results, 'box', {}).get('map', 0)
        precision = getattr(results, 'box', {}).get('p', 0)
        recall = getattr(results, 'box', {}).get('r', 0)
        
        # Convert numpy values to Python scalars if needed
        if hasattr(map50, 'item'):
            map50 = map50.item()
        if hasattr(map, 'item'):
            map = map.item()
        if hasattr(precision, 'mean'):
            precision = precision.mean().item() if hasattr(precision.mean(), 'item') else precision.mean()
        if hasattr(recall, 'mean'):
            recall = recall.mean().item() if hasattr(recall.mean(), 'item') else recall.mean()
            
        print(f"mAP50 (box): {map50:.4f}")
        print(f"mAP50-95 (box): {map:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    except Exception as e:
        print(f"Could not extract detailed metrics: {str(e)}")
        print("See results.csv for complete metrics")
    
    print("=" * 50)
    print(f"Trained for {results.epoch} epochs")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Space Station dataset")
    parser.add_argument('--config', type=str, default='yolo_params.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()
    
    # Setup paths
    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    
    # Load configuration
    config_path = this_dir / args.config
    config = load_config(config_path)
    
    # Extract training parameters with defaults
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch', 16)
    img_size = config.get('imgsz', 640)
    lr = config.get('lr0', 0.01)
    project_dir = config.get('project', 'runs/')
    exp_name = config.get('name', f'exp_space')
    device = config.get('device', None)  # None will auto-select
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Image size: {img_size}px")
    print(f"Learning rate: {lr}")
    print(f"Saving to: {project_dir}/{exp_name}")
    
    # Initialize model
    model_type = config.get('model', 'yolov8s.pt')
    if args.resume:
        # Try to find last checkpoint
        last_checkpoint = Path(project_dir) / exp_name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            model = YOLO(str(last_checkpoint))
            print(f"Resuming training from: {last_checkpoint}")
        else:
            print(f"No checkpoint found at {last_checkpoint}, starting from pretrained model")
            model = YOLO(model_type)
    else:
        # Start from pretrained model
        model = YOLO(model_type)
        print(f"Starting new training with model: {model_type}")
    
    # Start training
    results = model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        project=project_dir,
        name=exp_name,
        device=device,
        save=True,  # Save checkpoints
        save_period=config.get('save_period', 5),  # Save every N epochs
        exist_ok=True,  # Overwrite existing experiment
        resume=args.resume,  # Resume training
        plots=True,  # Generate plots
        verbose=True,  # Verbose output
        deterministic=True,  # Ensure reproducible results
    )
    
    # Print summary metrics
    print_metrics_summary(results)
    
    # Run validation on the test set
    print("\nRunning validation on test set...")
    val_results = model.val(data=str(config_path), split="test")
    
    print("\nTest Set Evaluation:")
    print_metrics_summary(val_results)
    
    print(f"\nTraining complete! Model saved to: {Path(project_dir) / exp_name / 'weights'}")
    print(f"Best model: {Path(project_dir) / exp_name / 'weights' / 'best.pt'}")
    print(f"Last checkpoint: {Path(project_dir) / exp_name / 'weights' / 'last.pt'}")

if __name__ == "__main__":
    main()