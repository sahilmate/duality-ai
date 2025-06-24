#!/usr/bin/env python3
# filepath: HackByte_Dataset/train.py
"""
YOLOv8 Training Script for Space Station Object Detection
This script reads configuration from yolo_params.yaml and trains a YOLOv8 model.
"""
import os
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_metrics(metrics):
    """Extract mAP, precision, recall using updated Ultralytics keys."""
    return {
        "mAP50": float(metrics.get('metrics/mAP50(B)', 0)),
        "mAP50-95": float(metrics.get('metrics/mAP50-95(B)', 0)),
        "precision": float(metrics.get('metrics/precision(B)', 0)),
        "recall": float(metrics.get('metrics/recall(B)', 0))
    }

def print_metrics_summary(results):
    """Print summary metrics from training results."""
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - METRICS SUMMARY")
    print("=" * 50)

    try:
        metrics_dict = results.results_dict if hasattr(results, 'results_dict') else results.__dict__
        extracted = extract_metrics(metrics_dict)
        f1 = 2 * (extracted["precision"] * extracted["recall"]) / (extracted["precision"] + extracted["recall"] + 1e-16)

        print(f"mAP50: {extracted['mAP50']:.4f}")
        print(f"mAP50-95: {extracted['mAP50-95']:.4f}")
        print(f"Precision: {extracted['precision']:.4f}")
        print(f"Recall: {extracted['recall']:.4f}")
        print(f"F1 Score: {f1:.4f}")

    except Exception as e:
        print(f"Could not extract detailed metrics: {str(e)}")
        print("See results.csv for complete metrics")

    print("=" * 50)

def save_training_summary(results, config, save_dir, config_path):
    """Save a summary of training parameters and results."""
    try:
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_file": str(config_path),
            "epochs": config.get("epochs", "N/A"),
            "image_size": config.get("imgsz", "N/A"),
            "batch_size": config.get("batch", "N/A"),
            "learning_rate": config.get("lr0", "N/A"),
            "model_type": config.get("model", "yolov8s.pt")
        }

        try:
            metrics_dict = results.results_dict if hasattr(results, 'results_dict') else results.__dict__
            extracted = extract_metrics(metrics_dict)
            p = extracted["precision"]
            r = extracted["recall"]
            f1 = 2 * (p * r) / (p + r + 1e-16) if (p + r) > 0 else 0

            summary["metrics"] = { **extracted, "F1": f1 }
        except Exception:
            summary["metrics"] = "See results.csv for metrics"

        with open(os.path.join(save_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Training summary saved to {os.path.join(save_dir, 'training_summary.json')}")
    except Exception as e:
        print(f"Warning: Could not save training summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Space Station dataset")
    parser.add_argument('--config', type=str, default='yolo_params.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--weights', type=str, default=None, help='Initial weights path (overrides model in config)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (empty for auto-select, cpu, 0, 1, etc.)')
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
    device = args.device or config.get('device', None)

    print(f"Configuration loaded from: {config_path}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    print(f"Image size: {img_size}px")
    print(f"Learning rate: {lr}")
    print(f"Saving to: {project_dir}/{exp_name}")

    model_path = args.weights or config.get('model', 'yolov8s.pt')

    if args.resume:
        last_checkpoint = Path(project_dir) / exp_name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            model = YOLO(str(last_checkpoint))
            print(f"Resuming training from: {last_checkpoint}")
        else:
            print(f"No checkpoint found at {last_checkpoint}, starting from {model_path}")
            model = YOLO(model_path)
    else:
        model = YOLO(model_path)
        print(f"Starting new training with model: {model_path}")

    train_params = {
        'data': str(config_path),
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'lr0': lr,
        'project': project_dir,
        'name': exp_name,
        'device': device,
        'save': True,
        'save_period': config.get('save_period', 5),
        'exist_ok': True,
        'resume': args.resume,
        'plots': True,
        'verbose': True,
        'deterministic': True,
    }

    possible_params = [
        'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum',
        'warmup_bias_lr', 'box', 'cls', 'dfl', 'fl_gamma', 'label_smoothing',
        'nbs', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
        'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup',
        'copy_paste', 'optimizer', 'seed', 'single_cls', 'rect', 'cos_lr',
        'close_mosaic', 'amp', 'fraction'
    ]

    for param in possible_params:
        if param in config:
            train_params[param] = config[param]

    results = model.train(**train_params)
    print_metrics_summary(results)

    save_dir = Path(project_dir) / exp_name
    save_training_summary(results, config, save_dir, config_path)

    print("\nRunning validation on test set...")
    val_results = model.val(data=str(config_path), split="test")

    print("\nTest Set Evaluation:")
    print_metrics_summary(val_results)

    save_dir = Path(project_dir) / exp_name / "test_evaluation"
    save_dir.mkdir(exist_ok=True)
    save_training_summary(val_results, config, save_dir, config_path)

    print(f"\nTraining complete! Model saved to: {Path(project_dir) / exp_name / 'weights'}")
    print(f"Best model: {Path(project_dir) / exp_name / 'weights' / 'best.pt'}")
    print(f"Last checkpoint: {Path(project_dir) / exp_name / 'weights' / 'last.pt'}")

if __name__ == "__main__":
    main()
