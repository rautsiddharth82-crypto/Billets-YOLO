"""
Automated Billet Quality Inspection System - YOLO Model Training Script
======================================================================
Trains YOLOv8n and YOLOv10n models on the NEU-DET Surface Defect Database.
Exports trained weights to ONNX format for accelerated edge deployment on
NVIDIA GTX 1650 / Intel Industrial PCs.

Classes:
0: crazing
1: inclusion
2: patches
3: pitted_surface
4: rolled-in_scale
5: scratches

Usage:
    python train_yolo.py --model yolov8n --epochs 50 --imgsz 200
    python train_yolo.py --model yolov10n --epochs 50 --imgsz 200
    python train_yolo.py --prepare-sample  # Generates verified synthetic training set
"""

import os
import sys
import argparse
import random
import yaml
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

DATASET_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "neu_det.yaml")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "data", "neu_det")


def generate_synthetic_neudet_dataset(output_dir=DATASET_ROOT, samples_per_class=15):
    """
    Generates a realistic synthetic NEU-DET formatted dataset (images and YOLO labels)
    for immediate verification and offline training pipelines.
    """
    print(f"[Dataset Generator] Generating synthetic NEU-DET dataset at {output_dir}...")
    splits = ["train", "val", "test"]
    split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}

    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    count = 0
    for cls_id, cls_name in enumerate(CLASSES):
        for i in range(samples_per_class):
            count += 1
            # Random split allocation
            rnd = random.random()
            if rnd < split_ratios["train"]:
                split = "train"
            elif rnd < split_ratios["train"] + split_ratios["val"]:
                split = "val"
            else:
                split = "test"

            # 200x200 steel surface patch as per NEU-DET specs
            img = np.ones((200, 200, 3), dtype=np.uint8) * random.randint(90, 140)
            # Add cold-rolled grain noise
            noise = np.random.normal(0, 12, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            boxes = []
            # Draw class-specific visual defect signatures
            if cls_name == "crazing":
                # Network of micro-fissures
                for _ in range(5):
                    pt1 = (random.randint(30, 170), random.randint(30, 170))
                    pt2 = (pt1[0] + random.randint(-25, 25), pt1[1] + random.randint(-25, 25))
                    cv2.line(img, pt1, pt2, (40, 40, 40), 1)
                bx, by, bw, bh = 100, 100, 80, 80
                boxes.append((cls_id, bx / 200.0, by / 200.0, bw / 200.0, bh / 200.0))

            elif cls_name == "scratches":
                # Longitudinal directional scratches
                pt1 = (random.randint(20, 60), random.randint(30, 170))
                pt2 = (random.randint(140, 180), pt1[1] + random.randint(-15, 15))
                cv2.line(img, pt1, pt2, (25, 25, 25), 2)
                cx = (pt1[0] + pt2[0]) / 2.0
                cy = (pt1[1] + pt2[1]) / 2.0
                bw = abs(pt2[0] - pt1[0]) + 10
                bh = 20
                boxes.append((cls_id, cx / 200.0, cy / 200.0, bw / 200.0, bh / 200.0))

            elif cls_name == "inclusion":
                # Non-metallic oxide inclusions
                cx, cy = random.randint(50, 150), random.randint(50, 150)
                r = random.randint(8, 20)
                cv2.circle(img, (cx, cy), r, (30, 30, 30), -1)
                boxes.append((cls_id, cx / 200.0, cy / 200.0, (r * 2 + 6) / 200.0, (r * 2 + 6) / 200.0))

            elif cls_name == "patches":
                # Large darker scale patches
                cx, cy = random.randint(60, 140), random.randint(60, 140)
                bw, bh = random.randint(40, 70), random.randint(30, 60)
                cv2.rectangle(img, (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2), (50, 50, 50), -1)
                boxes.append((cls_id, cx / 200.0, cy / 200.0, bw / 200.0, bh / 200.0))

            elif cls_name == "pitted_surface":
                # Dispersed pitting holes
                cx, cy = random.randint(60, 140), random.randint(60, 140)
                for _ in range(6):
                    px = cx + random.randint(-20, 20)
                    py = cy + random.randint(-20, 20)
                    cv2.circle(img, (px, py), random.randint(2, 4), (20, 20, 20), -1)
                boxes.append((cls_id, cx / 200.0, cy / 200.0, 50 / 200.0, 50 / 200.0))

            elif cls_name == "rolled-in_scale":
                # Imbedded dark mill scale pattern
                cx, cy = random.randint(60, 140), random.randint(60, 140)
                cv2.ellipse(img, (cx, cy), (random.randint(25, 45), random.randint(12, 20)), 25, 0, 360, (35, 35, 35), -1)
                boxes.append((cls_id, cx / 200.0, cy / 200.0, 60 / 200.0, 35 / 200.0))

            # Save image
            img_filename = f"{cls_name}_{i:03d}.jpg"
            img_path = os.path.join(output_dir, "images", split, img_filename)
            cv2.imwrite(img_path, img)

            # Save label in YOLO format: <class> <x_center> <y_center> <width> <height>
            label_filename = f"{cls_name}_{i:03d}.txt"
            label_path = os.path.join(output_dir, "labels", split, label_filename)
            with open(label_path, "w") as lf:
                for b in boxes:
                    lf.write(f"{b[0]} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f}\n")

    print(f"[Dataset Generator] Created {count} images across train/val/test splits.")

    # Update yaml path to absolute path
    config_data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASSES)},
        "nc": len(CLASSES)
    }
    with open(DATASET_CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    print(f"[Dataset Generator] Updated {DATASET_CONFIG_PATH} with absolute path.")


def train_model(model_name="yolov8n", epochs=50, imgsz=200, batch_size=16, export_onnx=True):
    """
    Trains YOLO model and exports weights.
    
    Args:
        model_name (str): 'yolov8n' or 'yolov10n'
        epochs (int): Number of training epochs (default 50)
        imgsz (int): Image resolution (default 200 for NEU-DET)
        batch_size (int): Batch size
        export_onnx (bool): Whether to export ONNX model
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Ensure dataset exists
    train_dir = os.path.join(DATASET_ROOT, "images", "train")
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("[Training] NEU-DET dataset not found. Generating initial synthetic verification dataset...")
        generate_synthetic_neudet_dataset()

    base_model_file = f"{model_name}.pt"
    print(f"\n==================================================")
    print(f"STARTING TRAINING: {model_name} on NEU-DET")
    print(f"Epochs: {epochs} | ImgSz: {imgsz} | Batch: {batch_size}")
    print(f"==================================================")

    model = YOLO(base_model_file)

    # Run training
    results = model.train(
        data=DATASET_CONFIG_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=0,  # Stable on Windows
        project="runs/detect",
        name=f"train_{model_name}",
        exist_ok=True,
        verbose=True
    )

    best_pt = os.path.join("runs", "detect", f"train_{model_name}", "weights", "best.pt")
    target_pt = os.path.join(MODELS_DIR, f"best_{model_name}.pt")

    if os.path.exists(best_pt):
        import shutil
        shutil.copyfile(best_pt, target_pt)
        print(f"\n[Training] Saved best weights to {target_pt}")
    else:
        # Fallback to current model save
        model.save(target_pt)
        print(f"\n[Training] Model weights saved to {target_pt}")

    # Evaluate validation metrics
    print("\n--- Validation Evaluation ---")
    val_results = model.val(data=DATASET_CONFIG_PATH)
    print(f"Validation mAP50: {val_results.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_results.box.map:.4f}")

    if export_onnx:
        print(f"\n[Export] Exporting {model_name} to ONNX format...")
        onnx_file = model.export(format="onnx", imgsz=imgsz, dynamic=True)
        target_onnx = os.path.join(MODELS_DIR, f"best_{model_name}.onnx")
        if os.path.exists(onnx_file):
            import shutil
            shutil.copyfile(onnx_file, target_onnx)
            print(f"[Export] Saved ONNX model to {target_onnx}")

    return target_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 / YOLOv10 on NEU-DET defect dataset")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov10n"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=200, help="Image size (NEU-DET default is 200)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--prepare-sample", action="store_true", help="Generate synthetic NEU-DET sample dataset")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")

    args = parser.parse_args()

    if args.prepare_sample:
        generate_synthetic_neudet_dataset()
    else:
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            export_onnx=not args.no_onnx
        )
