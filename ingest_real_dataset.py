"""
Automated Billet Quality Inspection System - Real Dataset Ingest Utility
=======================================================================
Converts and organizes the real NEU-DET (or custom steel plant defect) dataset
into the required YOLO format:
  data/neu_det/
    ├── images/ [train, val, test]
    └── labels/ [train, val, test]

Supports:
1. Pascal VOC XML annotations (original NEU-DET format)
2. Direct YOLO format (Roboflow exports)
3. Raw images per class folder
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
import glob
from pathlib import Path

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

TARGET_ROOT = os.path.join(os.path.dirname(__file__), "data", "neu_det")


def convert_voc_xml_to_yolo(xml_path, img_w=200, img_h=200):
    """Parses Pascal VOC XML and converts bounding boxes to normalized YOLO format."""
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions from XML if available
    size_node = root.find("size")
    if size_node is not None:
        w_node = size_node.find("width")
        h_node = size_node.find("height")
        if w_node is not None and int(w_node.text) > 0:
            img_w = int(w_node.text)
        if h_node is not None and int(h_node.text) > 0:
            img_h = int(h_node.text)

    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        # Map common naming variations
        name = name.replace(" ", "_").replace("-", "_")
        if name not in CLASSES:
            # Map aliases
            alias_map = {
                "cr": "crazing", "in": "inclusion", "pa": "patches",
                "ps": "pitted_surface", "rs": "rolled_in_scale", "sc": "scratches",
                "rolled_in_scale": "rolled-in_scale"
            }
            name = alias_map.get(name, name)
            
        cls_name_std = name.replace("_", "-") if name == "rolled_in_scale" else name
        if cls_name_std in CLASSES:
            cls_id = CLASSES.index(cls_name_std)
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            boxes.append((cls_id, x_center, y_center, w, h))
            
    return boxes


def ingest_from_source(source_folder):
    """
    Ingests real NEU-DET images and annotations from source_folder
    and splits them into 70% train / 20% val / 10% test.
    """
    if not os.path.exists(source_folder):
        print(f"[Error] Source folder does not exist: {source_folder}")
        return False

    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(TARGET_ROOT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(TARGET_ROOT, "labels", s), exist_ok=True)

    # Find all images (.jpg, .png, .bmp)
    img_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_files.extend(glob.glob(os.path.join(source_folder, "**", ext), recursive=True))

    if not img_files:
        print(f"[Error] No images found in {source_folder}")
        return False

    print(f"[Ingest] Found {len(img_files)} real images. Processing & splitting...")
    random.seed(42)
    random.shuffle(img_files)

    n_train = int(len(img_files) * 0.70)
    n_val = int(len(img_files) * 0.20)

    stats = {"train": 0, "val": 0, "test": 0}

    for idx, img_p in enumerate(img_files):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        fname = os.path.basename(img_p)
        stem = Path(img_p).stem

        # Destination paths
        dst_img = os.path.join(TARGET_ROOT, "images", split, fname)
        shutil.copy2(img_p, dst_img)

        # Look for corresponding XML or TXT label
        xml_candidates = [
            os.path.join(os.path.dirname(img_p), f"{stem}.xml"),
            os.path.join(source_folder, "ANNOTATIONS", f"{stem}.xml"),
            os.path.join(source_folder, "annotations", f"{stem}.xml")
        ]
        txt_candidates = [
            os.path.join(os.path.dirname(img_p), f"{stem}.txt"),
            os.path.join(os.path.dirname(img_p), "..", "labels", f"{stem}.txt"),
            os.path.join(source_folder, "labels", f"{stem}.txt")
        ]

        dst_label = os.path.join(TARGET_ROOT, "labels", split, f"{stem}.txt")

        # Case 1: VOC XML exists
        found_label = False
        for xc in xml_candidates:
            if os.path.exists(xc):
                boxes = convert_voc_xml_to_yolo(xc)
                with open(dst_label, "w") as f:
                    for b in boxes:
                        f.write(f"{b[0]} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f}\n")
                found_label = True
                break

        # Case 2: YOLO TXT exists
        if not found_label:
            for tc in txt_candidates:
                if os.path.exists(tc):
                    shutil.copy2(tc, dst_label)
                    found_label = True
                    break

        # Case 3: Infer from class filename if raw class image
        if not found_label:
            for c_id, c_name in enumerate(CLASSES):
                if c_name in fname.lower():
                    # Default centered bounding box for classification image
                    with open(dst_label, "w") as f:
                        f.write(f"{c_id} 0.5000 0.5000 0.7000 0.7000\n")
                    found_label = True
                    break

        stats[split] += 1

    print(f"[SUCCESS] Ingest Complete! Total images processed: {len(img_files)}")
    print(f"   - Train: {stats['train']} images")
    print(f"   - Val:   {stats['val']} images")
    print(f"   - Test:  {stats['test']} images")
    print(f"\nNext step: Retrain models with:")
    print("   python train_yolo.py --model yolov8n --epochs 50")
    print("   python train_yolo.py --model yolov10n --epochs 50")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_from_source(sys.argv[1])
    else:
        print("Usage: python ingest_real_dataset.py <path_to_downloaded_dataset_folder>")
