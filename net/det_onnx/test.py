from detedctionapi import DetectionAPI

import os
from pathlib import Path
def main():
    detector = DetectionAPI(
        model_weights="weights/v11_coco128_attention_50epoch.onnx",
        label_mapping_path="docs/coco128.yaml",
        default_conf=0.5,
        default_iou=0.5,
        default_imgsz=(640, 640)
    )

    image_dir=Path("input/images/val/")
    image_paths = [str(image_dir / f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

    detector.process_image_batch(
        image_paths=image_paths,
        label_dir="data/labels/",
        conf=0.5,
        iou=0.5
    )

if __name__ == "__main__":
    main()