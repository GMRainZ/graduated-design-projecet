import argparse
import os
from pathlib import Path
from multiprocessing import get_start_method, set_start_method
from typing import Callable, List, Tuple, Dict, Optional
from utils.v11_key_op import load_model, pred, save_detections, process_hbb_gt, batch_evaluate_hbb, load_label_mappings
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp


def main():
    # 配置参数
    model_weights = "weights/v11_coco128_attention_50epoch.onnx"
    conf_threshold = 0.5
    iou_threshold = 0.5
    imgsz = (640, 640)
    
    # 数据集配置
    dataset_root = Path("input")
    image_dir = dataset_root / "images/val"
    label_dir = dataset_root / "labels/val"
    
    # 加载模型和标签映射
    id_to_name, _ = load_label_mappings(r"docs/coco128.yaml")
    session = load_model(model_weights)
    
    # 初始化数据收集器
    all_preds = []
    all_truths = []
    
    # 遍历数据集
    image_paths = list(image_dir.glob("*.jpg"))
    for img_path in tqdm(image_paths, desc="Processing images"):
        # 读取并预处理图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # 执行推理
        det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)
        
        # 收集预测结果（转换为比例坐标）
        h, w = img.shape[:2]
        pred_boxes = []
        for d in det_objs:
            x_center = (d["bbox"][0] + d["bbox"][2]/2) / w  # 转换为比例坐标
            y_center = (d["bbox"][1] + d["bbox"][3]/2) / h
            width = d["bbox"][2] / w
            height = d["bbox"][3] / h
            pred_boxes.append([
                x_center, y_center, width, height, 
                d["confidence"], int(d["class_id"])
            ])
        all_preds.append(pred_boxes)
        
        # 读取并处理真值标注
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            gt_boxes = process_hbb_gt(str(label_path), id_to_name, (w, h))
            all_truths.append(gt_boxes)
        else:
            all_truths.append([])  # 空真值

    # 计算mAP
    results = batch_evaluate_hbb(all_preds, all_truths)
    
    # 输出结果
    print("\nEvaluation Results:")
    print(f"mAP@[0.5:0.95]: {results['mAP']:.4f}")
    print("Per-class AP:")
    for cls_id, ap in results['ap_per_class'].items():
        print(f"  {id_to_name[cls_id]:<15}: {ap:.4f}")

if __name__ == "__main__":
    main()