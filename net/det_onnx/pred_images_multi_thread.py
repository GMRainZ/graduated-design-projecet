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
def process_single_image(args):
    """
    单张图像处理函数，适用于多进程调用
    返回格式：(预测框列表, 真值框列表)
    """
    img_path, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name = args
    
    # 每个进程独立加载模型（避免共享资源冲突）
    session = load_model(model_weights)  # 需要确保load_model的线程安全性
    
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        return [], []
    h, w = img.shape[:2]
    
    # 推理
    det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)
    
    # 处理预测结果
    pred_boxes = []
    for d in det_objs:
        x_center = (d["bbox"][0] + d["bbox"][2]/2) / w
        y_center = (d["bbox"][1] + d["bbox"][3]/2) / h
        width = d["bbox"][2] / w
        height = d["bbox"][3] / h
        pred_boxes.append([x_center, y_center, width, height, d["confidence"], int(d["class_id"])])
    
    # 处理真值标注
    label_path = label_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        gt_boxes = process_hbb_gt(str(label_path), id_to_name, (w, h))
    else:
        gt_boxes = []
    
    return pred_boxes, gt_boxes

def main():
    # 配置参数
    model_weights = "weights/v11_coco128_attention_50epoch.onnx"
    conf_threshold = 0.5
    iou_threshold = 0.5
    imgsz = (640, 640)
    num_workers = min(8, mp.cpu_count()//2)  # 根据CPU核心数自动调整
    
    # 数据集配置
    dataset_root = Path("input")
    image_dir = dataset_root / "images/val"
    label_dir = dataset_root / "labels/val"
    
    # 加载标签映射
    id_to_name, _ = load_label_mappings(r"docs/coco128.yaml")
    
    # 准备任务参数（在主进程完成数据预处理）
    image_paths = list(image_dir.glob("*.jpg"))
    task_args = [
        (p, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name)
        for p in image_paths
    ]
    
    # 多进程处理
    all_preds, all_truths = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_single_image, task_args)
        
        # 使用tqdm显示进度
        for future in tqdm(futures, total=len(task_args), desc="并行推理"):
            preds, truths = future
            all_preds.append(preds)
            all_truths.append(truths)
    
    # 计算mAP
    results = batch_evaluate_hbb(all_preds, all_truths)
    
    # 输出结果
    print("\nEvaluation Results:")
    print(f"mAP@[0.5:0.95]: {results['mAP']:.4f}")
    print(f"总处理图像数: {len(all_preds)}")
    print(f"使用进程数: {num_workers}")

if __name__ == "__main__":
    # 确保多进程安全
    mp.set_start_method('spawn', force=True)  # 解决CUDA与多进程的兼容问题
    main()