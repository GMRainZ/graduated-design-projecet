import os
import cv2
import numpy as np
import onnxruntime as ort
from utils.general import convert_dict_to_list, process_hbb_gt,load_label_mappings
import val_with_np  # 假设这是处理水平框的评估模块
import time
from functools import wraps

from val_with_np import evaluate_hbb_detection,batch_evaluate_hbb

def average_execution_time(func):
    total_time = 0.0  # 毫秒累计值
    num_calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal total_time, num_calls
        # 使用性能计数器获取高精度时间戳（纳秒级）
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        
        # 计算纳秒差值并转换为毫秒（保留6位小数）
        elapsed_ms = (end - start) / 1_000_000  # 1毫秒 = 1,000,000纳秒
        total_time += elapsed_ms
        num_calls += 1
        
        return result

    def get_average(precision=3):
        """获取平均执行时间（毫秒），默认保留3位小数"""
        return round(total_time / num_calls, precision) if num_calls > 0 else 0.0

    def reset():
        nonlocal total_time, num_calls
        total_time = 0.0
        num_calls = 0

    def get_stats():
        """获取详细统计信息"""
        return {
            "calls": num_calls,
            "total_ms": round(total_time, 6),
            "avg_ms": get_average(),
            "min_precision": "0.001ms"  # 理论最小精度为1微秒
        }

    wrapper.get_average = get_average
    wrapper.reset = reset
    wrapper.get_stats = get_stats
    
    return wrapper
def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=False, scale_fill=False, scale_up=False, stride=32):
    '''
    调整图像尺寸

    Args:
        img (numpy.ndarray): 输入图像
        new_shape (int or tuple): 目标尺寸，可以是整数或元组
        color (tuple): 填充颜色，默认为黑色
        auto (bool): 是否自动调整填充为步幅的整数倍，默认为False
        scale_fill (bool): 是否强制缩放以完全填充目标尺寸，默认为False
        scale_up (bool): 是否允许放大图像，默认为False
       stride (int): 步幅，用于自动调整填充，默认
    '''
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例
    if not scale_up:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # 填充均分
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def load_model(weights):
    use_gpu = False
    session = ort.InferenceSession(weights, providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider'])
    return session

def bbox_nms(boxes, scores, iou_threshold=0.5):
    """水平框NMS实现"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = boxes[:, 2] * boxes[:, 3]
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

@average_execution_time
def do_inference(session, img_data, imgsz=(640, 640)):
    '''
    预处理 + det

    Args:
        session (obj): ONNX runtime object
        img_data (np.ndarray): 输入图像数据
        imgsz (tuple): 模型输入的尺寸
    Return:
        推理结果、缩放比例、填充尺寸
    '''

    img, ratio, (dw, dh) = letterbox(img_data, new_shape=imgsz)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = img[np.newaxis, ...].astype(np.float32) / 255.0

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img})

    return result[0], ratio, (dw, dh)

def post_process(output, ratio, dwdh, conf_threshold=0.5, iou_threshold=0.5):
    """修改后的后处理（水平框版本）"""
    boxes, scores, classes, detections = [], [], [], []
    num_detections = output.shape[2]
    num_classes = output.shape[1] - 5  # 现在输出是x,y,w,h,cls_conf...

    for i in range(num_detections):
        detection = output[0, :, i]
        x_center, y_center, width, height = detection[0], detection[1], detection[2], detection[3]

        if num_classes > 0:
            class_confidences = detection[4 : 4 + num_classes]
            class_id = np.argmax(class_confidences)
            confidence = class_confidences[class_id]
        else:
            confidence = detection[4]
            class_id = 0

        if confidence > conf_threshold:
            # 坐标转换
            x_center = (x_center - dwdh[0]) / ratio[0]
            y_center = (y_center - dwdh[1]) / ratio[1]
            width /= ratio[0]
            height /= ratio[1]

            boxes.append([x_center, y_center, width, height])
            scores.append(confidence)
            classes.append(class_id)

    if not boxes:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    # 使用水平框NMS
    keep_indices = bbox_nms(boxes, scores, iou_threshold=iou_threshold)

    # 转换为中心点+宽高格式
    for idx in keep_indices:
        x, y, w, h = boxes[idx]
        confidence = scores[idx]
        class_id = classes[idx]
        
        detections.append({
            "bbox": [x, y, w, h],
            "confidence": float(confidence),
            "class_id": int(class_id)
        })

    return detections

def save_detections(image, detections, output_path,id_to_name):
    """修改后的可视化函数（绘制水平框）"""
    for det in detections:
        x, y, w, h = det["bbox"]
        confidence = det["confidence"]
        class_id = det["class_id"]
        
        # 转换为中心点转左上角坐标
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text_line2 = f"Class: {id_to_name[class_id]}"
        text_line1 = f"Conf: {confidence:.2f}"

        # 获取文本尺寸用于计算位置
        (line1_width, line1_height), _ = cv2.getTextSize(text_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (line2_width, line2_height), _ = cv2.getTextSize(text_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # 计算垂直偏移（行高+间距）
        vertical_offset = line1_height + 5  # 5像素间距

        # 绘制第一行
        cv2.putText(
            image,
            text_line1,
            (x1, y1 - 10),  # 基础偏移
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # 绘制第二行
        cv2.putText(
            image,
            text_line2,
            (x1, y1 - 10 - vertical_offset),  # 应用计算偏移
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    cv2.imwrite(output_path, image)

def pred(img_data, session, conf_threshold, iou_threshold, imgsz):
    raw_output, ratio, dwdh = do_inference(session=session, img_data=img_data.copy(), imgsz=imgsz)  # 执行推理
    det_objs = post_process(raw_output, ratio, dwdh, conf_threshold=conf_threshold, iou_threshold=iou_threshold)  # 解析输出
    return det_objs


def draw_result(det_objs, ori_img_path, output_path,id_to_name):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    img_data = cv2.imread(ori_img_path)
    save_detections(img_data, det_objs, output_path,id_to_name)  # 保存检测结果

# 主函数和其他工具函数需要相应调整

from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_single_image(args):
    """
    单张图像处理函数，适用于多进程调用
    返回格式：(预测框列表, 真值框列表)
    """
    img_path,label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name = args
    
    # 每个进程独立加载模型（避免共享资源冲突）
    session = load_model(model_weights)  # 需要确保load_model的线程安全性
    
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        return [], []
    h, w = img.shape[:2]
    
    # 推理
    det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)
    

    #可视化
    # output_path = output_dir / f"{img_path.stem}.jpg"
    # draw_result(det_objs, img_path, output_path,id_to_name)


    # 处理预测结果（修复归一化逻辑）
    pred_boxes = []
    for d in det_objs:
        x_center = (d["bbox"][0] + d["bbox"][2] / 2) / w  # Normalize center x
        y_center = (d["bbox"][1] + d["bbox"][3] / 2) / h  # Normalize center y
        width = d["bbox"][2] / w  # Normalize width
        height = d["bbox"][3] / h  # Normalize height
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
    output_dir = dataset_root / "output/viz" 
    task_args = [
        (p,label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name)
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
