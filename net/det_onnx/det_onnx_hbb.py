from utils.v11_key_op import *

import cv2 

from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_single_image(args):
    """
    单张图像处理函数，适用于多进程调用
    返回格式：(预测框列表, 真值框列表)
    """
    img_path, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name, output_dir = args
    
    # 每个进程独立加载模型（避免共享资源冲突）
    session = load_model(model_weights)
    
    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        return [], []
    h, w = img.shape[:2]
    
    # 推理
    det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)
    
    # 可视化：添加画图功能
    output_path = output_dir / img_path.name  # 构造输出路径
    save_detections(img.copy(), det_objs, str(output_path), id_to_name)  # 使用copy避免修改原始图像
    
    # 处理预测结果（修复归一化逻辑）
    pred_boxes = []
    for d in det_objs:
        x_center = d["bbox"][0] / w  # 正确归一化中心坐标
        y_center = d["bbox"][1] / h
        width = d["bbox"][2] / w
        height = d["bbox"][3] / h
        pred_boxes.append([x_center, y_center, width, height, d["confidence"], int(d["class_id"])])
    
    # 处理真值标注
    label_path = label_dir / f"{img_path.stem}.txt"
    gt_boxes = process_hbb_gt(str(label_path), id_to_name, (w, h)) if label_path.exists() else []
    
    return pred_boxes, gt_boxes

@average_execution_time
def process_video(input_path, output_path, model_weights, id_to_name,conf_threshold, iou_threshold, imgsz):
    """
    视频处理函数，支持并行处理视频帧
    """
    # 加载模型
    session = load_model(model_weights)
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 并行处理视频帧
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    
    # 定义任务队列和结果队列
    task_queue = Queue()
    result_queue = Queue()
    
    # 定义单帧处理函数
    def process_frame(frame, frame_id):
        h, w = frame.shape[:2]
        det_objs = pred(frame, session, conf_threshold, iou_threshold, imgsz)
        save_detections(frame.copy(), det_objs, f"frame_{frame_id}.jpg", id_to_name)
        result_queue.put((frame_id, det_objs))
    
    # 启动线程池
    num_workers = min(8, mp.cpu_count() // 2)  # 动态调整线程池大小
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            task_queue.put((frame, frame_id))
            executor.submit(process_frame, *task_queue.get())  # 提交任务到线程池
            frame_id += 1
        
        # 等待所有任务完成
        while not result_queue.empty():
            frame_id, det_objs = result_queue.get()
            # 将处理后的帧写入输出视频
            out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()

def main():
    # Parse arguments
    args = parse_args()
    
    # 配置参数
    model_weights = "weights/v11_coco128_attention_50epoch.onnx"
    conf_threshold = 0.5
    iou_threshold = 0.5
    imgsz = (640, 640)
    num_workers = min(8, mp.cpu_count() // 2)  # 根据CPU核心数自动调整
    
    # 数据集配置
    dataset_root = Path(".")
    image_dir = Path(args.image_dir)  # Use parsed argument
    label_dir = Path(args.label_dir)  # Use parsed argument
    
    # 加载标签映射
    id_to_name, _ = load_label_mappings(r"docs/coco128.yaml")
    
    # 输出目录配置
    output_dir = dataset_root / "output/viz"
    os.makedirs(output_dir, exist_ok=True)  # 预先创建目录
    
    if args.mode == "video":  # Use parsed argument for mode

        process_video(
            input_path=args.video_input,  # Use parsed argument
            output_path=args.video_output,  # Use parsed argument
            model_weights=model_weights,
            id_to_name=id_to_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )
    else:
        # 准备任务参数（包含output_dir）
        image_paths = list(image_dir.glob("*.jpg"))
        task_args = [
            (p, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name, output_dir)
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

        # 输出推理时间 多线程下不适用
        # avg_time = do_inference.get_average()
        # print(f"Average inference time: {avg_time:.4f}ms")

    if args.mode == "image":
        # 输出结果
        print("\nEvaluation Results:")
        print(f"mAP@[0.5:0.95]: {results['mAP']:.4f}")
        print(f"总处理图像数: {len(all_preds)}")
        print(f"使用进程数: {num_workers}")
    else:
        print("Evaluation Results:")
        print(f"Excuted in {process_video.get_average():.2f} seconds")

# 新增参数解析部分
import argparse

# 在main函数前添加参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="目标检测推理脚本")
    parser.add_argument("--video_input", type=str, default="input/videoes/demo.mp4", help="输入视频路径")
    parser.add_argument("--video_output", type=str, default="output/videoes/demo_result.mp4", help="输出视频路径")
    parser.add_argument("--image_dir", type=str, default="input/images/val", help="输入图片目录")
    parser.add_argument("--label_dir", type=str, default="input/labels/val", help="标签目录")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"], help="运行模式：image或video")
    return parser.parse_args()



if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    main()