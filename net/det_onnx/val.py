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
    img_path, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name, output_dir, precision , is_v12= args
    
    # 每个进程独立加载模型（避免共享资源冲突）
    session = load_model(model_weights)
    
    img = cv2.imread(str(img_path))
    if img is None:
        return [], []
    
    # 根据精度设置调整图像数据类型
    if precision == "fp16":
        img = img.astype(np.float16) 
    elif precision == "fp32":
        img = img.astype(np.float32) 
    elif precision == "int8":
        img = img.astype(np.int8)
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    # 将图像数据转换回 uint8 类型以便显示
    # img_to_show = (img * 255).astype(np.uint8)
    # cv2.imshow("Original Image", img_to_show)
    # cv2.waitKey(0)

    h, w = img.shape[:2]
    
    # 推理
    det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz,precision=precision,is_v12=is_v12)
    
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

def main():
    # Parse arguments
    args = parse_args()
    
    # 配置参数
    model_weights = args.model_weights
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    imgsz = args.img_siz
    is_v12 = args.is_v12
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
        all_preds, all_truths = [], []
        
        # Sequential processing of images
        precision = args.precision
        for img_path in tqdm(image_paths, desc="顺序推理"):
            preds, truths = process_single_image(
                (img_path, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name, output_dir,precision,is_v12)
            )
            all_preds.append(preds)
            all_truths.append(truths)
        
        # 计算mAP
        results = batch_evaluate_hbb(all_preds, all_truths)
        
        # 输出结果
        print("\nEvaluation Results:")
        print(f"mAP@[0.5:0.95]: {results['mAP']:.4f}")
        print(f"总处理图像数: {len(all_preds)}")

        print_executed_time()
        # print(f"Executed Time: {do_inference.get_average()} ms")

def process_video(input_path, output_path, model_weights, id_to_name, conf_threshold, iou_threshold, imgsz):
    """
    视频处理函数，改为顺序处理视频帧
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
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理单帧
        det_objs = pred(frame, session, conf_threshold, iou_threshold, imgsz)
        save_detections(frame.copy(), det_objs, f"frame_{frame_id}.jpg", id_to_name)
        
        # 写入输出视频
        out.write(frame)
        frame_id += 1
    
    # 释放资源
    cap.release()
    out.release()

# 新增参数解析部分
import argparse

# 在main函数前添加参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="目标检测推理脚本")
    parser.add_argument("--model_weights", type=str, default="weights/v11_attention_fp16.onnx", help="模型权重路径")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IOU阈值")
    parser.add_argument("--img_siz",type=tuple, default=(640,640), help="图像尺寸")
    parser.add_argument("--video_input", type=str, default="input/videoes/demo.mp4", help="输入视频路径")
    parser.add_argument("--video_output", type=str, default="output/videoes/demo_result.mp4", help="输出视频路径")
    parser.add_argument("--image_dir", type=str, default="input/images/val", help="输入图片目录")
    parser.add_argument("--label_dir", type=str, default="input/labels/val", help="标签目录")
    parser.add_argument("--is_v12", type=bool, default=False, help="是否为v12模型")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16","int8"], help="精度：fp32或fp16")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"], help="运行模式：image或video")
    return parser.parse_args()



if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    main()