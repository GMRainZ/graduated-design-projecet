# detection_api.py
import argparse
import os
from pathlib import Path
from multiprocessing import get_start_method, set_start_method
from typing import Callable, List, Tuple, Dict, Optional
from utils.v11_key_op import load_model, pred, save_detections, process_hbb_gt, batch_evaluate_hbb, load_label_mappings
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


from det_onnx_hbb import *

class DetectionAPI:
    def __init__(self, 
                 model_weights: str = "weights/v11_coco128_attention_50epoch.onnx",
                 label_mapping_path: str = "docs/coco128.yaml",
                 default_conf: float = 0.5,
                 default_iou: float = 0.5,
                 default_imgsz: Tuple[int, int] = (640, 640)):
        """
        初始化检测API核心组件
        """
        # 多进程安全设置
        if get_start_method(allow_none=True) != 'spawn':
            set_start_method('spawn', force=True)
            
        # 加载模型和配置
        self.model_weights = model_weights
        self.session = load_model(model_weights)  # 预加载模型（如果不需要多进程）
        self.id_to_name, _ = load_label_mappings(label_mapping_path)
        
        # 默认参数
        self.default_conf = default_conf
        self.default_iou = default_iou
        self.default_imgsz = default_imgsz

    def process_image_batch(self,
                           image_paths: List[str],
                           label_dir: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           conf: Optional[float] = None,
                           iou: Optional[float] = None,
                           imgsz: Optional[Tuple[int, int]] = None,
                           num_workers: int = 4) -> Tuple[List[List], List[List]]:
        """
        批量处理图像并返回评估结果
        """
        # 参数处理
        conf = conf or self.default_conf
        iou = iou or self.default_iou
        imgsz = imgsz or self.default_imgsz
        num_workers = min(num_workers, os.cpu_count()//2)

        # 路径转换
        image_dir = Path(image_paths[0]).parent if len(image_paths) > 0 else Path()
        label_dir = Path(label_dir) if label_dir else image_dir.parent / "labels"
        output_dir = Path(output_dir) if output_dir else image_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 准备任务参数
        task_args = [
            (Path(p), label_dir, self.model_weights, conf, iou, imgsz, 
             self.id_to_name, output_dir)
            for p in image_paths
        ]

        # 多进程处理
        all_preds, all_truths = [], []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = executor.map(self._process_single_image, task_args)
            
            for future in tqdm(futures, total=len(task_args), desc="批量推理"):
                preds, truths = future
                all_preds.append(preds)
                all_truths.append(truths)

        return all_preds, all_truths

    def _process_single_image(self, args):
        """
        供多进程调用的单图处理（内部方法）
        """
        img_path, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name, output_dir = args
        
        # 每个进程独立加载模型（避免共享资源冲突）
        session = load_model(model_weights)
        
        img = cv2.imread(str(img_path))
        if img is None:
            return [], []
        h, w = img.shape[:2]
        
        # 推理
        det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)
        
        # 可视化输出
        output_path = output_dir / img_path.name
        save_detections(img.copy(), det_objs, str(output_path), id_to_name)
        
        # 处理预测结果
        pred_boxes = []
        for d in det_objs:
            x_center = d["bbox"][0] / w
            y_center = d["bbox"][1] / h
            width = d["bbox"][2] / w
            height = d["bbox"][3] / h
            pred_boxes.append([x_center, y_center, width, height, d["confidence"], int(d["class_id"])])
        
        # 处理真值标注
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = process_hbb_gt(str(label_path), id_to_name, (w, h)) if label_path.exists() else []
        
        return pred_boxes, gt_boxes

    def process_video(self,
                     input_path: str,
                     output_path: str,
                     conf: Optional[float] = None,
                     iou: Optional[float] = None,
                     imgsz: Optional[Tuple[int, int]] = None,
                     progress_callback: Optional[Callable] = None):
        """
        处理视频流
        """
        # 参数处理
        conf = conf or self.default_conf
        iou = iou or self.default_iou
        imgsz = imgsz or self.default_imgsz

        # 实际调用视频处理函数
        process_video(
            input_path=input_path,
            output_path=output_path,
            model_weights=self.model_weights,
            id_to_name=self.id_to_name,
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=imgsz,
            progress_callback=progress_callback,
        )

    def evaluate_mAP(self, all_preds: List[List], all_truths: List[List]) -> Dict:
        """
        计算评估指标
        """
        return batch_evaluate_hbb(all_preds, all_truths)

def main():
    """
    保留命令行接口
    """
    parser = argparse.ArgumentParser(description="目标检测推理API")
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--image_dir", help="图像目录路径")
    parser.add_argument("--label_dir", help="标签目录路径")
    parser.add_argument("--video_input", help="输入视频路径")
    parser.add_argument("--video_output", help="输出视频路径")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="IOU阈值")
    args = parser.parse_args()

    # 初始化API
    detector = DetectionAPI()

    if args.mode == "video":
        detector.process_video(
            input_path=args.video_input,
            output_path=args.video_output,
            conf=args.conf,
            iou=args.iou
        )
    else:
        image_paths = list(Path(args.image_dir).glob("*.jpg"))
        all_preds, all_truths = detector.process_image_batch(
            image_paths=[str(p) for p in image_paths],
            label_dir=args.label_dir,
            conf=args.conf,
            iou=args.iou
        )
        results = detector.evaluate_mAP(all_preds, all_truths)
        print(f"mAP@[0.5:0.95]: {results['mAP']:.4f}")

if __name__ == "__main__":
    main()