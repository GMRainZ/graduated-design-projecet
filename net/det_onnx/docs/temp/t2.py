def main():
    # 解析命令行参数
    args = parse_args()
    
    # 数据集配置（使用参数中的路径）
    dataset_root = Path("det_onnx")
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    
    # 其他原有配置...
    # 配置参数
    model_weights = "det_onnx/weights/v11_coco128_attention_50epoch.onnx"
    conf_threshold = 0.5
    iou_threshold = 0.5
    imgsz = (640, 640)
    num_workers = min(8, mp.cpu_count()//2)
    
    # 加载标签映射
    id_to_name, _ = load_label_mappings(r"det_onnx/docs/coco128.yaml")
    
    # 根据模式选择处理流程
    if args.mode.lower() == "video":
        # 验证视频路径参数
        if not args.video_input:
            raise ValueError("视频模式需要指定--video_input参数")
        if not args.video_output:
            raise ValueError("视频模式需要指定--video_output参数")
        
        # 处理视频
        process_video(
            input_path=args.video_input,
            output_path=args.video_output,
            model_weights=model_weights,
            id_to_name=id_to_name,
        )
    else:
        # 图片处理模式
        # 准备任务参数（在主进程完成数据预处理）
        image_paths = list(image_dir.glob("*.jpg"))
        # 输出目录配置
        output_dir = dataset_root / "output/viz"
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备任务参数（包含output_dir）
        task_args = [
            (p, label_dir, model_weights, conf_threshold, iou_threshold, imgsz, id_to_name, output_dir)
            for p in image_paths
        ]
        
        # 多进程处理（原有逻辑保持不变）...
        # ... [保留原有图片处理逻辑]
        