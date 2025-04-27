import os
import cv2
import numpy as np
import onnxruntime as ort
from utils.general import convert_dict_to_list, process_hbb_gt,load_label_mappings
import val_with_np  # 假设这是处理水平框的评估模块
import time
from functools import wraps

from tqdm import tqdm
from pathlib import Path
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


def do_inference(session, img_data, imgsz=(640, 640),precision="fp32",is_v12=False):
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
    if precision == "fp16":
        img = img[np.newaxis, ...].astype(np.float16) / 255.0
    elif precision == "fp32":
        img = img[np.newaxis, ...].astype(np.float32) / 255.0
    elif precision == "int8":
        img = img[np.newaxis, ...].astype(np.int8) // 255 #应用整数除法
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    
    result = _do_inference_(session,img)
    
    

    # output_v11_format = output_v12.permute(0, 2, 1)  # 转换为 (1, 84, 8400)
    if is_v12:
        output = np.transpose(np.array(result[0]),axes=(0,2,1))
        return output, ratio, (dw, dh)


    return result[0], ratio, (dw, dh)

@average_execution_time
def _do_inference_(session,img):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img})

    return result

def print_executed_time():
    print(f"Executed Time: {_do_inference_.get_average()} ms")

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

    return image

def pred(img_data, session, conf_threshold, iou_threshold, imgsz,precision="fp32",is_v12=True):
    raw_output, ratio, dwdh = do_inference(session=session, img_data=img_data.copy(), imgsz=imgsz,precision=precision,is_v12=is_v12)  # 执行推理
    det_objs = post_process(raw_output, ratio, dwdh, conf_threshold=conf_threshold, iou_threshold=iou_threshold)  # 解析输出
    return det_objs


def draw_result(det_objs, ori_img_path, output_path,id_to_name):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    img_data = cv2.imread(ori_img_path)
    save_detections(img_data, det_objs, output_path,id_to_name)  # 保存检测结果

# 主函数和其他工具函数需要相应调整


def get_valid_encoders():
    """获取系统支持的视频编码器列表"""
    test_encoders = ['mp4v', 'avc1', 'x264', 'h264', 'XVID', 'DIVX']
    return [enc for enc in test_encoders 
           if cv2.VideoWriter_fourcc(*enc) != -1]

def create_video_writer(output_path, frame_size, fps):
    """增强版编码器选择函数"""
    ext = Path(output_path).suffix.lower()
    encoder_priority = {
        '.mp4': ['avc1', 'mp4v', 'x264', 'h264'],
        '.avi': ['XVID', 'DIVX', 'MJPG'],
        '.mov': ['mp4v', 'avc1']
    }
    
    # 获取系统支持的编码器
    valid_encoders = [enc for enc in ['avc1','mp4v','XVID','MJPG','x264'] 
                     if cv2.VideoWriter_fourcc(*enc) != -1]
    
    # 按优先级尝试编码器
    for codec in encoder_priority.get(ext, ['mp4v']):
        if codec in valid_encoders:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
            if writer.isOpened():
                return writer, codec
    
    # 保底方案：使用原始帧保存
    print("无法找到视频编码器，将保存为图像序列")
    return None, 'image_sequence'



def process_video(
    input_path,
    output_path,
    model_weights,
    id_to_name,
    conf_threshold=0.5,
    iou_threshold=0.5,
    imgsz=(640, 640),
    show_preview=False,
    target_fps=None,
    skip_frames=0,
    precision="fp32"
):
    """
    视频处理主函数
    Args:
        skip_frames: 跳帧处理间隔（0=不跳帧）
    """
    # 初始化视频流
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {input_path}")

    # 获取视频参数
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算实际处理帧率
    process_fps = target_fps or original_fps
    frame_interval = int(original_fps / process_fps) if target_fps else 1
    frame_interval = max(1, frame_interval)  # 至少处理每帧
    
    # 创建视频写入器
    writer, used_codec = create_video_writer(
        output_path, 
        (frame_width, frame_height),
        process_fps
    )
    print(f"视频编码使用：{used_codec}，目标帧率：{process_fps:.1f}fps")

    # 进度条初始化
    pbar = tqdm(total=total_frames, desc="处理视频", unit="frame")
    frame_count = 0
    processed_count = 0
    start_time = time.time()


    model = load_model(model_weights)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧处理
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                pbar.update(1)
                continue

            # 执行推理
            det_objs = pred(frame, model, conf_threshold, iou_threshold, imgsz,precision=precision)
            
            # 绘制结果
            annotated_frame = frame.copy()
            annotated_frame = save_detections(annotated_frame, det_objs, None, id_to_name)
            
            # 写入视频
            writer.write(annotated_frame)
            processed_count += 1


            # 显示预览
            if show_preview:
                cv2.imshow('Preview', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)
            frame_count += 1

    except Exception as e:
        print(f"处理中断：{str(e)}")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        pbar.close()
        
        # 性能统计
        duration = time.time() - start_time
        print(f"处理完成，耗时：{duration:.1f}s")
        print(f"总帧数：{frame_count}，处理帧数：{processed_count}")
        print(f"实际处理速度：{processed_count/duration:.1f}fps")