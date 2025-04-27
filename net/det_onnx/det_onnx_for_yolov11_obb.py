import os
import cv2
import numpy as np
import onnxruntime as ort

from utils.general import convert_dict_to_list,gt_to_poly,process_file
import val_with_np_with_poly

import time
from functools import wraps
def average_execution_time(func):
    total_time = 0.0
    num_calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal total_time, num_calls
        start_time = time.perf_counter()  # 记录开始时间
        result = func(*args,**kwargs)    # 执行原函数
        end_time = time.perf_counter()    # 记录结束时间
        elapsed = end_time - start_time   # 计算单次耗时
        total_time += elapsed             # 累加总耗时
        num_calls += 1                    # 增加调用次数
        return result
    def reset():
        nonlocal total_time, num_calls
        total_time = 0.0
        num_calls = 0
    def get_average():
        """返回当前平均执行时间（单位：秒）"""
        return total_time / num_calls if num_calls > 0 else 0.0

    wrapper.get_average = get_average     # 附加方法用于获取平均值
    return wrapper

'''
ONNX output: x_center, y_center, width, height, cls1_confidence, ..., clsN_confidence, angle
'''


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
    session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
    return session


def _get_covariance_matrix(obb):
    '''
    计算 obb 的协方差矩阵

    Args:
        obb (np.ndarray) obb 数据，包含中心坐标、宽、高和旋转角度
    Return:
        协方差矩阵的三个元素 a, b, c
    '''
    widths = obb[..., 2] / 2
    heights = obb[..., 3] / 2
    angles = obb[..., 4]

    cos_angle = np.cos(angles)
    sin_angle = np.sin(angles)

    a = (widths * cos_angle) ** 2 + (heights * sin_angle) ** 2
    b = (widths * sin_angle) ** 2 + (heights * cos_angle) ** 2
    c = widths * cos_angle * heights * sin_angle

    return a, b, c


def batch_obbiou(obb1, obb2, eps=1e-7):
    '''
    计算 obb 间的 IOU

    Args:
        obb1 (np.ndarray): 第一个旋转边界框集合
        obb2 (np.ndarray): 第二个旋转边界框集合
        eps (float): 防止除零的极小值
    Return:
        两个旋转边界框之间的 ProbIoU
    '''
    x1, y1 = obb1[..., 0], obb1[..., 1]
    x2, y2 = obb2[..., 0], obb2[..., 1]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1[:, None] + a2) * (y1[:, None] - y2) ** 2 + (b1[:, None] + b2) * (x1[:, None] - x2) ** 2)
        / ((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2 + eps)
        * 0.25
    )
    t2 = (
        ((c1[:, None] + c2) * (x2 - x1[:, None]) * (y1[:, None] - y2))
        / ((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2 + eps)
        * 0.5
    )
    t3 = (
        np.log(
            ((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2) / (4 * np.sqrt((a1 * b1 - c1**2)[:, None] * (a2 * b2 - c2**2)) + eps)
            + eps
        )
        * 0.5
    )

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def obb_nms(boxes, scores, iou_threshold=0.5):
    '''
    OBB NMS

    Args:
        boxes (np.ndarray) obb
        scores (np.ndarray) 置信度得分
        iou_threshold (float) IoU 阈值
    Return:
        保留的边界框索引列表
    '''
    order = scores.argsort()[::-1]  # 根据置信度得分降序排序
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        remaining_boxes = boxes[order[1:]]
        iou_values = batch_obbiou(boxes[i : i + 1], remaining_boxes).squeeze(0)

        mask = iou_values < iou_threshold  # 保留 IoU 小于阈值的框
        order = order[1:][mask]

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
    '''
    onnx 结果的后处理

    Args:
        output (np.ndarray): ONNX 输出，包含边界框信息
        ratio (tuple): 缩放比例
        dwdh (tuple): 填充的宽高
        conf_threshold (float): 置信度阈值
        iou_threshold (float): IoU 阈值
    Return:
        筛选后的结果
    '''
    boxes, scores, classes, detections = [], [], [], []
    num_detections = output.shape[2]  # 获取检测的边界框数量
    num_classes = output.shape[1] - 6  # 计算类别数量

    for i in range(num_detections):
        detection = output[0, :, i]
        x_center, y_center, width, height = detection[0], detection[1], detection[2], detection[3]
        angle = detection[-1]  # 提取旋转角度

        if num_classes > 0:
            class_confidences = detection[4 : 4 + num_classes]
            if class_confidences.size == 0:
                continue
            class_id = np.argmax(class_confidences)
            confidence = class_confidences[class_id]
        else:
            confidence = detection[4]
            class_id = 0  # 默认 0

        if confidence > conf_threshold:
            x_center = (x_center - dwdh[0]) / ratio[0]
            y_center = (y_center - dwdh[1]) / ratio[1]
            width /= ratio[0]
            height /= ratio[1]

            boxes.append([x_center, y_center, width, height, angle])
            scores.append(confidence)
            classes.append(class_id)

    if not boxes:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    # NMS
    keep_indices = obb_nms(boxes, scores, iou_threshold=iou_threshold)

    # final results
    for idx in keep_indices:
        x_center, y_center, width, height, angle = boxes[idx]
        confidence = scores[idx]
        class_id = classes[idx]
        obb_corners = calc_obb_corners(x_center, y_center, width, height, angle)

        detections.append(
            {
                "position": obb_corners,  # obb角点坐标
                "confidence": float(confidence),
                "class_id": int(class_id),
                "angle": float(angle),
            }
        )

    return detections


def calc_obb_corners(x_center, y_center, width, height, angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dx = width / 2
    dy = height / 2

    corners = [
        (int(x_center + cos_angle * dx - sin_angle * dy), int(y_center + sin_angle * dx + cos_angle * dy)),
        (int(x_center - cos_angle * dx - sin_angle * dy), int(y_center - sin_angle * dx + cos_angle * dy)),
        (int(x_center - cos_angle * dx + sin_angle * dy), int(y_center - sin_angle * dx - cos_angle * dy)),
        (int(x_center + cos_angle * dx + sin_angle * dy), int(y_center + sin_angle * dx - cos_angle * dy)),
    ]
    return corners


def save_detections(image, detections, output_path):
    for det in detections:
        corners = det["position"]
        confidence = det["confidence"]
        class_id = det["class_id"]

        for j in range(4):
            pt1 = corners[j]
            pt2 = corners[(j + 1) % 4]
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)

        cv2.putText(
            image,
            f"Class: {class_id}, Conf: {confidence:.2f}",
            (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )

    cv2.imwrite(output_path, image)


def pred(img_data, session, conf_threshold, iou_threshold, imgsz):
    raw_output, ratio, dwdh = do_inference(session=session, img_data=img_data.copy(), imgsz=imgsz)  # 执行推理
    det_objs = post_process(raw_output, ratio, dwdh, conf_threshold=conf_threshold, iou_threshold=iou_threshold)  # 解析输出
    return det_objs


def draw_result(det_objs, ori_img_path, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    img_data = cv2.imread(ori_img_path)
    save_detections(img_data, det_objs, output_path)  # 保存检测结果


# 主函数：加载参数
if __name__ == "__main__":
    if True:

        image_path = r"input/images/P0000.jpg"
        model_weights = r"weights/yolov11_512_fp32.onnx"
        conf_threshold = 0.5
        iou_threshold = 0.5
        imgsz = (512, 512)

        session = load_model(weights=model_weights)
        img = cv2.imread(image_path)
        det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)


        name_list = [
                "baseball-diamond",
                "basketball-court",
                "bridge",
                "container-crane",
                "ground-track-field",
                "harbor",
                "helicopter",
                "large-vehicle",
                "plane",
                "roundabout",
                "ship",
                "small-vehicle",
                "soccer-ball-field",
                "storage-tank",
                "swimming-pool",
                "tennis-court",
        ]
        name_to_id = {name: idx for idx, name in enumerate(name_list)}
        output_path = 'output/result.jpg'
        draw_result(det_objs, image_path, output_path)


        pred_poly=convert_dict_to_list(det_objs)
        pred_poly=np.array(pred_poly)

        height, width = img.shape[:2]
        ground_truths = np.array(process_file(r"input/labels/P0000.txt", name_to_id, (width, height)))
        gt_poly = gt_to_poly(ground_truths)

        pred_poly = pred_poly[pred_poly[:, -1]==8]
        gt_poly = gt_poly[gt_poly[:, -1]==8]

        avg_time = do_inference.get_average()
        print(f"Average inference time: {avg_time:.4f} seconds")
        
        results = val_with_np_with_poly.evaluate_obb_detection_with_poly(pred_poly, gt_poly)
        print("mAP:", results["mAP"])
        print("AP per class:", results["ap_per_class"])
        print("mAP per iou_thres:", results["ap_per_iou"])

        # print(pred_poly)
    else:
         # 配置参数
        model_weights = r"weights/yolov11_512_fp32.onnx"
        conf_threshold = 0.5
        iou_threshold = 0.5
        imgsz = (512, 512)
        img_dir = r"input/selected_images_and_labels/images"
        label_dir = r"input/selected_images_and_labels/labels"
        output_results_dir = "output/results"
        output_viz_dir = "output/viz"
        os.makedirs(output_results_dir, exist_ok=True)
        os.makedirs(output_viz_dir, exist_ok=True)

        name_list = [
            "baseball-diamond",
            "basketball-court",
            "bridge",
            "container-crane",
            "ground-track-field",
            "harbor",
            "helicopter",
            "large-vehicle",
            "plane",
            "roundabout",
            "ship",
            "small-vehicle",
            "soccer-ball-field",
            "storage-tank",
            "swimming-pool",
            "tennis-court",
        ]
        name_to_id = {name: idx for idx, name in enumerate(name_list)}

        # 初始化模型
        session = load_model(weights=model_weights)
        all_preds = []
        all_gts = []
        metrics = []

        # 遍历图片目录
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            try:
                img_path = os.path.join(img_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt")
                
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图像：{img_path}")
                    continue
                
                # 执行推理
                det_objs = pred(img, session, conf_threshold, iou_threshold, imgsz)
                
                # 处理真实标签
                height, width = img.shape[:2]
                ground_truths = process_file(label_path, name_to_id, (width, height))
                gt_poly = gt_to_poly(ground_truths)
                
                # 转换预测结果为多边形格式
                pred_poly = convert_dict_to_list(det_objs)
                pred_poly = np.array(pred_poly)
                
                # 保存原始数据
                np.save(os.path.join(output_results_dir, f"{base_name}_pred.npy"), pred_poly)
                np.save(os.path.join(output_results_dir, f"{base_name}_gt.npy"), gt_poly)
                
                # 可视化并保存
                viz_path = os.path.join(output_viz_dir, f"{base_name}_viz.jpg")
                draw_result(det_objs, img_path, viz_path)
                
                # 筛选特定类别（例如类别8）
                if len(pred_poly) > 0:
                    pred_eval = pred_poly[pred_poly[:, -1] == 8]
                    gt_eval = gt_poly[gt_poly[:, -1] == 8]
                    
                    if len(pred_eval) > 0 and len(gt_eval) > 0:
                        # 收集用于评估的数据
                        all_preds.append(pred_eval)
                        all_gts.append(gt_eval)
                        
            except Exception as e:
                print(f"处理文件 {img_path} 时出错: {str(e)}")
                continue

        # 计算全局指标
        if len(all_preds) > 0 and len(all_gts) > 0:
            final_preds = np.concatenate(all_preds, axis=0)
            final_gts = np.concatenate(all_gts, axis=0)
            results = val_with_np_with_poly.evaluate_obb_detection_with_poly(final_preds, final_gts)
            print(f"全局 mAP: {results['mAP']:.4f}")
            print("各类别 AP:", results["ap_per_class"])
            print("各IoU阈值下的 mAP:", results["ap_per_iou"])
        
        # 输出平均推理时间
        avg_time = do_inference.get_average()
        print(f"平均推理时间: {avg_time:.4f} 秒")