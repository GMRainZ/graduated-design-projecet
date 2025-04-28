import sys

import cv2
import numpy as np
from shapely.geometry import Polygon

sys.path.append(".")
from draw_with_poly import show_obb, show_obb_with_poly


def poly_label(img, poly, label="", color=(128, 128, 128), txt_color=(255, 255, 255), thickness=2, fontScale=2, cls=None):
    # 绘制旋转后的矩形
    for i in range(len(poly)):
        rotated_points = poly[i, 0:8].reshape(-1, 2).astype(np.int32)
        class_id = poly[i, -1]
        if cls is None or class_id == cls:
            cv2.polylines(img, [rotated_points], True, color, 2)

            # 添加类别ID和得分作为标签
            # score = poly[i, 8]
            # label = f"Class {class_id}: {score:.2f}"
            label = f"{class_id}"
            cv2.putText(img, label, (rotated_points[0, 0], rotated_points[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def obb2poly(center_x, center_y, width, height, angle):
    """
    将 OBB (中心点坐标, 宽度, 高度, 角度) 转换为多边形顶点坐标。

    参数:
        center_x (float): OBB 中心点的 x 坐标。
        center_y (float): OBB 中心点的 y 坐标。
        width (float): OBB 的宽度。
        height (float): OBB 的高度。
        angle (float): OBB 的旋转角度，以弧度为单位。

    返回:
        numpy.ndarray: 包含多边形四个顶点坐标的数组，形状为 (4, 2)。
    """
    # 避免原地修改，创建新的数组
    rect = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]], dtype=np.float64)
    # 旋转矩阵
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype=np.float64)
    # 应用旋转
    rotated_rect = np.dot(rect, rotation_matrix)
    # 平移到中心点
    rotated_rect[:, 0] += center_x
    rotated_rect[:, 1] += center_y

    return rotated_rect.flatten()


def polygon_iou(poly1, poly2):
    """
    计算两个多边形之间的交并比 (IoU)。

    参数:
        poly1 (numpy.ndarray): 第一个多边形的顶点坐标，形状为 (n, 2)。
        poly2 (numpy.ndarray): 第二个多边形的顶点坐标，形状为 (m, 2)。

    返回:
        float: 两个多边形之间的 IoU 值。
    """
    poly1 = np.asarray(poly1).reshape(-1, 2)
    poly2 = np.asarray(poly2).reshape(-1, 2)
    # 创建 Shapely Polygon 对象
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    # 计算交集面积
    intersection_area = polygon1.intersection(polygon2).area
    # 计算并集面积
    union_area = polygon1.union(polygon2).area
    # 避免除以零
    if union_area == 0:
        return 0.0
    # 计算 IoU
    iou = intersection_area / union_area
    return iou


def obb_iou(obb1, obb2):
    """
    计算两个 OBB 之间的交并比 (IoU)。

    参数:
        obb1 (tuple): 第一个 OBB 的参数 (center_x, center_y, width, height, angle)。
        obb2 (tuple): 第二个 OBB 的参数 (center_x, center_y, width, height, angle)。

    返回:
        float: 两个 OBB 之间的 IoU 值。
    """
    # 将 OBB 转换为多边形
    poly1 = obb2poly(*obb1)
    poly2 = obb2poly(*obb2)
    # 计算多边形 IoU
    return polygon_iou(poly1, poly2)

def calculate_ap(recall, precision):
    '''PASCAL VOC 11点法'''
    ap = 0
    # 在 11 个召回率点（0.0, 0.1, ..., 1.0）计算精度最大值
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if any(recall >= t) else 0
        ap += p
    return ap / 11

def evaluate_obb_detection_with_poly(predictions, ground_truths, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    评估 OBB 目标检测模型的性能。

    参数:
        predictions (list): 模型预测结果的列表，每个元素是一个包含 (x1, y1, x2, y2, x3, y3, x4, y4, score, class_id) 的列表。
        ground_truths (list): 真实标注的列表，每个元素是一个包含 (x1, y1, x2, y2, x3, y3, x4, y4, class_id) 的列表。
        iou_thresholds (numpy.ndarray, 可选): 用于计算 mAP 的 IoU 阈值数组。默认为 np.arange(0.5, 1.0, 0.05)。

    返回:
        dict: 包含评估指标的字典，包括：
            - mAP (float): 平均精度均值。
            - ap_per_class (dict): 每个类别的 AP 值。
            - ap_per_iou (dict): 每个 IoU 阈值下的平均 AP 值。
    """
    aps = []
    ap_per_class = {}
    ap_per_iou = {iou: [] for iou in iou_thresholds}  # 存储每个 IoU 阈值的 AP 列表（按类别）

    num_classes = set(gt[-1] for gt in ground_truths)

    for class_id in num_classes:
        class_predictions = [p for p in predictions if p[-1] == class_id]
        class_ground_truths = [gt for gt in ground_truths if gt[-1] == class_id]

        if not class_ground_truths:
            ap_per_class[class_id] = 0.0
            for iou in iou_thresholds:
                ap_per_iou[iou].append(0.0)
            aps.append(0.0)
            continue

        # 按置信度排序预测结果
        class_predictions.sort(key=lambda x: x[-2], reverse=True)

        # 预处理：计算每个预测框的最佳 IoU 和对应的 GT 索引
        best_ious = []
        best_gt_indices = []
        for pred in class_predictions:
            pred_obb = pred[:8]
            best_iou_val = 0.0
            best_gt_idx = -1
            for j, gt in enumerate(class_ground_truths):
                gt_obb = gt[:8]
                iou = polygon_iou(pred_obb, gt_obb)
                if iou > best_iou_val:
                    best_iou_val = iou
                    best_gt_idx = j
            best_ious.append(best_iou_val)
            best_gt_indices.append(best_gt_idx)

        # 对每个 IoU 阈值计算 AP
        class_aps = []
        for iou_threshold in iou_thresholds:
            tp = np.zeros(len(class_predictions))
            fp = np.zeros(len(class_predictions))
            gt_matched = np.zeros(len(class_ground_truths), dtype=bool)

            for i in range(len(class_predictions)):
                if best_ious[i] > iou_threshold:
                    if not gt_matched[best_gt_indices[i]]:
                        tp[i] = 1
                        gt_matched[best_gt_indices[i]] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)

            # 避免除以零
            precision = np.divide(
                cumulative_tp,
                (cumulative_tp + cumulative_fp),
                out=np.zeros_like(cumulative_tp),
                where=(cumulative_tp + cumulative_fp) != 0
            )
            recall = cumulative_tp / len(class_ground_truths)

            ap = calculate_ap(recall, precision)
            ap_per_iou[iou_threshold].append(ap)
            class_aps.append(ap)

        # 当前类别的 AP 是所有 IoU 阈值的平均
        avg_class_ap = np.mean(class_aps)
        ap_per_class[class_id] = avg_class_ap
        aps.append(avg_class_ap)

    # 计算每个 IoU 阈值的平均 AP（mAP per IoU）
    mAp_per_iou = {
        iou: np.mean(ap_list) if ap_list else 0.0
        for iou, ap_list in ap_per_iou.items()
    }

    # 计算整体 mAP（所有类别的 AP 平均）
    mAP = np.mean(aps)

    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "ap_per_iou": mAp_per_iou,  # 每个 IoU 阈值的平均 AP
    }


def resize_obb(obb, original_width, original_height, resized_width=512, resized_height=512):
    """
    将 OBB 结果 resize 转换到 original

    Args:
        obb (tuple): OBB  (center_x, center_y, width, height, angle)
        original_width (int): 原始图像的宽度
        original_height (int): 原始图像的高度
        resized_width (int): 调整大小后的图像宽度
        resized_height (int): 调整大小后的图像高度

    返回:
        tuple: 转换后的 OBB 参数 (center_x, center_y, width, height, angle)
    """
    center_x, center_y, width, height, angle = obb

    width_k = original_width / resized_width
    height_k = original_height / resized_height

    original_center_x = center_x * width_k
    original_center_y = center_y * height_k
    original_width = width * width_k
    original_height = height * height_k

    return original_center_x, original_center_y, original_width, original_height, angle


if __name__ == "__main__":
    # 示例预测结果
    predictions = [
        (100, 100, 50, 30, 0.5, 0.9, 0),  # (x, y, w, h, angle, score, class)
        (150, 150, 60, 40, 1.0, 0.8, 0),
        (200, 200, 55, 35, 0.2, 0.7, 1),
        (250, 250, 70, 45, 1.2, 0.95, 1),
    ]
    predictions_poly = [
        [70, 98, 114, 74, 129, 101, 85, 125, 0, 0],
        [116, 164, 149, 113, 183, 135, 150, 186, 0, 0],
        [169, 188, 223, 177, 230, 211, 176, 222, 0, 1],
        [216, 274, 241, 209, 283, 225, 258, 290, 0, 1],
    ]

    # 示例真实标注
    ground_truths = [
        (105, 105, 45, 25, 0.4, 0),
        (155, 155, 55, 35, 0.9, 0),
        (205, 205, 50, 30, 0.1, 1),
        (260, 260, 65, 40, 1.1, 1),
        (300, 300, 80, 50, 0.0, 2),  # 额外的GT，没有对应的预测
    ]
    gt_poly = [
        [79, 102, 120, 84, 130, 107, 89, 125, 0],
        [124, 165, 158, 122, 185, 144, 151, 187, 0],
        [178, 192, 228, 187, 231, 217, 181, 222, 1],
        [227, 279, 256, 221, 292, 240, 263, 298, 1],
        [260, 275, 340, 275, 340, 325, 260, 325, 2],
    ]

    # predictions = np.load("predictions.npy").squeeze()
    # predictions[:, -3] = np.rad2deg(predictions[:, -3])
    # ground_truths = np.load("ground_truths.npy")

    # 绘制结果图
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    show_obb_with_poly(img, predictions, ground_truths, cls_id=0)

    # 评估模型
    results = evaluate_obb_detection_with_poly(predictions_poly, gt_poly)
    print("mAP:", results["mAP"])
    print("AP per class:", results["ap_per_class"])
    print("mAP per iou_thres:", results["ap_per_iou"])
