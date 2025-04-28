import numpy as np
from shapely.geometry import Polygon


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
    rect = np.array([[-width / 2, -height / 2],
                     [width / 2, -height / 2],
                     [width / 2, height / 2],
                     [-width / 2, height / 2]], dtype=np.float64)
    # 旋转矩阵
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val],
                                [sin_val, cos_val]], dtype=np.float64)
    # 应用旋转
    rotated_rect = np.dot(rect, rotation_matrix)
    # 平移到中心点
    rotated_rect[:, 0] += center_x
    rotated_rect[:, 1] += center_y
    return rotated_rect

def polygon_iou(poly1, poly2):
    """
    计算两个多边形之间的交并比 (IoU)。

    参数:
        poly1 (numpy.ndarray): 第一个多边形的顶点坐标，形状为 (n, 2)。
        poly2 (numpy.ndarray): 第二个多边形的顶点坐标，形状为 (m, 2)。

    返回:
        float: 两个多边形之间的 IoU 值。
    """
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



def evaluate_obb_detection(predictions, ground_truths, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    评估 OBB 目标检测模型的性能。

    参数:
        predictions (list): 模型预测结果的列表，每个元素是一个包含 (center_x, center_y, width, height, angle, score, class_id) 的列表。
        ground_truths (list): 真实标注的列表，每个元素是一个包含 (center_x, center_y, width, height, angle, class_id) 的列表。
        iou_thresholds (numpy.ndarray, 可选): 用于计算 mAP 的 IoU 阈值数组。默认为 np.arange(0.5, 1.0, 0.05)。

    返回:
        dict: 包含评估指标的字典，包括：
            - mAP (float): 平均精度均值。
            - ap_per_class (dict): 每个类别的 AP 值。
            - ap_per_iou (dict): 每个 IoU 阈值下的 AP 值。
    """
    # 初始化
    aps = []
    ap_per_class = {}
    ap_per_iou = {iou: [] for iou in iou_thresholds}  # 存储每个 IoU 的 AP
    mAp_per_iou = {iou: None for iou in iou_thresholds}  # 存储每个 IoU 的 mAP
    num_classes = len(set(gt[-1] for gt in ground_truths))  # 获取类别数量

    for class_id in range(num_classes):
        # 获取当前类别的预测和真实标注
        class_predictions = [p for p in predictions if p[-1] == class_id]
        class_ground_truths = [gt for gt in ground_truths if gt[-1] == class_id]

        # 如果没有该类别的GT，则跳过
        if not class_ground_truths:
            ap_per_class[class_id] = 0.0
            for iou in iou_thresholds:
                ap_per_iou[iou].append(0.0)
            aps.append(0.0)
            continue

        # 按照置信度排序预测结果
        class_predictions.sort(key=lambda x: x[-2], reverse=True)
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        # 存储GT是否已经被匹配
        gt_matched = np.zeros(len(class_ground_truths), dtype=bool)

        for i, pred in enumerate(class_predictions):
            best_iou = 0.0
            best_gt_idx = -1
            pred_obb = pred[:5]  # 获取预测框的 OBB 参数
            for j, gt in enumerate(class_ground_truths):
                if not gt_matched[j]:
                    gt_obb = gt[:5] # 获取GT框的OBB参数
                    iou = obb_iou(pred_obb, gt_obb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > iou_thresholds[0]:  # 使用最低的 IoU 阈值来判断是否匹配
                tp[i] = 1.0
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1.0

        # 计算精度和召回率
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / len(class_ground_truths)

        # 计算 AP
        ap = 0.0
        ap_iou_thresholds = []
        for t in iou_thresholds:
            # 在不同的 IoU 阈值下计算精度和召回率
            t_p = precision[recall >= t]
            if t_p.size > 0:
                ap_val = np.max(t_p)
                ap += ap_val
                ap_iou_thresholds.append(ap_val)  # 保存当前阈值的AP
                ap_per_iou[t].append(ap_val)  # 存储每个iou阈值的ap
            else:
                ap_iou_thresholds.append(0.0)
                ap_per_iou[t].append(0.0)
        ap /= len(iou_thresholds)  # 平均 AP
        mAp_per_iou = {t:np.mean(ap_per_iou[t]) for t in iou_thresholds} 
        ap_per_class[class_id] = ap
        aps.append(ap)

    # 计算 mAP
    mAP = np.mean(aps)
    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "ap_per_iou": mAp_per_iou,
    }


if __name__ == "__main__":
    # 示例预测结果
    predictions = [
        (100, 100, 50, 30, 0.5, 0.9, 0),  # (x, y, w, h, angle, score, class)
        (150, 150, 60, 40, 1.0, 0.8, 0),
        (200, 200, 55, 35, 0.2, 0.7, 1),
        (250, 250, 70, 45, 1.2, 0.95, 1),
    ]

    # 示例真实标注
    ground_truths = [
        (105, 105, 45, 25, 0.4, 0),
        (155, 155, 55, 35, 0.9, 0),
        (205, 205, 50, 30, 0.1, 1),
        (260, 260, 65, 40, 1.1, 1),
        (300, 300, 80, 50, 0.0, 2),  # 额外的GT，没有对应的预测
    ]

    # 评估模型
    results = evaluate_obb_detection(predictions, ground_truths)
    print("mAP:", results["mAP"])
    print("AP per class:", results["ap_per_class"])
    print("mAP per iou_thres:", results["ap_per_iou"])
