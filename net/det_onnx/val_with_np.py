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
    # 创建一个矩形框，表示未旋转的OBB
    rect = np.array([[-width / 2, -height / 2],
                     [width / 2, -height / 2],
                     [width / 2, height / 2],
                     [-width / 2, height / 2]], dtype=np.float64)
    # 计算旋转矩阵
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val],
                                [sin_val, cos_val]], dtype=np.float64)
    # 应用旋转矩阵到矩形框
    rotated_rect = np.dot(rect, rotation_matrix)
    # 将旋转后的矩形框平移到中心点
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
    # 使用Shapely库创建多边形对象
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    # 计算交集面积
    intersection_area = polygon1.intersection(polygon2).area
    # 计算并集面积
    union_area = polygon1.union(polygon2).area
    # 避免除以零的情况
    if union_area == 0:
        return 0.0
    # 计算IoU值
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
    # 将OBB转换为多边形
    poly1 = obb2poly(*obb1)
    poly2 = obb2poly(*obb2)
    # 计算多边形IoU
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
    # 初始化变量
    aps = []
    ap_per_class = {}
    ap_per_iou = {iou: [] for iou in iou_thresholds}
    num_classes = len(set(gt[-1] for gt in ground_truths))

    # 遍历每个类别
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
        gt_matched = np.zeros(len(class_ground_truths), dtype=bool)

        # 遍历每个预测框
        for i, pred in enumerate(class_predictions):
            best_iou = 0.0
            best_gt_idx = -1
            pred_obb = pred[:5]  # 获取预测框的 OBB 参数
            for j, gt in enumerate(class_ground_truths):
                if not gt_matched[j]:
                    gt_obb = gt[:5]  # 获取GT框的OBB参数
                    iou = obb_iou(pred_obb, gt_obb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # 判断是否匹配
            if best_iou > iou_thresholds[0]:
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
        for t in iou_thresholds:
            t_p = precision[recall >= t]
            if t_p.size > 0:
                ap_val = np.max(t_p)
                ap += ap_val
                ap_per_iou[t].append(ap_val)
            else:
                ap_per_iou[t].append(0.0)
        ap /= len(iou_thresholds)
        ap_per_class[class_id] = ap
        aps.append(ap)

    # 计算 mAP
    mAP = np.mean(aps)
    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "ap_per_iou": ap_per_iou,
    }

def hbb_iou(box1, box2):
    """
    计算两个水平边界框的 IoU
    输入格式：[x_center, y_center, width, height]
    """
    # 转换为角点坐标 (xmin, ymin, xmax, ymax)
    box1 = [
        box1[0] - box1[2]/2, 
        box1[1] - box1[3]/2,
        box1[0] + box1[2]/2,
        box1[1] + box1[3]/2
    ]
    box2 = [
        box2[0] - box2[2]/2,
        box2[1] - box2[3]/2,
        box2[0] + box2[2]/2,
        box2[1] + box2[3]/2
    ]

    # 计算交集区域
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

def evaluate_hbb_detection(predictions, ground_truths, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    评估 HBB 目标检测模型的性能（支持多 IoU 阈值 mAP 计算）

    参数:
        predictions (list): 模型预测结果列表，每个元素格式为 [x_center, y_center, width, height, score, class_id]
        ground_truths (list): 真实标注列表，每个元素格式为 [x_center, y_center, width, height, class_id]
        iou_thresholds (numpy.ndarray, 可选): IoU 阈值数组，默认 np.arange(0.5, 1.0, 0.05)

    返回:
        dict: 包含评估指标的字典：
            - mAP (float): 平均精度均值
            - ap_per_class (dict): 每个类别的 AP 值
            - ap_per_iou (dict): 每个 IoU 阈值下的 AP 值
    """
    # 初始化变量
    aps = []
    ap_per_class = {}
    ap_per_iou = {iou: [] for iou in iou_thresholds}
    num_classes = len(set(gt[-1] for gt in ground_truths))

    # 遍历每个类别
    for class_id in range(num_classes):
        # 提取当前类别的预测和标注
        class_predictions = [p for p in predictions if p[-1] == class_id]
        class_ground_truths = [gt[:-1] for gt in ground_truths if gt[-1] == class_id]

        if not class_ground_truths:
            ap_per_class[class_id] = 0.0
            for iou in iou_thresholds:
                ap_per_iou[iou].append(0.0)
            aps.append(0.0)
            continue

        # 按置信度排序预测结果
        class_predictions.sort(key=lambda x: x[-2], reverse=True)
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        gt_matched = np.zeros(len(class_ground_truths), dtype=bool)

        # 遍历每个预测框
        for i, pred in enumerate(class_predictions):
            best_iou = 0.0
            best_gt_idx = -1
            pred_box = pred[:4]  # 仅取中心坐标+宽高
            for j, gt in enumerate(class_ground_truths):
                if not gt_matched[j]:
                    iou = hbb_iou(pred_box, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # 判断是否匹配
            if best_iou > iou_thresholds[0]:
                tp[i] = 1.0
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1.0

        # 计算精度和召回率
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)
        recall = cumulative_tp / len(class_ground_truths)

        # 计算 AP
        ap = 0.0
        for t in iou_thresholds:
            relevant_indices = np.where(recall >= t)[0]
            if relevant_indices.size > 0:
                max_precision = np.max(precision[relevant_indices])
                ap += max_precision
                ap_per_iou[t].append(max_precision)
            else:
                ap_per_iou[t].append(0.0)
        ap /= len(iou_thresholds)
        
        ap_per_class[class_id] = ap
        aps.append(ap)

    # 计算最终指标
    mAP = np.mean(aps)
    mAP_per_iou = {t: np.mean([ap for ap in ap_per_iou[t] if ap is not None]) for t in iou_thresholds}
    
    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "ap_per_iou": mAP_per_iou
    }


from collections import defaultdict

def batch_evaluate_hbb(preds_list, truths_list, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    批量评估多张图片的HBB检测性能
    
    参数:
        preds_list (list): 每张图片的预测列表，格式为
            [
                # 图片1的预测
                [[x_center, y_center, width, height, score, class_id], ...],
                # 图片2的预测
                [[x_center, y_center, width, height, score, class_id], ...],
                ...
            ]
        truths_list (list): 每张图片的真值列表，格式为
            [
                # 图片1的真值
                [[x_center, y_center, width, height, class_id], ...],
                # 图片2的真值
                [[x_center, y_center, width, height, class_id], ...],
                ...
            ]
    """
    # 合并所有图片的数据并保留图片ID信息
    all_preds = []
    all_truths = []
    
    # 为每个检测框添加图片ID
    for img_id, (preds, truths) in enumerate(zip(preds_list, truths_list)):
        # Debug log for empty predictions or truths
        if not preds:
            print(f"Warning: No predictions for image {img_id}")
        if not truths:
            print(f"Warning: No ground truths for image {img_id}")
        
        # 处理预测
        for p in preds:
            # 格式：[x_center, y_center, width, height, score, class_id, image_id]
            all_preds.append(p + [img_id])
        
        # 处理真值
        for t in truths:
            # 格式：[x_center, y_center, width, height, class_id, image_id]
            all_truths.append(t + [img_id])
    
    # 按类别和图片分组处理
    class_stats = defaultdict(lambda: {'tp': [], 'fp': [], 'n_gt': 0})
    
    # 遍历每个类别
    for class_id in set(p[5] for p in all_preds):
        # 获取当前类别的预测和真值
        class_preds = [p for p in all_preds if p[5] == class_id]
        class_truths = [t for t in all_truths if t[4] == class_id]
        
        # 按图片ID分组
        preds_per_image = defaultdict(list)
        truths_per_image = defaultdict(list)
        
        for p in class_preds:
            preds_per_image[p[6]].append(p)
        for t in class_truths:
            truths_per_image[t[5]].append(t[:4])  # 去掉image_id和class_id
        
        # 初始化统计
        scores = []
        tp = []
        fp = []
        
        # 逐图片处理
        for img_id in preds_per_image:
            img_preds = sorted(preds_per_image[img_id], key=lambda x: x[4], reverse=True)
            img_truths = truths_per_image.get(img_id, [])
            
            # 初始化匹配状态
            matched = np.zeros(len(img_truths), dtype=bool)
            
            for pred in img_preds:
                best_iou = 0.0
                best_idx = -1
                pred_box = pred[:4]
                
                # 只与当前图片的真值比较
                for gt_idx, gt_box in enumerate(img_truths):
                    if not matched[gt_idx]:
                        iou = hbb_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gt_idx
                
                # 记录结果
                scores.append(pred[4])
                if best_idx != -1 and best_iou >= iou_thresholds[0]:
                    tp.append(1)
                    fp.append(0)
                    matched[best_idx] = True
                else:
                    tp.append(0)
                    fp.append(1)
        
        # 记录结果
        class_stats[class_id]['tp'] = np.array(tp)
        class_stats[class_id]['fp'] = np.array(fp)
        class_stats[class_id]['scores'] = np.array(scores)  # 新增
        class_stats[class_id]['n_gt'] = sum(len(t) for t in truths_per_image.values())
    
    # 计算最终指标
    return compute_final_metrics(class_stats, iou_thresholds)

def compute_final_metrics(class_stats, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Compute final metrics with interpolated precision-recall curves.
    """
    aps = []
    ap_per_class = {}
    ap_per_iou = {t: [] for t in iou_thresholds}
    
    for class_id, stats in class_stats.items():
        if stats['n_gt'] == 0:
            ap = 0.0
        else:
            # Sort predictions by confidence score
            sort_idx = np.argsort(-stats['scores'])
            tp = stats['tp'][sort_idx]
            fp = stats['fp'][sort_idx]
            
            # Compute cumulative true positives and false positives
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / stats['n_gt']
            prec = cum_tp / (cum_tp + cum_fp + 1e-6)
            
            # Interpolate precision-recall curve
            unique_recalls, indices = np.unique(rec, return_index=True)
            interpolated_prec = np.maximum.accumulate(prec[indices][::-1])[::-1]
            
            # Compute AP for each IoU threshold
            ap = 0.0
            for t in iou_thresholds:
                prec_t = np.interp(t, unique_recalls, interpolated_prec, right=0)
                ap += prec_t
                ap_per_iou[t].append(prec_t)
            ap /= len(iou_thresholds)
        
        ap_per_class[class_id] = ap
        aps.append(ap)
    
    mAP = np.mean(aps)
    mAP_per_iou = {t: np.mean(vals) for t, vals in ap_per_iou.items()}
    
    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "ap_per_iou": mAP_per_iou
    }











'''
def compute_final_metrics(class_stats, iou_thresholds):
    aps = []
    ap_per_class = {}
    ap_per_iou = {t: [] for t in iou_thresholds}
    
    for class_id, stats in class_stats.items():
        if stats['n_gt'] == 0:
            ap = 0.0
        else:
            # 按置信度排序
            sort_idx = np.argsort(-np.array(stats['scores']))
            tp = stats['tp'][sort_idx]
            fp = stats['fp'][sort_idx]
            
            # 计算累积指标
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / stats['n_gt']
            prec = cum_tp / (cum_tp + cum_fp + 1e-6)
            
            # 计算各IoU阈值下的AP
            ap = 0.0
            for t in iou_thresholds:
                prec_t = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
                ap += prec_t
                ap_per_iou[t].append(prec_t)
            ap /= len(iou_thresholds)
        
        ap_per_class[class_id] = ap
        aps.append(ap)
    
    mAP = np.mean(aps)
    mAP_per_iou = {t: np.mean(vals) for t, vals in ap_per_iou.items()}
    
    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "ap_per_iou": mAP_per_iou
    }

'''

# 使用示例
if __name__ == "__main__":
    # 模拟数据：3张图片的预测和真值
    preds = [
        # 图片1的预测
        [[0.5, 0.5, 0.3, 0.3, 0.9, 0], [0.2, 0.2, 0.2, 0.2, 0.8, 1]],
        # 图片2的预测
        [[0.4, 0.4, 0.3, 0.3, 0.85, 0]],
        # 图片3的预测
        []
    ]
    
    truths = [
        # 图片1的真值
        [[0.5, 0.5, 0.3, 0.3, 0], [0.7, 0.7, 0.2, 0.2, 1]],
        # 图片2的真值
        [[0.4, 0.4, 0.3, 0.3, 0]],
        # 图片3的真值
        [[0.1, 0.1, 0.2, 0.2, 1]]
    ]
    
    results = batch_evaluate_hbb(preds, truths)
    print(f"mAP: {results['mAP']:.4f}")
    print("Per class AP:", results['ap_per_class'])


# if __name__ == "__main__":
#     # 示例预测结果
#     predictions = [
#         (100, 100, 50, 30, 0.5, 0.9, 0),  # (x, y, w, h, angle, score, class)
#         (150, 150, 60, 40, 1.0, 0.8, 0),
#         (200, 200, 55, 35, 0.2, 0.7, 1),
#         (250, 250, 70, 45, 1.2, 0.95, 1),
#     ]

#     # 示例真实标注
#     ground_truths = [
#         (105, 105, 45, 25, 0.4, 0),
#         (155, 155, 55, 35, 0.9, 0),
#         (205, 205, 50, 30, 0.1, 1),
#         (260, 260, 65, 40, 1.1, 1),
#         (300, 300, 80, 50, 0.0, 2),  # 额外的GT，没有对应的预测
#     ]

#     # 评估模型
#     results = evaluate_obb_detection(predictions, ground_truths)
#     print("mAP:", results["mAP"])
#     print("AP per class:", results["ap_per_class"])
#     print("mAP per iou_thres:", results["ap_per_iou"])
