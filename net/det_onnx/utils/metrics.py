# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import torch


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


# def process_batch(detections, labels, iouv):
#     """
#     Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
#     Arguments:
#         detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#         labels (Array[M, 5]), class, x1, y1, x2, y2
#     Returns:
#         correct (Array[N, 10]), for 10 IoU levels
#     """
#     correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
#     iou = box_iou(labels[:, 1:], detections[:, :4])
#     x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
#     if x[0].shape[0]:
#         matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
#         if x[0].shape[0] > 1:
#             matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#             # matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         matches = torch.Tensor(matches).to(iouv.device)
#         correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
#     return correct

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (np.ndarray[N, 6]), x1, y1, x2, y2, conf, class
        labels (np.ndarray[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (np.ndarray[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    
    # Compute IoU between labels and detections
    iou = box_iou_numpy(labels[:, 1:], detections[:, :4])
    
    # Find matches where IoU >= threshold and classes match
    x = np.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # 注意维度对齐
    
    if x[0].size > 0:
        # 合并匹配项并添加IoU值 [label, detection, iou]
        matches = np.column_stack((x[0], x[1], iou[x[0], x[1]]))
        
        # 处理多个匹配的情况
        if x[0].size > 1:
            # 按IoU降序排序
            matches = matches[matches[:, 2].argsort()[::-1]]
            
            # 保留每个检测框的最佳匹配
            _, unique_indices = np.unique(matches[:, 1], return_index=True)
            matches = matches[unique_indices]
            
            # 保留每个标签的最佳匹配
            _, unique_indices = np.unique(matches[:, 0], return_index=True)
            matches = matches[unique_indices]
        
        # 更新correct矩阵
        detection_indices = matches[:, 1].astype(int)
        iou_values = matches[:, 2]
        correct[detection_indices] = iou_values[:, None] >= iouv
    
    return correct
def box_iou_numpy(box1, box2):
    """
    NumPy版本的IoU计算，输入格式为(x1, y1, x2, y2)
    """
    # 计算交集区域
    inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])
    
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    
    # 计算并集区域
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, None] + area2 - inter_area
    
    return inter_area / union_area






def convert_poly_to_hbb(label_filename, name_to_id):
    """
    从标签文件读取多边形标注，转换为水平边界框格式
    :param label_filename: 标签文件路径，每行格式为 "x1 y1 x2 y2 x3 y3 x4 y4 class_name class_id"
    :param name_to_id: 类别名称到ID的映射字典（自动更新新类别）
    :return:
        labels_hbbn: 水平框数组，形状 (n, 5)，格式 [class_id, x1, y1, x2, y2]
        name_to_id: 更新后的类别映射表
    """
    labels_hbbn = []
    
    try:
        # 读取标签文件
        with open(label_filename, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"标签文件 {label_filename} 不存在")
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {e}")

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue  # 跳过空行

        parts = line.split()
        if len(parts) != 10:
            print(f"警告: 跳过格式错误的行: {line}")
            continue

        try:
            # 解析坐标、类别名称和ID
            coords = list(map(int, parts[:8]))  # 前8个字段为坐标
            class_name = parts[8]               # 类别名称
            class_id = int(parts[9])            # 类别ID
        except ValueError as e:
            print(f"警告: 解析数值失败 ({line}): {e}")
            continue

        # 检查类别名称与ID是否匹配
        if class_name in name_to_id:
            if name_to_id[class_name] != class_id:
                print(f"警告: 类别 {class_name} 的ID冲突: 映射表中为 {name_to_id[class_name]}，输入中为 {class_id}")
                class_id = name_to_id[class_name]  # 优先使用映射表的ID
        else:
            name_to_id[class_name] = class_id

        # 计算水平框
        x_coords = coords[0::2]  # 所有x坐标
        y_coords = coords[1::2]  # 所有y坐标
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)

        labels_hbbn.append([class_id, x1, y1, x2, y2])

    return np.array(labels_hbbn)

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        # x = torch.where(iou > self.iou_thres)
        # if x[0].shape[0]:
        #     matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        #     if x[0].shape[0] > 1:
        #         matches = matches[matches[:, 2].argsort()[::-1]]
        #         matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        #         matches = matches[matches[:, 2].argsort()[::-1]]
        #         matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        # else:
        #     matches = np.zeros((0, 3))


        # 转换为NumPy的代码
        x = np.where(iou > self.iou_thres)  # 获取满足条件的坐标
        if x[0].size > 0:  # 等价于原版 x[0].shape[0]
            # 组合匹配矩阵 (shape: [n_matches, 3])
            matches = np.column_stack((
                x[0],          # 标签索引
                x[1],          # 检测框索引
                iou[x[0], x[1]]  # 对应的IoU值
            ))
            
            if matches.shape[0] > 1:  # 当存在多个匹配时
                # 第一次降序排序 (按IoU)
                matches = matches[matches[:, 2].argsort()[::-1]]  # 使用[::-1]实现倒序
                
                # 按检测框索引去重 (保留每个检测框的最佳匹配)
                _, unique_idx = np.unique(matches[:, 1], return_index=True)
                matches = matches[unique_idx]
                
                # 再次降序排序 (保证标签去重前保持最高IoU)
                matches = matches[matches[:, 2].argsort()[::-1]]
                
                # 按标签索引去重 (保留每个标签的最佳匹配)
                _, unique_idx = np.unique(matches[:, 0], return_index=True)
                matches = matches[unique_idx]
        else:
            matches = np.zeros((0, 3))  # 空矩阵保持形状一致性

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # # Intersection area
    # inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
    #         (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)


    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(min=0) * \
        (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(min=0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                v = (4 / math.pi ** 2) * ((np.arctan(w2 / h2) - np.arctan(w1 / h1)**2), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:
                return iou - rho2 / c2  # DIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f hbb mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
