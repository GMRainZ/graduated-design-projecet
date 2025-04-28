import time
import numpy as np
import sys
import math
import cv2


def obb_nms_iou(boxes, scores, iou_threshold, output_max_num=10):
    """
    obb nms iou

    Args:
        boxes (ndarray): 推理得到的检测框的坐标信息, dim(n, 5)  x,y,x,y,theta
        scores (ndarray): 推理得到的检测框的得分信息, dim(n, )
        iou_threshold (float): iou 阈值
        output_max_num (int): optional, 最大输出框数量
    Return:
        keep_idx (list): 保留的索引
    """
    keep_idx = []  # 保留的索引
    order = scores.argsort()[::-1]  # 按分数高低排序的索引
    num = len(boxes)  # 备选框的个数
    suppressed = np.zeros((num), dtype=np.int32)  # 与确信检测框交并比过大的框的索引

    # 遍历所有检测框
    for _i in range(num):
        # 满了就跳出
        if len(keep_idx) >= output_max_num:
            break

        # 当前分数最高的索引
        i = order[_i]
        # 对于抑制的检测框直接跳过
        if suppressed[i] == 1:
            continue

        keep_idx.append(i)  # 保留当前框的索引
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])  # 左上，右下，角度
        area_r1 = boxes[i, 2] * boxes[i, 3]
        # 遍历剩下的框与当前框的交并比
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            iou = 0.0
            inter_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]  # 求两个旋转矩形的交集
            if inter_pts is not None:
                contours = cv2.convexHull(inter_pts, returnPoints=True)
                inter_area = cv2.contourArea(contours)
                iou = inter_area * 1.0 / (area_r1 + area_r2 - inter_area + 0.0000001)
            # 超过阈值的记为抑制
            if iou >= iou_threshold:
                suppressed[j] = 1

    return keep_idx


def non_max_suppression_obb(prediction, conf_thres, iou_thres, max_det, classes=None, agnostic=False, multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results_obb
    Args:
        prediction (tensor): (b, n_all_anchors, [cx cy l s obj num_cls theta_cls])
        agnostic (bool): True = NMS will be applied between elements of different categories
        labels : () or

    Returns:
        list of detections, len=batch_size, on (n,7) tensor per image [xylsθ, conf, cls] θ ∈ [-pi/2, pi/2)
    """
    nc = prediction.shape[2] - 5 - 180  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    class_index = nc + 5

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    max_wh = 4096  # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [np.zeros((0, 7))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence, (tensor): (n_conf_thres, [cx cy l s obj num_cls theta_cls])

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf6
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        theta_pred = np.max(x[:, class_index:], 1, keepdims=True)  # [n_conf_thres, 1] θ ∈ int[0, 179]
        theta_pred = np.argmax(x[:, class_index:], axis=1)
        theta_pred = theta_pred.reshape(-1, 1)
        theta_pred = (theta_pred - 90) / 180 * np.pi  # [n_conf_thres, 1] θ ∈ [-pi/2, pi/2)

        # Detections matrix nx7 (xyls, θ, conf, cls) θ ∈ [-pi/2, pi/2)
        if multi_label:
            # i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T # ()
            # 获取非零索引
            i, j = np.nonzero(x[:, 5:class_index] > conf_thres)

            # 如果需要转置以匹配之前的形状
            i, j = i.T, j.T
            x = np.concatenate((x[i, :4], theta_pred[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdims=True)
            x = np.concatenate((x[:, :4], theta_pred, conf, j.astype(np.float32)), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 6:7] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            conf_indices = x[:, 5].argsort()[-max_nms:][::-1]
            x = x[conf_indices]

        # Batched NMS
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        rboxes = x[:, :5].copy()
        rboxes[:, :2] = rboxes[:, :2] + c  # rboxes (offset by class)
        scores = x[:, 5]  # scores
        i = np.array(obb_nms_iou(rboxes, scores, iou_thres, output_max_num=max_det))
        # print(f'=========================== nms i = {i}')
        output[xi] = x[i, :7]

    return output


def rbox2poly_np(obboxes):
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.concatenate([w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, -h / 2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    order = obboxes.shape[:-1]
    return np.concatenate([point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # h_ratios
        pad = ratio_pad[1]  # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain  # Rescale poly shape to img0_shape
    # clip_polys(polys, img0_shape)
    return polys


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


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints
    Returns:
        im (array): (height, width, 3)
        ratio (array): [w_ratio, h_ratio]
        (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [h_rect, w_rect]
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # wh ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # [w h]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # [w_ratio, h_ratio]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxyxyxy2xywh(x):
    """
    将四边形顶点坐标转换为轴对齐外接矩形的中心点坐标和宽高
    Args:
        x (np.ndarray): 输入为nx8数组，每行格式为[x1,y1,x2,y2,x3,y3,x4,y4]

    Returns:
        np.ndarray: 输出为nx4数组，格式为[x_center, y_center, width, height]
    """
    # 提取所有x和y坐标（向量化操作）
    x_coords = x[:, [0, 2, 4, 6]]  # 所有x坐标
    y_coords = x[:, [1, 3, 5, 7]]  # 所有y坐标

    # 计算极值（无需循环）
    x_min = np.min(x_coords, axis=1)
    y_min = np.min(y_coords, axis=1)
    x_max = np.max(x_coords, axis=1)
    y_max = np.max(y_coords, axis=1)

    # 构造输出数组
    y = np.empty((x.shape[0], 4))
    y[:, 0] = (x_min + x_max) / 2  # 中心x
    y[:, 1] = (y_min + y_max) / 2  # 中心y
    y[:, 2] = x_max - x_min  # 宽度
    y[:, 3] = y_max - y_min  # 高度

    return y


def resize_obb(obb, original_width, original_height, resized_width=512, resized_height=512):
    """
    将 OBB 结果 resize 转换到 original

    Args:
    obb (tuple): OBB  (center_x, center_y, width, height, angle)
    original_width (int): 原始图像的宽度
    original_height (int): 原始图像的高度
    resized_width (int): 调整大小后的图像宽度
    resized_height (int): 调整大小后的图像高度

    返回:
    tuple: 转换后的 OBB 参数 (center_x, center_y, width, height, angle)
    """
    center_x, center_y, width, height, angle = obb[0:5]

    width_k = original_width / resized_width
    height_k = original_height / resized_height

    original_center_x = center_x * width_k
    original_center_y = center_y * height_k
    original_width = width * width_k
    original_height = height * height_k

    return (original_center_x, original_center_y, original_width, original_height, angle, obb[5], obb[6])


def resize_obb_baseon_poly(obb, original_width, original_height, resized_width=512, resized_height=512):
    """
    将 OBB 结果 resize 转换到 original
    """
    print(f"==== obb: {obb}")
    # obb to ploy
    center_x, center_y, w, h, theta = obb[0:5]
    center = np.array([center_x, center_y])
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.asarray([w / 2 * Cos, -w / 2 * Sin])
    vector2 = np.asarray([-h / 2 * Sin, -h / 2 * Cos])
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    poly = np.asarray([point1, point2, point3, point4])  # dim(4, 2)
    print(f"==== poly: {poly}")

    # resize
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain  # Rescale poly shape to img0_shape
    print(f"==== poly resize: {poly}")

    # poly to obb
    reszied_obb = obb_corners_to_center(poly.astype(np.int32))
    print(f"==== reszied_obb: {reszied_obb}")

    return reszied_obb


def convert_rotated_rect(quad):
    """
    处理旋转矩形（需要安装OpenCV）
    输入格式：x1,y1,x2,y2,x3,y3,x4,y4,score,tcls
    """
    # 将坐标转换为numpy数组
    points = np.array(quad[:8]).reshape(4, 2).astype(np.float32)
    (cx, cy), (w, h), angle = obb_corners_to_center(points)
    # 计算最小外接矩形
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect

    # 保留最后两个值
    return [cx, cy, w, h, quad[-2], quad[-1]]


def convert_to_rotated_rect(line: str, name_to_id: dict, shape: tuple) -> tuple:
    """
    将原始数据行转换为旋转矩形参数 (center_x, center_y, w, h, angle, class_id)

    参数:
    - line: 输入数据行，例如 "2753 2408 2861 2385 2888 2468 2805 2502 plane 0"
    - name_to_id: 类别名称到ID的映射字典，例如 {"plane": 0, "car": 1}

    返回:
    - (center_x, center_y, w, h, angle, class_id)
    """
    parts = line.strip().split()
    if len(parts) < 10:
        raise ValueError("Invalid data format")

    # 解析四个点的坐标 (x1, y1, x2, y2, x3, y3, x4, y4)
    points = np.array([[int(parts[0]), int(parts[1])], [int(parts[2]), int(parts[3])], [int(parts[4]), int(parts[5])], [int(parts[6]), int(parts[7])]], dtype=np.float32)

    # points = np.array(quad[:8]).reshape(4, 2).astype(np.float32)
    center_x, center_y, w, h, angle = obb_corners_to_center(points)

    # 计算最小包围旋转矩形
    # rect = cv2.minAreaRect(points)
    # (center_x, center_y), (w, h), angle = rect

    # if w < h:
    #     w, h = h, w
    #     angle += 90  # 补偿宽高交换带来的角度偏移
    # # else:
    # #     l, s = w, h

    # # 角度规范化（OpenCV角度转YOLO-OBB格式）
    # angle_rad = np.deg2rad(angle % 90 - 90) if angle > 90 else np.deg2rad(angle)

    # 类别名称转ID
    class_name = parts[8]
    if class_name not in name_to_id:
        raise KeyError(f"Class name '{class_name}' not found in name_to_id mapping")
    class_id = name_to_id[class_name]

    return (center_x, center_y, w, h, np.deg2rad(angle), class_id)


def obb_corners_to_center(corners):
    """
    将 OBB 的四个顶点坐标转换为中心点、宽度、高度和角度的格式。

    Args:
    corners (np.ndarray): OBB 的四个顶点坐标 dim(4,2)

    Returns:
    tuple: (center_x, center_y, width, height, angle)
    """
    center_x = np.mean(corners[:, 0])
    center_y = np.mean(corners[:, 1])

    rect = cv2.minAreaRect(corners)  # return (center_x, center_y), (width, height), angle
    width, height = rect[1]
    # keep angle in [-90, 90]
    if width < height:
        width, height = height, width
        angle = rect[2] + 90
    else:
        angle = rect[2]

    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    return center_x, center_y, width, height, angle


def process_file(file_path: str, name_to_id: dict, shape: tuple) -> list:
    results = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                result = convert_to_rotated_rect(line, name_to_id, shape)
                # result = resize_obb(result,shape[0],shape[1])
                results.append(result)
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid line: {line.strip()} | Error: {str(e)}")
    return results
