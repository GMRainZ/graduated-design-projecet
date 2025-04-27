import cv2
import numpy as np


def show_obb(image, predictions, ground_truths, cls_id=8):
    """
    展示 OBB 检测结果

    Args:
        image (numpy.ndarray): 原始图像
        predictions (list): 模型预测结果的列表，每个元素是一个包含 (center_x, center_y, width, height, angle, score, class_id) 的列表
        ground_truths (list): 真实标注的列表，每个元素是一个包含 (center_x, center_y, width, height, angle, class_id) 的列表
        cls_id (int, 可选): 要展示的类别的 ID。默认为 8
    """

    # 确保图像是BGR格式，这是OpenCV期望的格式
    if image.shape[2] == 3 and image.shape[2] != 3:
        raise ValueError("图像必须是BGR格式")

    # 绘制预测框
    for pred in predictions:
        if pred[6] == cls_id:
            center_x, center_y, width, height, angle, score, _ = pred
            draw_obb(image, center_x, center_y, width, height, angle, color=(0, 255, 0))  # 绿色
            # 在边界框旁边添加文本
            text = f"{score:.2f}"  # 显示两位小数的置信度
            text_x = int(center_x - width / 2)  # 文本位置可以调整
            text_y = int(center_y - height / 2 - 10)  # 文本位置可以调整
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 绘制真实框
    for gt in ground_truths:
        if gt[5] == cls_id:
            center_x, center_y, width, height, angle, _ = gt
            draw_obb(image, center_x, center_y, width, height, np.rad2deg(angle), color=(0, 0, 255))  # 红色

    # # 显示图像
    # cv2.imshow("OBB Detections", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("result.jpg", image)


def show_obb_with_poly(image, predictions, ground_truths, cls_id=None):
    """
    展示 OBB 检测结果

    Args:
        image (numpy.ndarray): 原始图像
        predictions (list): 模型预测结果的列表，每个元素是一个包含 (x1, y1, x2, y2, x3, y3, x4, y4, score, class_id) 的列表
        ground_truths (list): 真实标注的列表，每个元素是一个包含 (x1, y1, x2, y2, x3, y3, x4, y4, score, class_id) 的列表
        cls_id (int, 可选): 要展示的类别的 ID。默认为 8
    """

    # 确保图像是BGR格式，这是OpenCV期望的格式
    if image.shape[2] == 3 and image.shape[2] != 3:
        raise ValueError("图像必须是BGR格式")

    # 绘制预测框
    for pred in predictions:
        if cls_id is None or pred[-1] == cls_id:
            for i in range(4):
                pt1 = np.array((pred[i], pred[i + 1])).astype(np.int32)
                pt2 = np.array((pred[(i + 1) % 4], pred[(i + 1) % 4 + 1])).astype(np.int32)
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)
            # cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 绘制真实框
    for gt in ground_truths:
        if cls_id is None or gt[-1] == cls_id:
            for i in range(4):
                pt1 = np.array((gt[i], gt[i + 1])).astype(np.int32)
                pt2 = np.array((gt[(i + 1) % 4], gt[(i + 1) % 4 + 1])).astype(np.int32)
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)

    # # 显示图像
    # cv2.imshow("OBB Detections", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("result.jpg", image)


def draw_obb(image, center_x, center_y, width, height, angle, color):
    """
    在图像上绘制一个定向边界框 (OBB)。

    Args:
        image (numpy.ndarray): 要在其上绘制 OBB 的图像。
        center_x (float): OBB 中心的 X 坐标。
        center_y (float): OBB 中心的 Y 坐标。
        width (float): OBB 的宽度。
        height (float): OBB 的高度。
        angle (float): OBB 的旋转角度，以度为单位。
        color (tuple): 用于绘制 OBB 的颜色 (B, G, R)。
    """
    # 将角度从度转换为弧度
    angle_rad = np.radians(angle)

    # 计算 OBB 的角
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)

    # OBB的四个顶点相对于中心点的偏移
    points = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]

    # 旋转并平移顶点
    rotated_points = [(int(center_x + cos_val * x - sin_val * y), int(center_y + sin_val * x + cos_val * y)) for x, y in points]

    # 绘制 OBB 的边
    for i in range(4):
        cv2.line(image, rotated_points[i], rotated_points[(i + 1) % 4], color, 2)
