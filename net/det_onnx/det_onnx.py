import sys
import cv2
import numpy as np
import os
import time
from functools import wraps

np.set_printoptions(precision=3, suppress=True)
import onnxruntime as ort

import val_with_np_with_poly
import draw_with_poly

sys.path.append(".")
from utils.general import non_max_suppression_obb, process_file, resize_obb, resize_obb_baseon_poly
from utils.general import letterbox, rbox2poly_np, scale_polys, poly_label,gt_to_poly
from utils.metrics import *



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



class ONNX_Pred:
    def __init__(self, cfg):
        """
        Initializes an instance of the detector class.

        Args:
            onnx_model: Path to the ONNX model.
            classes: Dict of class names.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.name_list = [
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

        self.name_to_id = {name: idx for idx, name in enumerate(self.name_list)}

        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        # self.args.task = "detect"
        # self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = np.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        # self.niou = self.iouv.numel()
        # self.lb = []  # for autolabelling
        self.stats = []
        # self.name_to_id={}

        try:
            onnx_model = cfg.get("model_file")
            self.conf_thres = cfg.get("conf_thres")
            self.iou_thres = cfg.get("iou_thres")
            self.max_det = cfg.get("max_det")
            device = "cuda" if cfg.get("device") == "cuda" else "cpu"

            self.img_w = cfg.get("image_w")
            self.img_h = cfg.get("image_h")
            
            self.data_len = self.img_w * self.img_h * 3
            exit_flag = False

            self.label_file = "input/labels/" + cfg.get("img_file").split("/")[-1].replace(".jpg", ".txt")
        except:
            exit_flag = True
        finally:
            print(f"Configuration:")
            print(f"  model_file: {onnx_model}")
            print(f"  conf_thres: {self.conf_thres}")
            print(f"  iou_thres: {self.iou_thres}")
            print(f"  max_det: {self.max_det}")
            print(f"  image_w: {self.img_w}")
            print(f"  image_h: {self.img_h}")
            if exit_flag:
                exit()

        # Load the class names
        self.classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}

        # Set the device
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Load model
        try:
            self.session = ort.InferenceSession(
                onnx_model,
                providers=providers,
            )
        except Exception as e:
            print(f"Create detector failed!: {e}")
            exit()

        self.input_width = self.img_w
        self.input_height = self.img_h

        self.model_inputs = self.session.get_inputs()
    # def reset_
    def preprocess(self, input_img):
        """
        Preprocesses the input image for model inference.

        Args:
            input_img (numpy.ndarray): The input image read by OpenCV.

        Returns:
            numpy.ndarray: The preprocessed image data.

        Steps:
            1. Extracts the height and width of the input image.
            2. Converts the input image from BGR to RGB color space.
            3. Resizes the image to the specified input width and height.
            4. Normalizes the pixel values of the image to the range [0, 1].
            5. Transposes the image data to have the channel as the first dimension.
            6. Expands the image data to a batch of size 1 and converts it to float32 format.
        """
        self.img_height, self.img_width = input_img.shape[:2]
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        self.wh_scale = (self.img_width / self.input_width, self.img_height / self.input_height)
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    @average_execution_time
    def pred(self, input_img):
        # Image pred
        img_data = letterbox(input_img, self.img_w, stride=32, auto=False)[0]
        img_data = img_data.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_data = np.array([img_data])
        img_data = np.ascontiguousarray(img_data)
        img_data = img_data.astype(np.float32) / 255.0
        result_raw = self.session.run(None, {self.model_inputs[0].name: img_data})[0]

        result_objs = non_max_suppression_obb(result_raw, self.conf_thres, self.iou_thres, self.max_det, multi_label=True)  # list of [x, y, w, h, score, class_id]
        result_objs = np.array(result_objs)
        ## 1,n,7  中心点 (100, 200), 宽 50, 高 30, 角度 30°, 置信度 0.95, 类别 0

        # pred_hbbn (Array[N, 6]), x1, y1, x2, y2, conf, class
        # labels_hbbn (Array[M, 5]), class, x1, y1, x2, y2
        ##pred_poly[:, 8]：提取所有预测框的置信度（conf）。
        ##pred_poly[:, 9]：提取所有预测框的类别 ID（cls）。
        ### stats 的典型组成（每个批次处理后追加到列表中）
        # batch_stats = [
        #     pred_is_tp,        # 是否是真阳性（TP, True Positive，布尔数组，shape: [N]）
        #     pred_confidences,  # 预测框的置信度（shape: [N]）
        #     pred_cls,          # 预测框的类别（shape: [N]）
        #     target_cls         # 对应真实框的类别（shape: [M]）
        # ]

        # pred_hbbn=self.get_pred_hbbn(result_objs)

        # labels_hbbn=self.get_labels_hbbn()
        # correct = process_batch(pred_hbbn, labels_hbbn, self.iouv)

        # self.stats.append((correct, result_objs[:, 4], result_objs[:, 5], labels_hbbn[:,0]))

        # print(f"result_objs = \n{result_objs}")
        return result_objs

    def get_pred_hbbn(self, result_objs):
        """
        将旋转框检测结果转换为水平框 (xyxy 格式)
        :param result_objs: 输入数组，形状 (1, n, 7)，格式 [cx, cy, w, h, theta_rad, score, class_id]
        :return: 输出数组，形状 (n, 6)，格式 [x1, y1, x2, y2, score, class_id]
        """
        # 去除批次维度，形状变为 (n, 7)
        # result_objs = np.array(result_objs)
        result_objs = result_objs.squeeze(0)

        # 提取参数
        cx = result_objs[:, 0]  # 中心点 x 坐标
        cy = result_objs[:, 1]  # 中心点 y 坐标
        w = result_objs[:, 2]  # 宽度
        h = result_objs[:, 3]  # 高度
        theta_rad = result_objs[:, 4]  # 旋转角度（弧度）
        scores = result_objs[:, 5]  # 置信度
        class_ids = result_objs[:, 6]  # 类别 ID

        # 计算旋转后的顶点坐标（向量化操作）
        cos_theta = np.cos(theta_rad)[:, np.newaxis]  # 形状 (n, 1)
        sin_theta = np.sin(theta_rad)[:, np.newaxis]

        # 计算顶点的相对偏移
        dx = np.array([-0.5, 0.5, 0.5, -0.5]) * w[:, np.newaxis]  # 形状 (n, 4)
        dy = np.array([-0.5, -0.5, 0.5, 0.5]) * h[:, np.newaxis]

        # 计算旋转后的坐标增量
        x_offset = dx * cos_theta - dy * sin_theta  # 形状 (n, 4)
        y_offset = dx * sin_theta + dy * cos_theta

        # 计算绝对坐标
        x_rot = cx[:, np.newaxis] + x_offset  # 形状 (n, 4)
        y_rot = cy[:, np.newaxis] + y_offset

        # 计算外接水平矩形
        x1 = np.min(x_rot, axis=1)
        y1 = np.min(y_rot, axis=1)
        x2 = np.max(x_rot, axis=1)
        y2 = np.max(y_rot, axis=1)

        # 合并结果
        pred_hbbn = np.column_stack([x1, y1, x2, y2, scores, class_ids])
        return pred_hbbn

    def get_labels_hbbn(self):
        print(self.label_file, self.name_to_id)
        return convert_poly_to_hbb(self.label_file, self.name_to_id)

    def draw_results(self, img, objs, resize_shape, raw_shape):
        pred_poly = rbox2poly_np(objs[:, :5])
        pred_poly = scale_polys(resize_shape, pred_poly, raw_shape)
        poly = np.hstack([pred_poly, objs[:, 5:]])  # (n, [poly conf cls])  dim(n, 10)
        label = str(objs[-1])  # TODO use classname instead of class_id
        poly_label(img, poly, label)

    def compute_accuracy(self, result_objs):
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(self.stats) and self.stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*self.stats, plot=False)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=nc)  # number of targets per class

            # Print results
            pf = "%20s" + "%11i" + "%11.3g" * 4  # print format
            # LOGGER.info("----------------------------------------------")
            print(pf % ("all", nt.sum(), mp, mr, map50, map))
            for i, c in enumerate(ap_class):
                print.info(pf % (self.name_to_id[c], nt[c], p[i], r[i], ap50[i], ap[i]))
            # LOGGER.info("----------------------------------------------")
        else:
            nt = np.zeros(1)

    def print_result():
        pass

    def get_average_time(self):
        return self.pred.get_average()


def predictions_to_poly(objs, resize_shape, raw_shape):
    pred_poly = rbox2poly_np(objs[:, :5])  # cx, cy, w, h, theta, score, cls
    pred_poly = scale_polys(resize_shape, pred_poly, raw_shape)
    poly = np.hstack([pred_poly, objs[:, 6:]])  # (n, [*poly, cls])  dim(n, 9)
    return poly





if __name__ == "__main__":

    if False:
        DRAW_RESULTS = True

        cfg = {"model_file": r"weights/yolov5s_obb_ori_512_fp32.onnx", "conf_thres": 0.25, "iou_thres": 0.45, "image_w": 512, "image_h": 512, "max_det": 10, "img_file": r"input/images/P0000.jpg"}
        det = ONNX_Pred(cfg)  # Init detector
        img = cv2.imread(r"input/images/P0000.jpg")
        result_objs = det.pred(img)

        predictions = result_objs[0]
        print(f"predictions[0]: {predictions[0]}")  # [cx, cy, w, h, theta, score, class_id]
        pred_poly = predictions_to_poly(predictions, (cfg["image_w"], cfg["image_h"]), img.shape)
        print(f"pred_poly.shape: {pred_poly.shape}")
        print(f"pred_poly: {pred_poly[0]}")

        height, width = img.shape[:2]
        ground_truths = np.array(process_file(r"input/labels/P0000.txt", det.name_to_id, (width, height)))
        gt_poly = gt_to_poly(ground_truths)
        print(f"gt_poly.shape: {gt_poly.shape}")
        print(f"gt_poly: {gt_poly[0]}")

        np.save(r"output/predictions.npy", predictions, allow_pickle=False)
        np.save(r"output/ground_truths.npy", ground_truths, allow_pickle=False)

        # 绘制结果图
        show_img = np.zeros((5000, 5000, 3), dtype=np.uint8)
        poly_label(show_img, gt_poly, color=(0, 255, 0), cls=8)
        poly_label(show_img, pred_poly, color=(0, 0, 255), cls=8)
        cv2.imwrite("result.jpg", show_img)

        # 评估模型
        pred_poly = np.concatenate([pred_poly[:, :8], predictions[:, 5:6], pred_poly[:, -1:]], axis=1)
        pred_poly = pred_poly[pred_poly[:, -1]==8]
        gt_poly = gt_poly[gt_poly[:, -1]==8]
        results = val_with_np_with_poly.evaluate_obb_detection_with_poly(pred_poly, gt_poly)
        print("mAP:", results["mAP"])
        print("AP per class:", results["ap_per_class"])
        print("mAP per iou_thres:", results["ap_per_iou"])

        det.draw_results(img, result_objs[0], (cfg["image_w"], cfg["image_h"]), img.shape)
        cv2.imwrite(r"output/1.jpg", img)

    else:
        cfg = {"model_file": r"weights/yolov5s_obb_ori_512_fp32.onnx", "conf_thres": 0.25, "iou_thres": 0.45, "image_w": 512, "image_h": 512, "max_det": 10, "img_file": r"input/images/P0000.jpg"}
        det = ONNX_Pred(cfg)  # Init detector
        
        # 获取所有待处理图片路径
        img_dir = r"input/selected_images_and_labels/images"
        label_dir = r"input/selected_images_and_labels/labels"
        # img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))  # 支持.jpg/.png等
        
        # 创建输出目录
        os.makedirs("output/results", exist_ok=True)
        os.makedirs("output/viz", exist_ok=True)

        # 结果汇总容器
        all_preds = []
        all_gts = []
        metrics = []

        for img_name in os.listdir(img_dir):
            try:
                # 生成对应标签路径
                img_path=os.path.join(img_dir,img_name)
                label_path = os.path.join(label_dir,img_name.replace(".jpg",".txt"))
                
                # 加载图像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图像：{img_path}")
                    continue
                    
                # 模型推理
                result_objs = det.pred(img)
                predictions = result_objs[0]
                
                # 处理标签
                height, width = img.shape[:2]
                ground_truths = process_file(label_path, det.name_to_id, (width, height))
                
                # 转换为多边形格式
                pred_poly = predictions_to_poly(predictions, (cfg["image_w"], cfg["image_h"]), img.shape)
                gt_poly = gt_to_poly(ground_truths)
                
                # 保存原始数据
                base_name=img_name.split(".")[0]
                np.save(f"output/results/{base_name}_pred.npy", predictions)
                np.save(f"output/results/{base_name}_gt.npy", ground_truths)
                
                # 可视化结果
                viz_img = img.copy()
                det.draw_results(viz_img, predictions, (cfg["image_w"], cfg["image_h"]), img.shape)
                cv2.imwrite(f"output/viz/{base_name}_viz.jpg", viz_img)
                
                # 评估指标计算
                if len(pred_poly) > 0 and len(gt_poly) > 0:
                    pred_eval = np.concatenate([pred_poly[:, :8], predictions[:,5:6], pred_poly[:,-1:]], axis=1)
                    pred_eval = pred_eval[pred_eval[:,-1]==8]
                    gt_eval = gt_poly[gt_poly[:,-1]==8]
                    
                    if len(pred_eval) > 0:
                        results = val_with_np_with_poly.evaluate_obb_detection_with_poly(pred_eval, gt_eval)
                        metrics.append(results)
                        
            except Exception as e:
                print(f"处理文件 {img_path} 时发生错误：{str(e)}")
                continue
        
        # 获取平均执行时间
        avg_time = det.get_average_time()
        print(f"Average inference time: {avg_time:.4f} seconds")
        # 汇总评估指标
        if len(metrics) > 0:
            final_mAP = np.mean([m["mAP"] for m in metrics])
            print(f"全局 mAP: {final_mAP:.4f}")
            np.save("output/metrics.npy", metrics)
