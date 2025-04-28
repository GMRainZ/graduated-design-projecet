from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import ffmpeg

from det_onnx import det_onnx_hbb
from det_onnx.detectionapi import DetectionAPI

app = Flask(__name__)
app.config.from_pyfile('config.py')

# 在 app = Flask(__name__) 后添加
print("[DEBUG] 当前配置:", app.config.keys())
print("[DEBUG] UPLOAD_FOLDER 是否存在:", 'UPLOAD_FOLDER' in app.config)

# 生成安全文件名
def generate_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name, ext = os.path.splitext(secure_filename(filename))
    return f"{name}_{timestamp}{ext}"

# 获取视频时长
def get_video_duration(filepath):
    try:
        probe = ffmpeg.probe(filepath)
        return float(probe['format']['duration'])
    except:
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify(success=False, error="未选择文件")
    
    video = request.files['video']
    if video.filename == '':
        return jsonify(success=False, error="空文件名")

    # 验证文件扩展名
    file_ext = video.filename.rsplit('.', 1)[1].lower() if '.' in video.filename else ''
    if file_ext not in app.config['ALLOWED_EXTENSIONS']:
        return jsonify(success=False, error=f"不支持的文件类型: {file_ext}")


    try:
        # 生成防重名文件名
        filename = generate_filename(video.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 分块写入（避免内存溢出）
        chunk_size = 4096
        with open(save_path, 'wb') as f:
            while True:
                chunk = video.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        
        # 验证文件完整性
        if not os.path.exists(save_path):
            raise Exception("文件保存失败")
        
        # 获取元数据
        duration = get_video_duration(save_path)
        return jsonify(
            success=True,
            filename=filename,
            duration=duration,
            filesize=os.path.getsize(save_path),
            url=f"/static/uploads/{filename}"
        )
    except Exception as e:
        # 清理不完整文件
        if 'save_path' in locals() and os.path.exists(save_path):
            os.remove(save_path)
        return jsonify(success=False, error=f"上传失败: {str(e)}")






from threading import Lock
import uuid
import time

# 任务状态存储
task_status = {}
task_lock = Lock()

@app.route('/process', methods=['POST'])
def start_processing():
    data = request.get_json()
    filename = data['filename']
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 启动后台任务
    start_background_task(task_id, filename)
    
    return jsonify({'task_id': task_id}), 202

def start_background_task(task_id, filename):
    # 初始化检测器
    detector = DetectionAPI(
    model_weights="det_onnx/weights/v11_coco128_attention_50epoch.onnx",
    label_mapping_path="det_onnx/docs/coco128.yaml"
    )
    def processing_task():
        with task_lock:
            task_status[task_id] = {
                'state': 'PROGRESS',
                'progress': 0,
                'result': None
            }
        
        try:
            # 处理视频
            result_filename = 'static/downloads/result.mp4'
            detector.process_video(
                input_path="static/uploads/385900522-1-208_20250421191945.mp4",
                output_path=result_filename,
                conf=0.6,  # 覆盖默认阈值
                iou=0.4
            )
            
            with task_lock:
                task_status[task_id].update({
                    'state': 'SUCCESS',
                    'result': result_filename
                })
        except Exception as e:
            with task_lock:
                task_status[task_id].update({
                    'state': 'FAILURE',
                    'error': str(e)
                })

    # 启动线程
    import threading
    thread = threading.Thread(target=processing_task)
    thread.start()

@app.route('/status/<task_id>')
def get_task_status(task_id):
    with task_lock:
        status = task_status.get(task_id, {'state': 'UNKNOWN'})
    
    if status['state'] == 'SUCCESS':
        return jsonify({
            'state': status['state'],
            'result_url': f"/static/uploads/{status['result']}"
        })
    elif status['state'] == 'FAILURE':
        return jsonify({
            'state': status['state'],
            'error': status.get('error', 'Unknown error')
        })
    else:
        return jsonify({
            'state': status['state'],
            'progress': status.get('progress', 0)
        })








if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000)