import cv2

# 初始化视频源


input_path = r"/home/grainz/demo_show1.mp4"

# # 使用 ffmpeg-python 包强制转换视频格式为 MP4
# import ffmpeg
# temp_output_path = "output/videoes/temp/temp_output.mp4"
# try:
#     ffmpeg.input(input_path).output(temp_output_path, c='libx264', acodec='aac', strict='experimental').run()
# except ffmpeg.Error as e:
#     print(f"FFmpeg error: {e.stderr}")
#     raise ValueError(f"无法转换视频格式: {input_path}")
    


cap = cv2.VideoCapture(input_path)  # 0 表示默认摄像头，文件路径则替换为字符串（如 "video.mp4"）

while cap.isOpened():
    ret, frame = cap.read()  # 读取一帧
    
    if not ret:
        print("无法读取帧！")
        break
    
    # 可在此处添加对 frame 的处理（如滤镜、检测等）
    
    cv2.imshow('Video', frame)  # 显示帧
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # 将帧率转换为 milliseconds 的延迟
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()