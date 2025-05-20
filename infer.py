from ultralytics import YOLO
import cv2
import os

def image_inference(model_path, image_dir, output_dir=None, conf_threshold=0.3, sampling_factor=1):
    """
    对指定目录中的所有图片进行目标检测和分割推理
    
    参数:
        model_path: 模型文件路径
        image_dir: 输入图片目录
        output_dir: 输出图片目录，默认为None(直接在原图上绘制)
        conf_threshold: 置信度阈值
        sampling_factor: 每帧采样次数，用于增强检测稳定性
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取图片文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.isfile(os.path.join(image_dir, f)) 
                  and os.path.splitext(f)[1].lower() in image_extensions]
    
    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图片 {image_path}，跳过")
                continue
            
            # 对图片进行多次推理以增加采样
            for _ in range(sampling_factor):
                results = model(img, conf=conf_threshold)
                annotated_img = results[0].plot()
                
                # 如果是最后一次采样，保存结果
                if _ == sampling_factor - 1:
                    final_img = annotated_img
            
            # 保存结果图片
            if output_dir:
                output_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_path, final_img)
                print(f"已保存结果到 {output_path}")
            else:
                # 若未指定输出目录，覆盖原图
                cv2.imwrite(image_path, final_img)
                print(f"已在原图上绘制结果 {image_path}")
                
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")

def video_inference(model_path, source_video, output_video=None, conf_threshold=0.7, sampling_factor=2):
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频文件
    cap = cv2.VideoCapture(source_video)
    
    # 获取视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果指定了输出视频，创建视频写入对象
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 跟踪上一帧的结果用于插值
    prev_results = None
    
    # 循环处理视频帧
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # 对当前帧进行多次推理以增加采样
            for _ in range(sampling_factor):
                # 对当前帧进行模型推理，设置置信度阈值
                results = model(frame, conf=conf_threshold)
                
                # 获取推理结果并渲染到帧上（保持原有颜色）
                annotated_frame = results[0].plot()
                
                # 如果是最后一次采样，保存结果用于输出
                if _ == sampling_factor - 1:
                    final_frame = annotated_frame
            
            # 如果指定了输出视频，写入当前帧
            if output_video:
                out.write(final_frame)
        else:
            # 视频播放完毕，退出循环
            break
    
    # 释放资源
    cap.release()
    if output_video:
        out.release()


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/deep-crack/pretrain/best.pt"
    source_video = "/root/autodl-tmp/deep-crack/data/test.mp4"
    output_video = "/root/autodl-tmp/deep-crack/data/infer.mp4"
    image_dir = "/root/autodl-tmp/datasets/crack-seg/valid/images"
    output_dir = "/root/autodl-tmp/deep-crack/images"
    # 执行视频推理
    # video_inference(model_path, source_video, output_video)    
    image_inference(model_path, image_dir=image_dir, output_dir=output_dir)