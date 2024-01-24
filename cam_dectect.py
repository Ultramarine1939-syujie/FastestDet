import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector

device = torch.device("cpu")

yaml_path = "configs/coco.yaml"
weight_path = "weights/weight_AP05:0.253207_280-epoch.pth"
video_path = "video/video.avi"
thresh = 0.75    #识别阈值
res = 0

# 解析yaml配置文件
cfg = LoadYaml(yaml_path)    
print(cfg) 

def draw_cap(output, ori_frame):
    global res
    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
    H, W, _ = ori_frame.shape
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width
    
    # 绘制预测框
    for box in output[0]:
        res = res + 1
        if res > 100:
            print(box)
            res = 0
        box = box.tolist()
        
        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
        cv2.putText(ori_frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

# 模型加载
print("load weight from:{}".format(weight_path))
model = Detector(cfg.category_num, True).to(device)
model.load_state_dict(torch.load(weight_path, map_location=device))
#将模型设置为评估（推理）模式，这将使其在推理过程中表现更稳定。
model.eval()
print("load finished")

#视频读取
video_capture = cv2.VideoCapture(video_path)
#视频写入
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output.avi', fourcc,fps, (width, height))

while True:
    ret,ori_frame = video_capture.read()
    if not ret:
        print("video load finished.")
        break;

    cv2.imshow('ori_frame',ori_frame)

    res_img = cv2.resize(ori_frame, (cfg.input_width, cfg.input_height), 
                        interpolation = cv2.INTER_LINEAR)  #线性插值方式缩放
    img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)  #整理输入格式
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))   #转成tensor格式
    img = img.to(device).float() / 255.0
    preds = model(img)
    output = handle_preds(preds, device, thresh)

    draw_cap(output, ori_frame)
    cv2.imshow('res_frame',ori_frame)

    video_writer.write(ori_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break;

# 释放视频资源
video_capture.release()
video_writer.release()

# 关闭窗口
cv2.destroyAllWindows()