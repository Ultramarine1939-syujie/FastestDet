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
img_path = "data/1.jpg"
thresh = 0.65

# 解析yaml配置文件
cfg = LoadYaml(yaml_path)    
print(cfg) 

# 模型加载
print("load weight from:{}".format(weight_path))
model = Detector(cfg.category_num, True).to(device)
model.load_state_dict(torch.load(weight_path, map_location=device))
#sets the module in eval node
model.eval()
print("load finished")

# 数据预处理
ori_img = cv2.imread(img_path)
res_img = cv2.resize(ori_img, (cfg.input_width, cfg.input_height), 
                     interpolation = cv2.INTER_LINEAR)  #线性插值方式缩放
img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)  #整理输入格式
img = torch.from_numpy(img.transpose(0, 3, 1, 2))   #转成tensor格式
img = img.to(device).float() / 255.0

# 模型推理
start = time.perf_counter()
preds = model(img)
end = time.perf_counter()
time = (end - start) * 1000.
print("forward time:%fms"%time)

output = handle_preds(preds, device, thresh)

# 加载label names
LABEL_NAMES = []
with open(cfg.names, 'r') as f:
	    for line in f.readlines():
	        LABEL_NAMES.append(line.strip())
    
H, W, _ = ori_img.shape
scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

# 绘制预测框
for box in output[0]:
    print(box)
    box = box.tolist()
       
    obj_score = box[4]
    category = LABEL_NAMES[int(box[5])]

    x1, y1 = int(box[0] * W), int(box[1] * H)
    x2, y2 = int(box[2] * W), int(box[3] * H)

    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
    cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

cv2.imwrite("result.png", ori_img)