# https://github.com/nicknochnack/YOLO-Drowsiness-Detection/blob/main/Drowsiness%20Detection%20Tutorial.ipynb
#https://www.kaggle.com/code/gauravsrivastav2507/drowsiness-detection-using-yolov8
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
labels = ['awake', 'drowsy']

img = 'duermy.png'

results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()

results.render()