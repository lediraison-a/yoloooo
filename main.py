import torch
import cv2
import cv2
import os
from PIL import Image

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Transform video into frames
video = "videos/mathis_squat.mp4"  # or file, Path, PIL, OpenCV, numpy, list
vidcap = cv2.VideoCapture(video)
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("images/frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1


list_images = os.listdir("images")

for image in list_images:
    results = model("images/"+image)
    # Inference
    results.crop()
