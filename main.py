import torch
import cv2
import cv2
import os
from PIL import Image
import shutil

import cv2 as cv
import numpy as np
import argparse
import math
import shutil

def checkSquat(image, BODY_PARTS, POSE_PAIRS, inWidth, inHeight, rep, testRep):

    angleL = 999
    angleR = 999

    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]


        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            # draw lines between the detected points
            cv.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            # Add square to the whole body
            if partFrom == "Nose" and partTo == "LAnkle":
                x1, y1 = points[idFrom]
                x2, y2 = points[idTo]
                w, h = abs(x2-x1), abs(y2-y1)
                x, y = min(x1, x2), min(y1, y2)
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        '''Calcule de l'angle au niveau des genoux'''
        if points[8] and points[9] and points[10]:
            cuisseL = math.sqrt((points[8][0]-points[9][0]) ** 2 + (points[8][1]-points[9][1]) ** 2)
            moletL = math.sqrt((points[9][0]-points[10][0]) ** 2 + (points[9][1]-points[10][1]) ** 2)
            videL = math.sqrt((points[8][0]-points[10][0]) ** 2 + (points[8][1]-points[10][1]) ** 2)
            angleL = float(math.degrees(math.acos((cuisseL * cuisseL + moletL * moletL - videL * videL) / (2.0 * cuisseL * moletL))))
            print("left angle : ", angleL)
        if points[11] and points[12] and points[13]:
            cuisseR = math.sqrt((points[11][0]-points[12][0]) ** 2 + (points[11][1]-points[12][1]) ** 2)
            moletR = math.sqrt((points[12][0]-points[13][0]) ** 2 + (points[12][1]-points[13][1]) ** 2)
            videR = math.sqrt((points[11][0]-points[13][0]) ** 2 + (points[11][1]-points[13][1]) ** 2)
            angleR = float(math.degrees(math.acos((cuisseR * cuisseR + moletR * moletR - videR * videR) / (2.0 * cuisseR * moletR))))
            print('right angle : ', angleR)
        if points[idFrom] and points[idTo]:
            cv.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            if angleL != 999:
                cv.putText(image,'%.2f' % angleL, (points[9]),cv.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), 3)
            if angleR != 999:
                cv.putText(image,'%.2f' % angleR, (points[12]),cv.FONT_HERSHEY_SIMPLEX,0.75, (0 , 0, 0), 3)
        if((angleR < 120.0 or angleL < 120.0)and testRep == 0):
            print("+1 : ",testRep, rep)
            rep = rep + 1
            testRep = 1
        if(testRep == 1 and angleL > 170.0 and angleR > 170.0):
            print("rien du tout ", testRep)
            testRep = 0

        # Les points concernant les jambes sont 8, 9, 10 pour la jambe droite et 11, 12, et 13 pour la jambe gauche on va donc chercher à trouver l'angle entre ces points.

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(image, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.putText(image, 'Nombre de rep : %.2f ' % rep, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('OpenPose using OpenCV', image)
    return rep, testRep




parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")


# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Transform video into frames
video = "videos/thomas_squat.mp4"  # or file, Path, PIL, OpenCV, numpy, list
vidcap = cv2.VideoCapture(video)
success,image = vidcap.read()
count = 0
rep=0
while success:
    cv2.imwrite("images/frame%04d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1


list_images = sorted(os.listdir("images"))

for image in list_images:
    results = model("images/"+image)
    # Inference
    results.crop()


# Affichage d'un message à l'utilisateur
print("Appuyez sur une touche pour continuer...")

# Attente d'une entrée utilisateur
input()

rep = 0
i=1
testRep = 0   
while i < len(list_images):
    nbCrop = str(i)
    nbFrame = str(i-1).rjust(4, '0')
    if(i==1):
        nbCrop = ""
    frame = cv2.imread('C:\ISEN\GL\OpenPose\yoloooo\\runs\detect\exp'+nbCrop+'\crops\person\\frame'+nbFrame+'.jpg')
    rep, testRep = checkSquat(frame, BODY_PARTS, POSE_PAIRS, inWidth, inHeight, rep, testRep)
    cv2.waitKey(30)
    print('nb rep = '+str(rep))
    i=i+1

cv2.destroyAllWindows() # fermer toutes les fenêtres

# Delete generated frames
print("Removing generated frames ...")
for file in os.listdir('images'):
    os.remove("images/"+file)

# Delete cropped images & directories
print("Removing /runs directory ...")
shutil.rmtree("runs/")