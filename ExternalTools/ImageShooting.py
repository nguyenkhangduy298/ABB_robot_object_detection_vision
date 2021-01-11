# This python control two cameras, running it to capture and save images
# when people press space. Take arguments of cameras ports and the directory
# that store images taken.

import argparse
import ntpath
import os
import time
from datetime import datetime

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftCamera', required=True, help='Left Camera: Port')
ap.add_argument('-r', '--RightCamera', required=True, help='Right Camera: Port')
ap.add_argument('-lf', '--LeftCameraFolder', required=True, help='Directory to the folder of Left Camera\'s Pictures')
ap.add_argument('-rf', '--RightCameraFolder', required=True, help='Directory to the folder of Right Camera\'s Pictures')

args = ap.parse_args()

CAMERA_PORT_L = int(args.LeftCamera)
CAMERA_PORT_R = int(args.RightCamera)
CAMERA_FOLDER_L = str(args.LeftCameraFolder)
CAMERA_FOLDER_R = str(args.RightCameraFolder)


if not os.path.exists(CAMERA_FOLDER_L):
    print("Not exist, create:" + CAMERA_FOLDER_L)
    os.makedirs(CAMERA_FOLDER_L)
if not os.path.exists(CAMERA_FOLDER_R):
    print("Not exist, create:" + CAMERA_FOLDER_R)
    os.makedirs(CAMERA_FOLDER_R)

camL = cv2.VideoCapture(CAMERA_PORT_L)
camR = cv2.VideoCapture(CAMERA_PORT_R)

cv2.namedWindow("test")

img_counter = 0

while True:
    retR, frameR = camR.read()
    retL, frameL = camL.read()
    if not retR and retL:
        print("failed to grab frame")
        break
    cv2.imshow("test", np.hstack((frameL, frameR)))
    cv2.waitKey(1)

    print("Counter: " + str(img_counter * 2))

    img_nameL = CAMERA_FOLDER_L + str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")) + ".png"
    retL = cv2.imwrite(img_nameL, frameL)
    img_nameR = CAMERA_FOLDER_R + str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")) + ".png"
    retR = cv2.imwrite(img_nameR, frameR)

    print("Save status: L:" + str(retL) + " R:" + str(retR))
    img_counter += 1
    time.sleep(3)

camR.release()
camL.release()

cv2.destroyAllWindows()
