import argparse

import numpy as np
from cv2 import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftCamera', required=True, help='Left Camera: Port')
ap.add_argument('-r', '--RightCamera', required=True, help='Right Camera: Port')

args = ap.parse_args()
leftCam = cv2.VideoCapture(int(args.LeftCamera))
rightCam = cv2.VideoCapture(int(args.RightCamera))

leftCam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
rightCam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
leftCam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
rightCam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
leftCam.set(cv2.CAP_PROP_AUTO_WB, 0)
rightCam.set(cv2.CAP_PROP_AUTO_WB, 0)


def none(x):
    pass


leftBrightness = "LeftBrightness"
rightBrightness = "RightBrightness"
leftExpo = "LeftExpo"
rightExpo = "RightExpo"
trackbarName = "Console"
cv2.namedWindow(trackbarName)
cv2.createTrackbar(leftBrightness, trackbarName, 0, 255, none)
cv2.createTrackbar(rightBrightness, trackbarName, 0, 255, none)
cv2.createTrackbar(leftExpo, trackbarName, 0, 5, none)
cv2.createTrackbar(rightExpo, trackbarName, 0, 5, none)

leftCam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
rightCam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

leftCam.set(cv2.CAP_PROP_FPS, 30)
rightCam.set(cv2.CAP_PROP_FPS, 30)

while True:
    _, leftframe = leftCam.read()
    _, rightframe = rightCam.read()

    leftBrightness_val = cv2.getTrackbarPos(leftBrightness, trackbarName)
    rightBrightness_val = cv2.getTrackbarPos(rightBrightness, trackbarName)
    rightExpo_val = cv2.getTrackbarPos(rightExpo, trackbarName)
    leftExpo_val = cv2.getTrackbarPos(leftExpo, trackbarName)

    leftCam.set(cv2.CAP_PROP_BRIGHTNESS, leftBrightness_val)
    rightCam.set(cv2.CAP_PROP_BRIGHTNESS, rightBrightness_val)
    print("Left Exposure: " + str(leftCam.get(cv2.CAP_PROP_EXPOSURE)))
    print("Right Exposure" + str(rightCam.get(cv2.CAP_PROP_EXPOSURE)))

    # leftCam.set(cv2.CAP_PROP_EXPOSURE, leftExpo_val )
    # rightCam.set(cv2.CAP_PROP_EXPOSURE, rightExpo_val )

    cv2.imshow("Window", np.hstack((leftframe, rightframe)))
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()
leftCam.release()
rightCam.release()
