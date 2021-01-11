import argparse

from cv2 import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftCamera', required=True, help='Left Camera: Port')
ap.add_argument('-r', '--RightCamera', required=True, help='Right Camera: Port')
args = ap.parse_args()

PORTL = int(args.LeftCamera)
PORTR = int(args.RightCamera)

capL = cv2.VideoCapture(PORTL)
capR = cv2.VideoCapture(PORTR)


capL.set(cv2.CAP_PROP_AUTOFOCUS, 0)
capR.set(cv2.CAP_PROP_AUTOFOCUS, 0)


while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    cv2.line(frameL, pt1=(320, 0), pt2=(320, 480), thickness=1, color=(0, 0, 255))
    cv2.line(frameL, pt1=(0, 240), pt2=(640, 240), thickness=1, color=(0, 0, 255))
    cv2.line(frameR, pt1=(320, 0), pt2=(320, 480), thickness=1, color=(0, 0, 255))
    cv2.line(frameR, pt1=(0, 240), pt2=(640, 240), thickness=1, color=(0, 0, 255))

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)


    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()
capL.release()
capR.release()
