# For Demostration (Monday and Tuesday)
import argparse
import math

import cv2
import numpy as np

from Models.Calib import Calib
from Utils.Lib import calculateXY, parseMatrix, convertDisToZ, calculateNewCenters
import socket

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftCam', required=True, help='path to port number from Left Camera')
ap.add_argument('-r', '--RightCam', required=True, help='path to port number from Right Camera')
ap.add_argument('-c', '--CaliPath', required=True, help='Path to Calibration file')

args = ap.parse_args()
LEFT_CAM_PORT = int(args.LeftCam)
RIGHT_CAM_PORT = int(args.RightCam)
TCP = False


def toDepth(z_robot):
    return (973.50 + 185.50)  - z_robot

# X ROBOT AT ORIGIN (REC LEFT): 687.44
# Y ROBOT AT ORIGIN : -124.85

def toX_Robot(x, const):
    return x + const

def toY_Robot(y, const):
    return y + const

def depthDirect(base, focal, disparity):
    return base*focal/disparity

def mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x,y,disp[y,x],filteredImg[y,x]
        average = 0
        average_origin = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
                average_origin += disp_origin[y + u, x + v]
        average = average / 9
        average_origin = average_origin // 9

        length_tool = 1
        z = convertDisToZ(average) + length_tool

        z_to_pick = z
        (x, y) = calculateXY(instrictMatrix=calib.mtxL, depth=toDepth(z), x_screen=x, y_screen=y)
        # depth = 281


        beta = 1  # Degree on x axis
        alpha = 1 # Degree on y axis

        #ORIGINAL POINT A: (650.79, 29.33,452.17); B: (660.30,11.00,698.32)
        (new_center_x, new_center_y) = calculateNewCenters(650.79, 29.33,452.17,
                                                           660.30,11.00,698.32,
                                                           z)
        print("New Center: {} {}".format(new_center_x, new_center_y))
        x_robot = toX_Robot(y, new_center_x)/math.cos(math.radians(beta))
        y_robot = toY_Robot(x, new_center_y)/math.cos(math.radians(alpha))

        print((x_robot, y_robot, z_to_pick))
        print("Real Z" + str(z))

        if (TCP):
            robot_coordinate = [x_robot, y_robot, z_to_pick]
            robot_rot = [-180,0,-180]
            command = str(robot_coordinate) + str(robot_rot)
            print("Robot Coordinate Results:")
            print(robot_coordinate)
            receive = ""
            print("Sutck here")
            while receive != "0xFC":
                print(receive)
                receive = s.recv(1024).decode('utf-8')
                print("Second" + str(receive))

                if (receive == "0xFD"):
                    print("Object delivered/n")
                elif (receive == "0xFC"):
                    print("Stand by/n")
                    break
                elif (receive == "0xFF" or receive == "0xFF0xFF"):
                    print("hello")
                    data = command
                    print("Object location request/n")
                    s.send(data.encode())
                elif (receive == "0xFB"):
                    print("Moving../n")
                elif (receive == "0xFA"):
                    print("server closed/n")
                    break
                elif (receive == ""):
                    pass;
                else:
                    print(".", type(receive))
            print("Finished Sending")

        print("Disparity Value:  " + str(average) + " Origin:" + str(average_origin) + " Z-robot " + str(z))


FILE_PATH_FOR_CALIBRATION = str(args.CaliPath)

calib = Calib(FILE_PATH_FOR_CALIBRATION)
# Filtering
(Left_Stereo_Map, Right_Stereo_Map, PL, RL, PR, RR) = Calib.rectifyReturn(calib)


# print(Left_Stereo_Map)
# print(Right_Stereo_Map)

# Create trackbar for Disparity & Stereo
def empty(x):
    pass


# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100, speckleRange=32, disp12MaxDiff=5,
                               P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2
                               # ,mode=cv2.STEREO_SGBM_MODE_HH4
                               )

if __name__ == '__main__':
    print("Program Start")
    rightCap = cv2.VideoCapture(RIGHT_CAM_PORT)
    leftCap = cv2.VideoCapture(LEFT_CAM_PORT)


    if (TCP):
    #Setup TCP/IP:
        TCP_IP = '192.168.125.1'
        TCP_PORT = 1025
        BUFFER_SIZE = 1024

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        print("Connected.")

    while True:
        _, leftFrame = leftCap.read()
        _, rightFrame = rightCap.read()
        # Rectify Images
        Left_nice = cv2.remap(leftFrame, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(rightFrame, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT, 0)

        # Draw a diagonal blue line with thickness of 5 px
        image = Left_nice.copy()
        cv2.line(image, (320, 0), (320, 480), (255, 0, 0), 1)
        cv2.line(image, (0, 240), (640, 240), (255, 0, 0), 1)
        cv2.imshow("Vision System", image)

        # Convert from color(BGR) to gray

        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        copyLeft = Left_nice.copy()

        (fx, fy, cx, cy) = parseMatrix(calib.mtxL)
        cv2.circle(copyLeft, (int(cx), int(cy)), radius=3, color=(100, 100, 100), thickness=2)
        cv2.imshow("CP", copyLeft)
        # cv2.imshow("Gray Rectified", np.hstack((grayL, grayR)))

        # Calculate Disp
        disp_origin = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
        disp = ((disp_origin.astype(
            np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

        cv2.imshow("Depth", disp)

        cv2.setMouseCallback("Depth", mouse_disp, disp)
        #cv2.setMouseCallback("CP", mouse_disp, disp)

        #cv2.imshow("Origin Left", leftFrame)
        cv2.imshow("Rectified", np.hstack((Left_nice, Right_nice)))

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    rightCap.release()
    leftCap.release()
    cv2.destroyAllWindows()

# trackbarNameWin = "Console"
#
# mindispBar = "MinDisp"
# numdispBar = "NumDisp"
# blocksizeBar = "BlockSize"
# P1Bar = "P1Bar"
# P12Diff = "DifferentP1-P2"
# Disp12MD = "Disp12MDBar"
# PFC = "PreferFilterCap"
# URatio = "URatioBar"
# SWindowSize = "SWindowSize"
# SWindowRange = "SWindowRange"
# Mode = "ModeSelection"


# Temp
# #Create StereoSGBM
# # TODO: MAKE THIS INTO A FUCKING OBJECT
# window_size = cv2.getTrackbarPos(trackbarname=blocksizeBar, winname=trackbarNameWin)
# min_disp = cv2.getTrackbarPos(trackbarname=mindispBar, winname=trackbarNameWin)
# # num_disp must be divisible by 16
# num_disp = cv2.getTrackbarPos(trackbarname=numdispBar, winname=trackbarNameWin) * 16
# if (num_disp == 0):
#     num_disp = 1
# P1 = cv2.getTrackbarPos(trackbarname=P1Bar, winname=trackbarNameWin)
# diffP1P2 = cv2.getTrackbarPos(P12Diff, winname=trackbarNameWin)
# P2 = P1 + diffP1P2
# d12md = cv2.getTrackbarPos(trackbarname=Disp12MD, winname=trackbarNameWin)
# pfc = cv2.getTrackbarPos(trackbarname=PFC, winname=trackbarNameWin)
# ratio = cv2.getTrackbarPos(trackbarname=URatio, winname=trackbarNameWin)
# sws = cv2.getTrackbarPos(trackbarname=SWindowSize, winname=trackbarNameWin)
# sr = cv2.getTrackbarPos(trackbarname=SWindowRange, winname=trackbarNameWin)
#
# stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
#                                P1=P1,P2=P2, disp12MaxDiff=d12md,preFilterCap=pfc, uniquenessRatio=ratio,
#                                speckleWindowSize=sws, speckleRange=sr)
# stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size, mode=cv2.STEREO_SGBM_MODE_HH4)


# cv2.namedWindow(trackbarNameWin)
# cv2.createTrackbar( mindispBar, trackbarNameWin, 0, 255,empty)
# cv2.createTrackbar(numdispBar, trackbarNameWin, 1, 255, empty)
# cv2.createTrackbar(blocksizeBar, trackbarNameWin, 0, 255, empty)
# cv2.createTrackbar(P1Bar, trackbarNameWin, 0, 255, empty)
# cv2.createTrackbar(P12Diff, trackbarNameWin, 0,255, empty)
# cv2.createTrackbar(Disp12MD, trackbarNameWin, 0, 255, empty)
# cv2.createTrackbar(PFC, trackbarNameWin, 0,255, empty)
# cv2.createTrackbar(URatio, trackbarNameWin, 0, 255, empty)
# cv2.createTrackbar(SWindowSize, trackbarNameWin, 0, 255, empty)
# cv2.createTrackbar(SWindowRange, trackbarNameWin, 0, 255, empty)
# cv2.createTrackbar(Mode, trackbarNameWin, 0, 3, empty)

# stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=numDisp, blockSize=None, P1=None, P2=None, disp12MaxDiff=None,
#                   preFilterCap=None, uniquenessRatio=None, speckleWindowSize=None, speckleRange=None,
#                   mode=None): # real signature unknown; restored from __doc__
