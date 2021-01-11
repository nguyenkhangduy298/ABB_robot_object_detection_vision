import argparse
import random
import socket

import cv2

from Models.BoundingBoxWidget import BoundingBoxWidget
from Models.Calib import Calib
from Models.YoloObject import Yolo
from Utils.ObjectDetectionTools import ObjectDetection, PrepareClass
from Utils.ObjectProcessing  import *

ap = argparse.ArgumentParser()
# Cameras and Cameras Calibration parametesr
ap.add_argument('-l', '--LeftCam', required=True, help='path to port number from Left Camera') # Port to Left Camera
ap.add_argument('-r', '--RightCam', required=True, help='path to port number from Right Camera') # Port To Right Camera
ap.add_argument('-c', '--CaliPath', required=True, help='Path to Calibration file') # Calibration files for cameras


args = ap.parse_args()
LeftCameraPort = int(args.LeftCam) # Get Port Number and assign it to correct cap object (Left camera)
RightCameraPort = int(args.RightCam) # Get Port Number and assign it to correct cap object (right camera)
url_calibPath = str(args.CaliPath) # Get Cali Path
calib = Calib(url_calibPath) # Create calib object from calib path
(Left_Stereo_Map, Right_Stereo_Map, _,_,_,_) = Calib.rectifyReturn(calib) # Create Left and Right Stereo map

# This function take the array of objects. Set the PickPoint for each Objects
if __name__ == '__main__':
    #setupSystem() # Setup Camera and Calib & Yolo Object

    camL = cv2.VideoCapture(LeftCameraPort); camR = cv2.VideoCapture(RightCameraPort) # Assign and register cam
    # Create StereoSGBM and prepare all parameters TODO: Add this to setup
    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100, speckleRange=32, disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2
                                   # ,mode=cv2.STEREO_SGBM_MODE_HH4
                                   )



    while True:
        _, frameLeft = camL.read(); _, frameRight = camR.read() # Frame Left and Right captured fromt he images; no processing yet
        # If detect auto -> Using Yolo to find the object, if not, select the areas by hand
        cpLeft = frameLeft.copy()
        cv2.imshow("Left and Right", np.hstack((frameLeft, frameRight)))
        if ((cv2.waitKey(1) & 0xFF == ord('e'))):

            # Data:
            (Left_Stereo_Map, Right_Stereo_Map, RL, RR, PL, PR) = calib.rectifyReturn(calib)

            (LeftNice, RightNice) = calib.returnRectifiedImage(frameLeft, frameRight, Left_Stereo_Map,
                                                               Right_Stereo_Map)
            grayL = cv2.cvtColor(LeftNice, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(RightNice, cv2.COLOR_BGR2GRAY)
            # Calculate Disp
            disp_origin = stereo.compute(grayL, grayR)
            disp = ((disp_origin.astype(
                np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

            (ret, middleX, middleY, pitch, yaw, roll) = aruco_processing(calib.mtxL, calib.disL,img=frameLeft)


            #Find the marker
            if (ret):
                (x_rectified, y_rectified) = findCorresPixel(x=middleX, y=middleY, mtxL=calib.mtxL,
                                                             disL=calib.disL, P=PL,
                                                             R=RL)  # Convert marker middle points to
                disp_val = getDispAtPoint(x=x_rectified, y=y_rectified, disp=disp)
                z = convertDisToZ(disp_val) - 10

                depth_to_camera = FromZToDepth(z)
                (x, y) = calculateXY(instrictMatrix=calib.mtxL, depth=depth_to_camera, x_screen=x_rectified,
                                     y_screen=y_rectified)
                # ORIGINAL POINT A: (650.79, 29.33,452.17); B: (660.30,11.00,698.32)
                (new_center_x, new_center_y) = calculateNewCenters(pointA.x, pointA.y, pointA.z,
                                                                   pointB.x, pointB.y, pointB.z,
                                                                   z)

                # print("New Center: {} {}".format(new_center_x, new_center_y))
                x_robot = toX_Robot(y, new_center_x) / math.cos(math.radians(BETA))
                y_robot = toY_Robot(x, new_center_y) / math.cos(math.radians(ALPHA))

                print("X :{} ;Y {}; Z {}; Pitch {} Yaw {}; Roll {}".format(x_robot-120,y_robot,z,pitch,yaw,roll))

                # while (True):
                #
                #     cv2.imshow("Result", frameLeft )
                #     if (cv2.waitKey(1) & 0xFF == ord('q')):
                #         cv2.destroyWindow("Result")
                #         break
            else:
                print("No marker found")
                continue

    camL.release()
    camR.release()
    cv2.destroyAllWindows()