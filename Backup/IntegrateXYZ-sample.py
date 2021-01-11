# For Demostration (Monday and Tuesday)
import argparse

import cv2
import numpy as np
import xlsxwriter

from Models.ObjectFound import ObjectFound
from Backup.DistanceCalculator import DepthImageProcessing
from Utils.FileStorage import ReadCalibration
from Utils.Lib import draw_prediction, convertDisToDis, Unproject, DuyCalculateXY, FIX_HEIGHT
from Utils.ObjectDetectionTools import ObjectDetection, PrepareClass
from Utils.StereoTools import returnStereoSGBM, returnDisparityMap

ObjLeft = []  # List of OBJ detected from Left Cam
ObjRight = []  # List of OBJ detected from Left Cam

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftCam', required=True, help='path to port number from Left Camera')
ap.add_argument('-r', '--RightCam', required=True, help='path to port number from Right Camera')

ap.add_argument('-cl', '--classYolo', required=True, help='Path to class name file of YoloV3')
ap.add_argument('-cfg', '--cfgYolo', required=True, help='Path to config file of YoloV3')
ap.add_argument('-w', '--weightYolo', required=True, help='Path to weight file of YoloV3')

ap.add_argument('-debug', '--ShowDisparity', action="store_true",
                required=False, help='Show Disparity Map')
ap.add_argument('-YO', '--YoloOriginFrame', action="store_true",
                required=False, help='Show original frame with object detected')
ap.add_argument('-c', '--CaliPath', required=True, help='Path to Calibration file')
ap.add_argument('-e', '--ExcelPath', required=False, help='Path to the Excel File with data of disparity and distance')

args = ap.parse_args()

# LEFT_CAM_PORT = CAMERA1_PORT
# RIGHT_CAM_PORT = CAMERA2_PORT

LEFT_CAM_PORT = int(args.LeftCam)
RIGHT_CAM_PORT = int(args.RightCam)

# *************************************************
# ***** Parameters for CAMERA CALIBRATION    ******
# *************************************************

# TODO: make this thing become object -class
FILE_PATH_FOR_CALIBRATION = str(args.CaliPath)
calibratedDict = ReadCalibration(read_path=FILE_PATH_FOR_CALIBRATION)

MLS = calibratedDict["MLS"]
MRS = calibratedDict["MRS"]
dLS = calibratedDict["dLS"]
dRS = calibratedDict["dRS"]
T = calibratedDict["T"]
R = calibratedDict["R"]
mtxR = calibratedDict["mtxR"]
mtxL = calibratedDict["mtxL"]
disL = calibratedDict["disL"]
disR = calibratedDict["disR"]
OpMatL = calibratedDict["OpMatL"]
OpMatR = calibratedDict["OpMatR"]

imgShape = (int(calibratedDict["width"]), int(calibratedDict["height"]))

# *************************************************
# ***** Parameters for Stereo Rectification  ******
# *************************************************

# Filtering
kernel = np.ones((3, 3), np.uint8)
# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# StereoRectify function
rectify_scale = 0  # if 0 image croped, if 1 image nr croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, imgShape, R, T, rectify_scale,
                                                  (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped

# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, imgShape,
                                              cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, imgShape, cv2.CV_16SC2)

mapLeft1 = Left_Stereo_Map[0]
mapLeft2 = Left_Stereo_Map[1]

window_size = 3
min_disp = 0
num_disp = 120 - min_disp
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

# TODO: make this thing become object -class

stereoSGBM = returnStereoSGBM(window_size=window_size, min_disp=min_disp, max_disp=num_disp + min_disp,
                              lmbda=lmbda, sigma=sigma, visual_multiplier=visual_multiplier)
stereo = stereoSGBM["StereoSGBM"]
stereoR = stereoSGBM["StereoR"]
stereoL = stereoSGBM["StereoL"]
wls_filter = stereoSGBM["WLS_Filter"]

# *************************************************
# ***** Parameters for Yolo Detection        ******
# *************************************************
weight_path = str(args.weightYolo)
cfg_path = str(args.cfgYolo)
class_path = str(args.classYolo)
# *************************************************
# ***** Debug Flag      ***************************
# *************************************************
flagOriginWithObjectDetected = args.YoloOriginFrame
flagShowingDisparityMap = args.ShowDisparity

# prepare dataworkbook sheet
excel_path = args.ExcelPath
workbook = xlsxwriter.Workbook(filename=excel_path)
worksheet = workbook.add_worksheet()
row_exel = 0
col_exel_dis_orig = 0
col_exel_dis_fil = 1
data_workbook = ()


def coords_mouse_disp_orig(event, x, y, flags=None, param=None):
    global row_exel
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y, filteredImg[y, x], disparityMap[y, x])
        print("Filter: " + str(filteredImg[y, x]))
        print("Origin: " + str(disparityMap[y, x]))
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disparityMap[y + u, x + v]
        average = average / 9

        Distance = convertDisToDis(average=average)
        worksheet.write(row_exel, col_exel_dis_orig, average)

        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += filteredImg[y + u, x + v]
        average = average / 9

        worksheet.write(row_exel, col_exel_dis_fil, average)

        row_exel += 1
        print('Original-Distance: ' + str(Distance) + ' m' + " Disparity: " + str(average))


def coords_mouse_getXY(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        instrictMatrix = mtxL
        depth = 85
        f_x = instrictMatrix[0][0]
        f_y = instrictMatrix[1][1]
        c_x = instrictMatrix[0][2]
        c_y = instrictMatrix[1][2]
        print(c_y)
        print(mtxL)
        print(c_x)
        print(f_y)
        print(f_x)
        (X, Y, Z) = Unproject((x, y), Z=depth, intrinsic_matrix=mtxL, distortion=disL, P=PL, R=RL)
        print(np.round(X, 2), np.round(Y, 2), np.round(Z, 2))


# DUY IMPLEMENTATION:
# return 441.6/(0.20072727272727273 * pixel_width) # (focal *  realWidth * 640)/(pixelWidth * distance)
# return (2 * 640)/ (2 * pixel_width * 20)
# def coords_mouse_getXYDUY(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         real_z = calculate_distance(pixel_width)
#         real_x = (real_z * x_coordinate) / 670
#         real_y = (real_z * y_coordinate) / 670
#         (X,Y,Z) = Unproject((x,y), Z=depth, intrinsic_matrix= mtxL, distortion= disL, P=PL, R=RL )
#         print(np.round(X, 2), np.round(Y, 2), np.round(Z,2))

def onOriginDepthCalculator(event, x, y, params, flags):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        detectedObj = ObjectFound(type="Clickbased", x=x, y=y, width=5, height=5, confidence=100)
        detectedObj.middle_x = x
        detectedObj.middle_y = y

        depth = DepthImageProcessing(time=20, leftCap=leftCap, rightCap=rightCap,
                                     Left_Stereo_Map=Left_Stereo_Map, Right_Stereo_Map=Right_Stereo_Map,
                                     wls_filter=wls_filter, min_disp=min_disp, num_disp=num_disp, stereoR=stereoR,
                                     stereo=stereo, detectedObj=detectedObj, mtx=mtxL, dis=disL, P=PL, R=RL)
        # (y, x) = DuyCalculateXY(x=x + leftDetectedObj.width / 2,
        #                         y=y + leftDetectedObj.height / 2,
        #                         pixel_width=leftDetectedObj.width)
        print(depth)
        return depth


if __name__ == '__main__':
    print("Program Start")
    classes = PrepareClass(class_path=class_path)
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    rightCap = cv2.VideoCapture(RIGHT_CAM_PORT)
    leftCap = cv2.VideoCapture(LEFT_CAM_PORT)
    while (True):
        # Collect image continously
        ret1, leftFrame = leftCap.read()
        ret2, rightFrame = rightCap.read()

        if (flagShowingDisparityMap == True):
            Left_nice = cv2.remap(leftFrame, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT, 0)
            Right_nice = cv2.remap(rightFrame, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT, 0)
            (filteredImg, disparityMap) = returnDisparityMap(Left_nice=Left_nice, Right_nice=Right_nice,
                                                             wls_filter=wls_filter, min_disp=min_disp,
                                                             num_disp=num_disp, stereoR=stereoR, stereo=stereo)
            cv2.imshow("Disparity Map (Filtered)", filteredImg)
            # cv2.imshow("Disparity Map (Original)", disparityMap)
            cv2.imshow("Rectified Images", np.hstack((Left_nice, Right_nice)))
            # cv2.setMouseCallback("Disparity Map (Filtered)", coords_mouse_disp, filteredImg)
            # cv2.setMouseCallback("Disparity Map (Original)", coords_mouse_disp_orig)
            cv2.setMouseCallback("Disparity Map (Filtered)", coords_mouse_disp_orig, filteredImg)

        Left_frame_grid = leftFrame.copy()

        cv2.line(Left_frame_grid, pt1=(320, 0), pt2=(320, 480), thickness=1, color=(0, 0, 255))
        cv2.line(Left_frame_grid, pt1=(0, 240), pt2=(640, 240), thickness=1, color=(0, 0, 255))
        # cv2.imshow("Origin", np.hstack((Left_nice_angle, Right_nice_angle)))
        Left_frame_grid = cv2.circle(Left_frame_grid,
                                     (int(mtxL[0][2]), int(mtxL[1][2])),
                                     2, (234, 21, 232), thickness=2)

        cv2.imshow("Origin", leftFrame)
        cv2.setMouseCallback("Origin", onOriginDepthCalculator, leftFrame)

        #
        # If p is hit, begin to process the current frame
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Collects Object Detected:
            print("== Begin Processing to find Object ==")
            LeftCamera_ObjectFound = ObjectDetection(leftFrame, weight=weight_path, yoloCfg=cfg_path,
                                                     confident_threshold=0.05)

            for leftDetectedObj in LeftCamera_ObjectFound:
                depth = DepthImageProcessing(threshold_check=True, time=1, leftCap=leftCap, rightCap=rightCap,
                                             Left_Stereo_Map=Left_Stereo_Map, Right_Stereo_Map=Right_Stereo_Map,
                                             wls_filter=wls_filter, min_disp=min_disp, num_disp=num_disp,
                                             stereoR=stereoR,
                                             stereo=stereo, detectedObj=leftDetectedObj, mtx=mtxL, dis=disL, P=PL, R=RL)

                depth = 100
                (y, x) = DuyCalculateXY(x=leftDetectedObj.x_pixel_coordinate + leftDetectedObj.width / 2,
                                        y=leftDetectedObj.y_pixel_coordinate + leftDetectedObj.height / 2,
                                        pixel_width=leftDetectedObj.width)
                # Draw Prediction
                draw_prediction(leftFrame,
                                class_id=leftDetectedObj.Class,
                                confidence=leftDetectedObj.confidence_level,
                                x=leftDetectedObj.x_pixel_coordinate,
                                y=leftDetectedObj.y_pixel_coordinate,
                                x_plus_w=leftDetectedObj.x_pixel_coordinate + leftDetectedObj.width,
                                y_plus_h=leftDetectedObj.y_pixel_coordinate + leftDetectedObj.height,
                                classes=classes, COLORS=COLORS, distance=depth, angle=0,
                                coor1=(np.round(x, 2), np.round(y, 2), np.round(FIX_HEIGHT - depth, 2)))

            while (True):
                cv2.imshow("Origin", leftFrame)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    print("Quit Object Detection")
                    break

            cv2.destroyWindow("Object Left Frame")
            cv2.destroyWindow("Object Left Nice")
            if (flagShowingDisparityMap == True):
                cv2.destroyWindow("Disparity Map")
            print("Finish Processing")

        if cv2.waitKey(3) & 0xFF == ord('q'):
            print("Program Ends")
            break

    rightCap.release()
    leftCap.release()
    cv2.destroyAllWindows()
workbook.close()
