import cv2
import numpy as np

from Utils.FileStorage import ReadCalibration


class Calib:
    def __init__(self, fileCalibrationPath):
        calibratedDict = ReadCalibration(read_path=fileCalibrationPath)

        self.MLS = calibratedDict["MLS"]
        self.MRS = calibratedDict["MRS"]
        self.dLS = calibratedDict["dLS"]
        self.dRS = calibratedDict["dRS"]
        self.T = calibratedDict["T"]
        self.R = calibratedDict["R"]
        self.mtxR = calibratedDict["mtxR"]
        self.mtxL = calibratedDict["mtxL"]
        self.disL = calibratedDict["disL"]
        self.disR = calibratedDict["disR"]
        self.OpMatL = calibratedDict["OpMatL"]
        self.OpMatR = calibratedDict["OpMatR"]

        # self.tvecL = calibratedDict["tvecL"]
        # self.rvecL = calibratedDict["rvecL"]
        # self.tvecR = calibratedDict["tvecR"]
        # self.rvecR = calibratedDict["rvecR"]
        self.imgShape = {"width": int(calibratedDict["width"]), "height": int(calibratedDict["height"])}

    @staticmethod
    def rectifyReturn(calib):
        kernel = np.ones((3, 3), np.uint8)
        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # StereoRectify function
        rectify_scale = 0  # if 0 image croped, if 1 image nr croped
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(calib.MLS, calib.dLS, calib.MRS, calib.dRS,
                                                          (calib.imgShape["width"], calib.imgShape["height"]), calib.R,
                                                          calib.T,
                                                          rectify_scale, (0, 0))

        # initUndistortRectifyMap function
        Left_Stereo_Map = cv2.initUndistortRectifyMap(calib.MLS, calib.dLS, RL, PL,
                                                      (calib.imgShape["width"], calib.imgShape["height"]), cv2.CV_16SC2)
        Right_Stereo_Map = cv2.initUndistortRectifyMap(calib.MRS, calib.dRS, RR, PR,
                                                       (calib.imgShape["width"], calib.imgShape["height"]), cv2.CV_16SC2)
        return (Left_Stereo_Map, Right_Stereo_Map, RL, RR, PL, PR)

    @staticmethod
    def returnRectifiedImage(leftImg, rightImg, Left_Stereo_Map, Right_Stereo_Map):
        # Rectify the images on rotation and alignement
        Left_nice = cv2.remap(leftImg, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(rightImg, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT, 0)
        return (Left_nice, Right_nice)
