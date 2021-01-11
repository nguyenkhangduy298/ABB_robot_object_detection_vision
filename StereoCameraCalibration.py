# -*- coding: utf-8 -*-
# Thís module collects image taken from two cameras and deduce the instrict matrix and distortion matrix.
# Take 4 inputs of text file dictionary to images file:
#
import argparse

import numpy as np
from cv2 import cv2

from Utils.FileStorage import SaveCalibration

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--LeftFolder', required=True, help='path to dictionary of images taken from Left Camera')
ap.add_argument('-r', '--RightFolder', required=True, help='path to dictionary of images taken from Right Camera')

ap.add_argument('-il', '--LowerIndex', required=True, help='Lower range for index of image file')
ap.add_argument('-iu', '--UpperIndex', required=True, help='Upper range for index of image file')

ap.add_argument('-cw', '--chesswidth', required=True, help='Chess Width')
ap.add_argument('-ch', '--chessheight', required=True, help='Chess Height')

ap.add_argument('-show', '--ShowImage', action="store_true", required=False,
                help='Show image of calibrated picture with chessboard detected')

ap.add_argument('-o', '--output', required=True, help='path to the output file')
ap.add_argument('-d', '--dimension', required=True, help='Dimension (in mm) of chessboard square')
args = ap.parse_args()

# COLLECT ARGUMENT:
PATH_CALIBRATION_IMAGE_CAM1 = args.LeftFolder
PATH_CALIBRATION_IMAGE_CAM2 = args.RightFolder

LOWER_INDEX = int(args.LowerIndex)
UPPER_INDEX = int(args.UpperIndex)

ChessWidth = int(args.chesswidth)
ChessHeight = int(args.chessheight)

square_size = int(args.dimension)

DESTINATION_FILE = args.output  # Output YAML

FLAG_SHOW_CHESSBOARD_IMG = args.ShowImage

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((ChessWidth * ChessHeight, 3), np.float32)
objp[:, :2] = np.mgrid[0:ChessWidth, 0:ChessHeight].T.reshape(-1, 2)
objp = objp * square_size
# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []

if __name__ == '__main__':
    # Start calibration from the camera
    print('Starting calibration for the 2 cameras... ')
    print(PATH_CALIBRATION_IMAGE_CAM1)
    print(PATH_CALIBRATION_IMAGE_CAM2)
    print("Hit n to accept the image, hit q to ignore it")
    # Call all saved images
    for i in range(LOWER_INDEX,
                   UPPER_INDEX):  # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
        t = str(i)
        print(t)
        originL = cv2.imread(PATH_CALIBRATION_IMAGE_CAM1 + t + '.png')
        originR = cv2.imread(PATH_CALIBRATION_IMAGE_CAM2 + t + '.png')
        ChessImaL = cv2.imread(PATH_CALIBRATION_IMAGE_CAM1  + t + '.png', 0)  # Left side
        ChessImaR = cv2.imread(PATH_CALIBRATION_IMAGE_CAM2 + t + '.png', 0)  # Right side

        retR, cornersR = cv2.findChessboardCorners(ChessImaR, (ChessWidth, ChessHeight),
                                                   None)  # Define the number of chees corners we are looking for
        retL, cornersL = cv2.findChessboardCorners(ChessImaL, (ChessWidth, ChessHeight), None)  # Left side
        cv2.drawChessboardCorners(originR, (ChessWidth, ChessHeight), cornersR, retR)
        cv2.drawChessboardCorners(originL,(ChessWidth, ChessHeight), cornersL, retL)

        while True:
            cv2.imshow('Corner', np.hstack((originL, originR)))
            if (cv2.waitKey(1) & 0xFF == ord('n')):
                print("Couple with " + t + " has been added")
                if (True == retR) & (True == retL):
                    objpoints.append(objp)
                    cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
                    cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
                    imgpointsR.append(cornersR)
                    imgpointsL.append(cornersL)
                break
            elif (cv2.waitKey(2) & 0xFF == ord('q')):
                break
    cv2.destroyAllWindows()
    # Determine the new values for different parameters
    #   Right Side
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)
    hR, wR = ChessImaR.shape[:2]
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    #   Left Side
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)
    hL, wL = ChessImaL.shape[:2]
    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    print('Cameras Ready to use')
    print("Camera calibration value: " + str(retL) + " and " + str(retR))

    # ********************************************
    # ***** Calibrate the Cameras for Stereo *****
    # ********************************************

    # StereoCalibrate function
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR,
                                                               mtxL, distL, mtxR, distR,
                                                               ChessImaR.shape[::-1], criteria_stereo, flags)
    print("Stereo Camera calibration value: " + str(retS))
    log = SaveCalibration(DESTINATION_FILE, MRS=MRS, MLS=MLS,
                          dRS=dRS, dLS=dLS, T=T, R=R,

                          shapeWidth=ChessImaR.shape[::-1][0],
                          shapeHeight=ChessImaR.shape[::-1][1],
                          mtxL=mtxL, mtxR=mtxR, disL=distL, disR=distR, OpMatL=OmtxL, OpMatR=OmtxR,
                          )
    print(log)
