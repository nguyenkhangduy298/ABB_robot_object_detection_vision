import math

import cv2
import numpy as np

from Models.Calib import Calib
from Models.ObjectFound import ObjectFound

def OrientationCalculate(detectedObject : ObjectFound, calib : Calib, img):
    marker = detectedObject.MarkerCorners
    marker_corners = np.array(marker)

    corner = np.array(marker_corners[0][0])

    corners_convert = np.array([corner[2], corner[3], corner[1], corner[0]])
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
    objp = np.zeros((2 * 2, 3), np.float32)
    objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners_convert, calib.mtxL, calib.disL)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, calib.mtxL, calib.disL)

    rmat = cv2.Rodrigues(rvecs)[0]
    proj_matrix = np.hstack((rmat, tvecs))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    print("P/R/Y : ", pitch * 57.2957795, roll * 57.2957795, yaw * 57.2957795)
    pitch = pitch * 57.2957795
    yaw = yaw * 57.2957795
    cv2.putText(img, "Yaw:  %.2f" % yaw, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    cv2.putText(img, "Pitch: %.2f" % pitch, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    cv2.putText(img, "Roll: 180", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

    # Handle PYR robot inverse vector from the object normal vector
    if pitch < 8:  # If lower than X degree than set to directly straight angle
        pitch = 0
    if yaw < 8:  # If lower than X degree than set to directly straight angle
        yaw = 0
    if pitch < 0:  # Could be reverse >0 and <0 - Needs Testing With Robot
        pitch = 180 - abs(pitch)  # Object Negative then robot Positive
    else:
        pitch = (180 - abs(pitch)) * -1  # Object Positive then robot Negative
    if yaw < 0:  # Could be reverse >0 and <0 - Needs Testing With Robot
        yaw = 0 - abs(yaw)
    else:
        yaw = (0 - abs(yaw)) * -1
    pose = (pitch, yaw, 180)


    return pose


