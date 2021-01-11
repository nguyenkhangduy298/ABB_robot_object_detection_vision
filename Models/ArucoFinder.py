import math

from cv2 import aruco, cv2
import numpy as np

from Models.Calib import Calib
from Models.ObjectFound import ObjectFound


class ArucoFinder:
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    

    @classmethod
    def ReturnPose(cls, detectedObject: ObjectFound, mtx, dist):
        gray = cv2.cvtColor(detectedObject.img_cropped, cv2.COLOR_BGR2GRAY)
        marker_corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, cls.aruco_dict, parameters=cls.arucoParameters)

        if not marker_corners:
            return (180,0,0)

        max_area = 0
        choosen_coner = None
        for coner in marker_corners:
            if cv2.contourArea(coner) > max_area:
                max_area = cv2.contourArea(coner)
                choosen_coner = coner

        choosen_coner = np.array(choosen_coner[0])

        corners_convert = np.array([choosen_coner[2], choosen_coner[3], choosen_coner[1], choosen_coner[0]])
        objp = np.zeros((2 * 2, 3), np.float32)
        objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
        # print(objp)
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners_convert, mtx, dist)
        # Get Degree Up
        rmat = cv2.Rodrigues(rvecs)[0]
        proj_matrix = np.hstack((rmat, tvecs))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
        #print("P/R/Y : ", pitch * 57.2957795, roll * 57.2957795, yaw * 57.2957795)
        pitch = pitch * 57.2957795
        yaw = yaw * 57.2957795
        # cv2.putText(img, "Yaw:  %.2f" % yaw, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        # cv2.putText(img, "Pitch: %.2f" % pitch, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        # cv2.putText(img, "Roll: 180", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

        # Handle PYR robot inverse vector from the object normal vector
        if pitch < 9:  # If lower than X degree than set to directly straight angle
            pitch = 0
        if yaw < 9:  # If lower than X degree than set to directly straight angle
            yaw = 0
        if pitch < 0:  # Could be reverse >0 and <0 - Needs Testing With Robot
            pitch = 180 - abs(pitch)  # Object Negative then robot Positive
        else:
            pitch = (180 - abs(pitch)) * -1  # Object Positive then robot Negative
        if yaw < 0:  # Could be reverse >0 and <0 - Needs Testing With Robot
            yaw = 0 - abs(yaw)
        else:
            yaw = (0 - abs(yaw)) * -1
        return (pitch, yaw, 180)

    @classmethod
    def ReturnPoseAdv(cls, detectedObjet : ObjectFound):
        img = detectedObjet.img_cropped



    @classmethod
    def findCorners(cls, frame,x,y):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, cls.aruco_dict, parameters=cls.arucoParameters)

        max_area = 0
        choosen_coner = None
        for coner in corners:
            if cv2.contourArea(coner) > max_area:
                max_area = cv2.contourArea(coner)
                choosen_coner = coner

        if (choosen_coner is not None):
            choosen_coner = choosen_coner + [x,y] # Convert to original, pixel coordinate system

        return choosen_coner

    @classmethod
    def returnPose(cls, corner, img, calib: Calib):

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

        return img