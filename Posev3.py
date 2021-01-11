import math

import cv2
import numpy as np
from cv2 import aruco

from Models.Calib import Calib
from Utils.FileStorage import ReadSingleCalib


def caliCam():
    (mtx, dis) = ReadSingleCalib("WebCamDell.yaml")
    Calibrate = Calib("ABBCalib7Jan.yaml")
    return (Calibrate.mtxL, Calibrate.disL)


port = cv2.VideoCapture(1)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
arucoParameters = aruco.DetectorParameters_create()

(mtx, dis) = caliCam()


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return (x, y, z)


def rot_matrix_to_euler(R):
    y_rot = math.asin(R[2][0])
    x_rot = math.acos(R[2][2] / math.cos(y_rot))
    z_rot = math.acos(R[0][0] / math.cos(y_rot))
    y_rot_angle = y_rot * (180 / math.pi)
    x_rot_angle = x_rot * (180 / math.pi)
    z_rot_angle = z_rot * (180 / math.pi)

    yaw = x_rot_angle
    pitch = y_rot_angle
    roll = z_rot_angle

    return yaw, pitch, roll


def yawpitchrolldecomposition(R):
    sin_x = math.sqrt(R[2, 0] * R[2, 0] + R[2, 1] * R[2, 1])
    validity = sin_x < 1e-6
    if not validity:
        z1 = math.atan2(R[2, 0], R[2, 1])  # around z1-axis
        x = math.atan2(sin_x, R[2, 2])  # around x-axis
        z2 = math.atan2(R[0, 2], -R[1, 2])  # around z2-axis
    else:  # gimbal lock
        z1 = 0  # around z1-axis
        x = math.atan2(sin_x, R[2, 2])  # around x-axis
        z2 = 0  # around z2-axis
    yaw = z1
    pitch = x
    roll = z2
    return yaw, pitch, roll


#
# yawpitchroll_angles = -180*yawpitchrolldecomposition(rmat)/math.pi
# yawpitchroll_angles[0,0] = (360-yawpitchroll_angles[0,0])%360 # change rotation sense if needed, comment this line otherwise
# yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]+90

while True:
    _, img = port.read()
    #img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    marker_corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    img = aruco.drawDetectedMarkers(img, marker_corners)
    if not marker_corners:
        cv2.imshow("Image", img)
        #cv2.imshow("Image", gray)

        print("No Found")
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        continue

    choosen_marker = None
    max_area = 0
    for (i, marker) in enumerate(marker_corners):
        area = cv2.contourArea(marker[0])
        if (area > max_area):
            max_area = area
            choosen_marker = marker

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, 50, mtx, dis)
    print("Find {} markers".format(len(marker_corners)))
    for (i, marker) in enumerate(marker_corners):
        img = aruco.drawAxis(img, mtx, dis, rvecs[i], tvecs[i], 40)
        rmat = cv2.Rodrigues(rvecs[i])[0]
        #print(rmat)
        # print(tvecs[i])
        # print("Index {}".format(i))
        # print(tvecs)
        #print(tvecs[i])
        P = np.concatenate((rmat, np.reshape(tvecs[i], (rmat.shape[0], 1))), axis=1)
        #print(P)
        #P = np.concatenate((rmat, tvecs[i][:, None]), axis=1)
        #P = np.hstack((rmat, tvecs[i]))
        euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
        euler_angles_degrees = 180 * euler_angles_radians / math.pi
        eul = euler_angles_radians
        yaw = 180 * eul[1, 0] / math.pi  # warn: singularity if camera is facing perfectly upward. Value 0 yaw is given by the Y-axis of the world frame.
        pitch = 180 * ((eul[0, 0] + math.pi / 2) * math.cos(eul[1, 0])) / math.pi
        roll = 180 * ((-(math.pi / 2) - eul[0, 0]) * math.sin(eul[1, 0]) + eul[2, 0]) / math.pi

        yaw = eul[1, 0]  # warn: singularity if camera is facing perfectly upward. Value 0 yaw is given by the Y-axis of the world frame.
        pitch =  eul[0, 0]
        roll = eul[2, 0]


        centerX = (marker[0][0][0] + marker[0][1][0] + marker[0][2][0] + marker[0][3][0]) / 4
        centerY = (marker[0][0][1] + marker[0][1][1] + marker[0][2][1] + marker[0][3][1]) / 4
        center = (int(centerX), int(centerY))
        contentP = "P:{}".format(np.round(pitch,0))
        contentY = "Y:{}".format( np.round(yaw,0))
        contentR = "R:{}".format(np.round(roll,0))

        image = cv2.putText(img, contentP, center, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,0,255), 2, cv2.LINE_AA)
        image = cv2.putText(img, contentY, (center[0], center[1]+30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,255,9), 2, cv2.LINE_AA)
        image = cv2.putText(img, contentR, (center[0], center[1]+60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255,0,0), 2, cv2.LINE_AA)


    cv2.imshow("Image", img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
port.release()
