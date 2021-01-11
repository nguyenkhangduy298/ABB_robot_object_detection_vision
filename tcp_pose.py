# import the necessary packages

import cv2.aruco as aruco
import time
import cv2

import numpy as np
import cv2 as cv
import glob
import math
import socket

def caliCam():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*13,3), np.float32)
    objp[:,:2] = np.mgrid[0:13,0:9].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('WebCam/*.png')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (13,9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (13,9), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print(ret,mtx,dist,rvecs,tvecs)
    np.savetxt('mtx.txt', mtx, delimiter =', ')
    np.savetxt('dist.txt', dist, delimiter =', ')

    #Projection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    return ret, mtx, dist, rvecs, tvecs

#Pose Estimation
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    print("Original Draw: ",corner)
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,0,255), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    return img

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisCube = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

XYZ = [524.70, -85.31, 292.21]

#302.21
#REAL VAT 297.11 - Safe Test 317.11
# 126.01 cai ban
pose_estimation = [-120, 0, 180]
# Prepare = [663.95, 261.767, 737.84] #Already Set Formulat to PREPARE position directly in Robot Studio
#NOTE for robot studio: Replace the prepare Z with Z have to minus the height of the tool in different angles (reflection height of the tool))

TCP = 0

def main():
    if(TCP!=0):
        # TCP IP Connecting
        TCP_IP = "192.168.125.1"
        TCP_PORT = 1025
        BUFFER_SIZE = 1024
        # ______________Input the object locating algorithm here_________________#

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        print("Connected.")
        # ============END TCP IP CONNECTION===========

    ret, mtx, dist, rvecs, tvecs = caliCam()
    # USAGE
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    cv2.namedWindow("Vision system")

    from math import atan2, pi

    def get_angle(qrcode):
        poly = qrcode.polygon
        angle = atan2(poly[1].y - poly[0].y, poly[1].x - poly[0].x)
        return angle - pi / 2

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    # loop over the frames from the video stream
    img_counter = 0
    while True:
        # grab the frame from the threaded video stream and resize it to
        # have a maximum width of 400 pixels
        # frame = vs.read()
        # frame = imutils.resize(frame, width=400)

        ret, img = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        marker_corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=arucoParameters)
        img = aruco.drawDetectedMarkers(img, marker_corners)

        # print("Aruco Detected Markers:",corner)
        print("Corners: ",marker_corners)

        if not marker_corners:
            cv2.imshow("test", img)
            cv2.waitKey(1)
            print("No Marker")
        else:
            for i in range(0, len(marker_corners)):
                corner = np.array(marker_corners[i][0])
                print("First", corner)

                # Corner Sorting:
                corner_arr = []  # Right Bottom  -> Left Bottom -> Right Top -> Left Top
                code_list_y = [corner[0][1], corner[1][1], corner[2][1],
                                  corner[3][1]]
                print("Y Before Sort: ", code_list_y)
                code_list_y.sort(reverse=True)
                print("Y After Sort: ", code_list_y)
                for i in range(0, 4):
                    check = 0
                    for j in range(0, 4):
                        if check==0:
                            if code_list_y[i] == corner[j][1]:
                                corner_arr.append(j)
                                check=1
                print("Corner Array:",corner_arr)
                # Bottom 2
                if corner[corner_arr[0]][0] < corner[corner_arr[1]][0]:
                    # Switch place so Right Bottom -> Left Bottom
                    corner_arr[0], corner_arr[1] = corner_arr[1], corner_arr[0]
                    print("Switch Bottom")
                # Top 2
                if corner[corner_arr[2]][0] < corner[corner_arr[3]][0]:
                    # Switch place so Right Top -> Left Top
                    corner_arr[2], corner_arr[3] = corner_arr[3], corner_arr[2]
                    print("Switch Top")
                print("Final Sort: ", corner_arr)

                corners_convert = np.array(
                    [[corner[corner_arr[0]][0], corner[corner_arr[0]][1]],
                     [corner[corner_arr[1]][0], corner[corner_arr[1]][1]],
                     [corner[corner_arr[2]][0], corner[corner_arr[2]][1]],
                     [corner[corner_arr[3]][0], corner[corner_arr[3]][1]]])

                # corners_convert = np.array([corner[2],corner[3],corner[1],corner[0]])

                print(corners_convert)
                axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
                objp = np.zeros((2 * 2, 3), np.float32)
                objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
                # print(objp)
                ret, rvecs, tvecs = cv.solvePnP(objp, corners_convert, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

                # Rendering model on video frame
                corners_convert = corners_convert.astype(int)
                imgpts = imgpts.astype(int)
                # img = draw(img,corners2,imgpts)
                img = draw(img, corners_convert, imgpts)
                # img = drawCube(img, corners2_polygon, imgpts)

                #Get Degree Up
                rmat = cv2.Rodrigues(rvecs)[0]
                proj_matrix = np.hstack((rmat, tvecs))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
                print("P/R/Y : ", pitch * 57.2957795, roll * 57.2957795, yaw * 57.2957795)
                pitch ,yaw = yaw , pitch  # Switch and convert Pitch and Yaw

                cv2.putText(img, "Yaw: %.2f" % yaw, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                cv2.putText(img, "Pitch: %.2f" % pitch, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                cv2.putText(img, "Roll: %.2f" % roll, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                print("Real: ",yaw,pitch,yaw)

                #Handle PYR robot inverse vector from the object normal vector
                if abs(pitch) < 9: # If lower than X degree than set to directly straight angle
                    pitch = 0
                if abs(yaw) <9: # If lower than X degree than set to directly straight angle
                    yaw = 0
                if pitch <0: # Could be reverse >0 and <0 - Needs Testing With Robot
                    pitch = 180 - abs(pitch) #Object Negative then robot Positive
                else:
                    pitch = (180 - abs(pitch))*-1 #Object Positive then robot Negative
                if yaw <0: # Could be reverse >0 and <0 - Needs Testing With Robot
                    yaw = abs(yaw)
                else:
                    yaw = abs(yaw)*-1

                pose=[pitch,yaw,180]

                # ______________TCP/IP Connection_________________#
                # robot_coordinate = sendDuyCalculateXY(x_coordinate,y_coordinate,pixel_width)
                robot_coordinate = XYZ
                print("Robot Coordinate Results:")
                print(robot_coordinate)
                data = str(robot_coordinate) + str(pose)
                print("Data: ", data)
                if (TCP!=0):
                    s.send(data.encode())

            # Printing Pose Result
            img_name = "aruco_pose_result_{}.png".format(img_counter)
            #cv2.imwrite(img_name, img)
            print("{} written!".format(img_name))
            img_counter += 1
            cv2.imshow("test", img)
            cv2.waitKey(1)


    # close the output CSV file do a bit of cleanup
    print("[INFO] cleaning up...")
    # vs.stop()
    cam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()