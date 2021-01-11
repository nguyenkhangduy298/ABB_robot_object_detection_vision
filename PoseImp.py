import math
import random

import cv2
import cv2.aruco as aruco
import numpy as np

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)



def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
    return img

def sort_corner(corner):
    corner = corner[0]

    corner_arr = []  # Right Bottom  -> Left Bottom -> Right Top -> Left Top
    code_list_y = [corner[0][1], corner[1][1], corner[2][1],
                   corner[3][1]]
    #print("Y Before Sort: ", code_list_y)
    code_list_y.sort(reverse=True)
    #print("Y After Sort: ", code_list_y)
    for i in range(0, 4):
        check = 0
        for j in range(0, 4):
            if check == 0:
                if code_list_y[i] == corner[j][1]:
                    corner_arr.append(j)
                    check = 1
    #print("Corner Array:", corner_arr)
    # Bottom 2
    if corner[corner_arr[0]][0] < corner[corner_arr[1]][0]:
        # Switch place so Right Bottom -> Left Bottom
        corner_arr[0], corner_arr[1] = corner_arr[1], corner_arr[0]
        #print("Switch Bottom")
    # Top 2
    if corner[corner_arr[2]][0] < corner[corner_arr[3]][0]:
        # Switch place so Right Top -> Left Top
        corner_arr[2], corner_arr[3] = corner_arr[3], corner_arr[2]
        #print("Switch Top")
    #print("Final Sort: ", corner_arr)

    corners_convert = np.array(
        [[corner[corner_arr[0]][0], corner[corner_arr[0]][1]],  # Right Bottom
         [corner[corner_arr[1]][0], corner[corner_arr[1]][1]],  # Left Bottom
         [corner[corner_arr[2]][0], corner[corner_arr[2]][1]],  # Right Top
         [corner[corner_arr[3]][0], corner[corner_arr[3]][1]]])  # Left Top

    # corners_convert = np.array([corner[2],corner[3],corner[1],corner[0]])
    #print(corners_convert)
    return corners_convert

def aruco_processing(mtx, dist, img, save=False):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()

    arucoParameters.adaptiveThreshWinSizeMin = 3
    arucoParameters.adaptiveThreshWinSizeStep = 2
    arucoParameters.adaptiveThreshWinSizeMax = 50
    arucoParameters.minMarkerPerimeterRate = 0.01
    arucoParameters.maxMarkerPerimeterRate = 4.0
    arucoParameters.markerBorderBits = 1

    img = cv2.detailEnhance(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    marker_corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)

    # marker_corners += rejectedImgPoints
    img = aruco.drawDetectedMarkers(img, marker_corners)

    # cv2.imwrite("Object With Marker" + str(random.randint(0, 9000)) + ".jpg", img)

    if not marker_corners:
        # Do something next
        #print("No marker found")
        return (False, None, None, None, None, None)

    choosen_marker = None
    max_area = 0
    for (i, marker) in enumerate(marker_corners):
        area = cv2.contourArea(marker[0])
        if (area > max_area):
            max_area = area
            choosen_marker = marker

    corners_convert = sort_corner(choosen_marker)

    # Calculate the middle point

    # Left-top  [corner[corner_arr[3]][0], corner[corner_arr[3]][1]]] ;    [[corner[corner_arr[0]][0], corner[corner_arr[0]][1]], # Right Bottom
    left_top_x = corners_convert[3][0]
    left_top_y = corners_convert[3][1]

    right_bottom_x = corners_convert[0][0]
    right_bottom_y = corners_convert[0][1]

    middle_x = (left_top_x + right_bottom_x) / 2
    middle_y = (left_top_y + right_bottom_y) / 2

    # Calculate the pose
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
    objp = np.zeros((2 * 2, 3), np.float32)
    objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners_convert, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    # Rendering model on video frame
    corners_convert = corners_convert.astype(int)
    imgpts = imgpts.astype(int)

    img = draw(img, corners_convert, imgpts)

    #img = aruco.drawAxis(img, mtx, dist, rvecs, tvecs, 20)


    # Get Degree Up
    rmat = cv2.Rodrigues(rvecs)[0]
    proj_matrix = np.hstack((rmat, tvecs))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    #print("P/R/Y : ", pitch * 57.2957795, roll * 57.2957795, yaw * 57.2957795)
    pitch, yaw = yaw * 57.2957795, pitch * 57.2957795  # Switch and convert Pitch and Yaw
    # cv2.putText(img, "Yaw:  %.2f" % yaw, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    # cv2.putText(img, "Pitch: %.2f" % pitch, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    # cv2.putText(img, "Roll: 180", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

    #print("Real: ", yaw, pitch, yaw)
    randomNumber = random.randint(1,1000)

    cv2.imwrite("Ill"+str(randomNumber)+".jpg", img)
    threshold = 9
    # Handle PYR robot inverse vector from the object normal vector
    if abs(pitch) < threshold:  # If lower than X degree than set to directly straight angle
        pitch = 0
    if abs(yaw) < threshold:  # If lower than X degree than set to directly straight angle
        yaw = 0

    if pitch < 0:  # Could be reverse >0 and <0 - Needs Testing With Robot
        pitch = 180 - abs(pitch)  # Object Negative then robot Positive
    else:
        pitch = (180 - abs(pitch)) * -1  # Object Positive then robot Negative
    if yaw < 0:  # Could be reverse >0 and <0 - Needs Testing With Robot
        yaw = abs(yaw)
    else:
        yaw = abs(yaw) * -1
    roll = 180
    # pose = [pitch, yaw, roll]
    return (True, middle_x, middle_y, pitch, yaw, roll)



def rot_matrix_to_euler(R):
    y_rot = math.asin(R[2][0])
    x_rot = math.acos(R[2][2]/math.cos(y_rot))
    z_rot = math.acos(R[0][0]/math.cos(y_rot))
    y_rot_angle = y_rot *(180/math.pi)
    x_rot_angle = x_rot *(180/math.pi)
    z_rot_angle = z_rot *(180/math.pi)

    yaw = 180 - x_rot_angle
    pitch = y_rot_angle
    roll = z_rot_angle

    return yaw,pitch,roll

def aruco_processing2(mtx, dist, img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()

    #img = cv2.detailEnhance(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    marker_corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)

    # marker_corners += rejectedImgPoints
    img = aruco.drawDetectedMarkers(img, marker_corners)

    # cv2.imwrite("Object With Marker" + str(random.randint(0, 9000)) + ".jpg", img)

    if not marker_corners:
        # Do something next
        # print("No marker found")
        return (False, None, None, None, None, None)

    choosen_marker = None
    max_area = 0
    for (i, marker) in enumerate(marker_corners):
        area = cv2.contourArea(marker[0])
        if (area > max_area):
            max_area = area
            choosen_marker = marker

    centerX = (choosen_marker[0][0][0] + choosen_marker[0][1][0] +choosen_marker[0][2][0] + choosen_marker[0][3][0]) / 4
    centerY = (choosen_marker[0][0][1] + choosen_marker[0][1][1] + choosen_marker[0][2][1] + choosen_marker[0][3][1]) / 4
    center = (int(centerX), int(centerY))

    middle_x = center[0]
    middle_y = center[1]

    #Pose Calculate
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(choosen_marker, 50, mtx, dist)
    choosen_rvec = rvecs[0]
    choosen_tvec = tvecs[0]

    img = aruco.drawAxis(img, mtx, dist, choosen_rvec, choosen_tvec, 40)

    rmat = cv2.Rodrigues(choosen_rvec)[0]
    P = np.concatenate((rmat, np.reshape(choosen_tvec, (rmat.shape[0], 1))), axis=1)

    euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
    euler_angles_degrees = 180 * euler_angles_radians / math.pi
    eul = euler_angles_radians
    # yaw = 180 * eul[1, 0] / math.pi  # warn: singularity if camera is facing perfectly upward. Value 0 yaw is given by the Y-axis of the world frame.
    # pitch = 180 * ((eul[0, 0] + math.pi / 2) * math.cos(eul[1, 0])) / math.pi
    # roll = 180 * ((-(math.pi / 2) - eul[0, 0]) * math.sin(eul[1, 0]) + eul[2, 0]) / math.pi

    yaw = eul[ 1, 0]  # warn: singularity if camera is facing perfectly upward. Value 0 yaw is given by the Y-axis of the world frame.
    pitch = eul[0, 0]
    roll = eul[2, 0]

    # cv2.imshow("Pose Select", img)
    # rmat = cv2.Rodrigues(rvecs)[0]
    # yaw, pitch, roll = rot_matrix_to_euler(rmat)
    # (yaw, pitch, roll) = (yaw*-1, pitch-180, roll)
    return (True, middle_x, middle_y, pitch, yaw, roll)

# if __name__ == "__main__":
#     # mtx, dist, img = 0
#     # aruco_processing(mtx, dist, img)
