import math

from Models.ArucoFinder import ArucoFinder
from Models.Calib import *
from PoseImp import aruco_processing, aruco_processing2
from Utils.Lib import *
from Models.PickPoint import *



class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# CONST
LENGTH_TOOL_ADDED = True

LENGTH_TOOL = - 30
ALPHA = 1 #DEGREE BETWEEN X AXIS (ROBOT) and Y AXIS (CAMERA)
BETA = 1 #DEGREE BETWEEN Y AXIS (ROBOT) and X AXIS (CAMERA)

pointA = Point(731.51, 8.45, 107.21)
pointB = Point(725.10, 14.26, 410.73)


def PickPointsSelect(ObjectsFound, LeftOrigin, RightOrigin, stereo, min_disp, num_disp, calib : Calib):
    (Left_Stereo_Map, Right_Stereo_Map, RL, RR, PL, PR) = calib.rectifyReturn(calib)

    (LeftNice, RightNice) = calib.returnRectifiedImage(LeftOrigin, RightOrigin, Left_Stereo_Map, Right_Stereo_Map)
    # cv2.imwrite("RectifiedLeft.jpg", LeftNice)
    # cv2.imwrite("RectifiedRight.jpg", RightNice)

    grayL = cv2.cvtColor(LeftNice, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(RightNice, cv2.COLOR_BGR2GRAY)
    # Calculate Disp
    disp_origin = stereo.compute(grayL, grayR)
    disp = ((disp_origin.astype(
        np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect
    #cv2.imwrite("DepthMap.jpg", disp)
    # cv2.imshow("Depth Map", disp)
    # Asuming that the object had marker detected, if it doesn't skip that object (Un-picked able)
    # for objectFound in ObjectsFound:
    ObjectProcess = []
    for object in ObjectsFound:
        ret, marker_middle_x, marker_middly_y, pitch, yaw, roll = aruco_processing2(mtx=calib.mtxL, dist=calib.disL, img=object.img_cropped)

        if (ret == False):
            continue
        marker_middle_x_inRealImg = marker_middle_x + object.x_pixel_coordinate
        marker_middle_y_inRealImg = marker_middly_y + object.y_pixel_coordinate

        (marker_x_central, marker_y_central) = (marker_middle_x_inRealImg, marker_middle_y_inRealImg)

        (x_rectified, y_rectified) = findCorresPixel(x=marker_x_central, y= marker_y_central, mtxL=calib.mtxL, disL=calib.disL, P=PL, R=RL) # Convert marker middle points to
        disp_val = getDispAtPoint(x=x_rectified, y=y_rectified, disp=disp)
        z = convertDisToZ(disp_val)

        depth_to_camera = FromZToDepth(z)
        (x,y) = calculateXY(instrictMatrix=calib.mtxL, depth=depth_to_camera, x_screen=x_rectified, y_screen=y_rectified)

        z_to_pick = z + LENGTH_TOOL if LENGTH_TOOL_ADDED else z

        (new_center_x, new_center_y) = calculateNewCenters(pointA.x, pointA.y, pointA.z,
                                                           pointB.x, pointB.y, pointB.z,
                                                           z)

        #print("New Center: {} {}".format(new_center_x, new_center_y))
        x_robot = toX_Robot(y, new_center_x) / math.cos(math.radians(BETA))
        y_robot = toY_Robot(x, new_center_y) / math.cos(math.radians(ALPHA))

        pick = PickPoint(x = x_robot, y= y_robot, z = z_to_pick, pitch=pitch, yaw=yaw, roll=roll)
        object.PickedPoint = pick # Set new Pick Point for object
        #print("Set Picked Point")
        ObjectProcess.append(object)
    return ObjectProcess


def PickPointsSelectWOMarker(ObjectsFound, LeftOrigin, RightOrigin, stereo, min_disp, num_disp, calib : Calib):
    (Left_Stereo_Map, Right_Stereo_Map, RL, RR, PL, PR) = calib.rectifyReturn(calib)

    (LeftNice, RightNice) = calib.returnRectifiedImage(LeftOrigin, RightOrigin, Left_Stereo_Map, Right_Stereo_Map)
    grayL = cv2.cvtColor(LeftNice, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(RightNice, cv2.COLOR_BGR2GRAY)
    # Calculate Disp
    disp_origin = stereo.compute(grayL, grayR)
    disp = ((disp_origin.astype(
        np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

    # Asuming that the object had marker detected, if it doesn't skip that object (Un-picked able)
    # for objectFound in ObjectsFound:

    for object in ObjectsFound:

        (marker_x_central, marker_y_central) = (object.middle_x, object.middle_y)

        (x_rectified, y_rectified) = findCorresPixel(x=marker_x_central, y= marker_y_central, mtxL=calib.mtxL, disL=calib.disL, P=PL, R=RL) # Convert marker middle points to
        disp_val = getDispAtPoint(x=x_rectified, y=y_rectified, disp=disp)
        z = convertDisToZ(disp_val)

        depth_to_camera = FromZToDepth(z)
        (x,y) = calculateXY(instrictMatrix=calib.mtxL, depth=depth_to_camera, x_screen=x_rectified, y_screen=y_rectified)

        z_to_pick = z + LENGTH_TOOL if LENGTH_TOOL_ADDED else z
        # depth = 281

        # ORIGINAL POINT A: (650.79, 29.33,452.17); B: (660.30,11.00,698.32)
        (new_center_x, new_center_y) = calculateNewCenters(pointA.x, pointA.y, pointA.z,
                                                           pointB.x, pointB.y, pointB.z,
                                                           z)

        x_robot = toX_Robot(y, new_center_x) / math.cos(math.radians(BETA))
        y_robot = toY_Robot(x, new_center_y) / math.cos(math.radians(ALPHA))

        pick = PickPoint(x = x_robot, y= y_robot, z = z_to_pick, pitch=180, yaw=0, roll=-180)
        object.PickedPoint = pick # Set new Pick Point for object

    return ObjectsFound



def getMarkerCorner(_corner):
    max_x = max(_corner[0][0][0], _corner[0][1][0], _corner[0][2][0], _corner[0][3][0])
    min_x = min(_corner[0][0][0], _corner[0][1][0], _corner[0][2][0], _corner[0][3][0])
    max_y = max(_corner[0][0][1], _corner[0][1][1], _corner[0][2][1], _corner[0][3][1])
    min_y = min(_corner[0][0][1], _corner[0][1][1], _corner[0][2][1], _corner[0][3][1])
    cen_x = (max_x + min_x) / 2; cen_y = (max_y + min_y) / 2; cen_x = int(cen_x); cen_y = int(cen_y)
    return (cen_x, cen_y)

