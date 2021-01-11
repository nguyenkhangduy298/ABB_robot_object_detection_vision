import cv2
import numpy as np

def getDispAtPoint(x, y, disp):
    x = int(x)
    y = int(y)
    average = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[y + u, x + v]
    average = disp[y, x]
    return average


def findCorresPixelCoordinateFromObject(object, mtxL, disL, R, P):
    xy = cv2.undistortPoints((object.x_pixel_coordinate, object.y_pixel_coordinate),
                             cameraMatrix=mtxL, distCoeffs=disL, R=R, P=P)
    return xy[0][0]

def findCorresPixel(x,y, mtxL, disL, R, P):
    xy = cv2.undistortPoints((x,y), cameraMatrix=mtxL, distCoeffs=disL, R=R, P=P)
    return xy[0][0]

def Unproject(point, Z, intrinsic_matrix, distortion, P, R):
    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]
    c_x = intrinsic_matrix[0, 2]
    c_y = intrinsic_matrix[1, 2]
    # Step 1. Undistort.
    points_undistorted = cv2.undistortPoints(point, cameraMatrix=intrinsic_matrix, distCoeffs=distortion,
                                             P=P, R=R)
    print(points_undistorted)
    # Step 2. Reproject.
    x = (points_undistorted[0][0] - c_x) / f_x * Z
    print(x)
    y = (points_undistorted[0][1] - c_y) / f_y * Z
    return (x, y, Z)


def drawOnFrame(ObjectList, frame, classes, COLORS):
    for object in ObjectList:
        draw_prediction(frame,
                        class_id=object.Class,
                        confidence=object.confidence_level,
                        x=object.x_pixel_coordinate,
                        y=object.y_pixel_coordinate,
                        x_plus_w=object.x_pixel_coordinate + object.width,
                        y_plus_h=object.y_pixel_coordinate + object.height,
                        classes=classes, COLORS=COLORS, distance=0, angle=0.0,
                        coor1=(0, 0, 0.0))


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, distance, classes, COLORS, coor1):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), tuple(color), 2)
    values = str(round(coor1[0], 2)) + "," + str(round(coor1[1], 2))+ "," + str(round(coor1[2], 2))
    cv2.putText(img, label, (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #cv2.putText(img, values, (int(x - 10), int(y +50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Apply regression trendline to deduce distance from disparity (average)
def convertDisToZ(average, useOriginalDisparity=False):
    # return -215.5276*(x**3) + 602.2604*(x**2) - 618.2860*(x) + 272.5236
    if (useOriginalDisparity == False):
        return  3296.4*(average**3) - 7379.1*(average**2) + 5935.2*average - 973.64
        # x3 - 7379.1
        # x2 + 5935.2
        # x - 973.64

        return -2352*(average**2) + 3519.8*average - 638.69
        return -2352*(average**2) + 4173*average - 788
        return -2977*(average**2) + 4173*average - 788
        return -2251.2*(average**2) + 3425.1*average - 629.71
    return -204.27 * average ** (3) + 557.69 * average ** (2) - 565.101 * average ** (1) + 251.08



def calculateXY(instrictMatrix, depth, x_screen, y_screen):
    # print(instrictMatrix)
    f_x = instrictMatrix[0][0]
    # print(f_x)
    f_y = instrictMatrix[1][1]
    # print(f_y)
    c_x = instrictMatrix[0][2]
    # print(c_x)
    c_y = instrictMatrix[1][2]
    # print(c_y)
    x_world = (x_screen - c_x) * depth / f_x
    y_world = (y_screen - c_y) * depth / f_y
    return (np.round(x_world, 2), np.round(y_world, 2))



def toX_Robot(x, const):
    return x + const

def toY_Robot(y, const):
    return y + const


def parseMatrix(instrictMatrix):
    # print(instrictMatrix)
    f_x = instrictMatrix[0][0]
    # print(f_x)
    f_y = instrictMatrix[1][1]
    # print(f_y)
    c_x = instrictMatrix[0][2]
    # print(c_x)
    c_y = instrictMatrix[1][2]
    return (f_x, f_y, c_x, c_y)

# Calculate Distance
def Duycalculate_distance(pixel_width):
    return (3 * 640) / (2 * pixel_width * 0.4383561643835616)


def DuyCalculateXY(x, y, pixel_width):
    x_coordinate = abs(320 - x)
    y_coordinate = abs(240 - y)

    # CONST_X = 1
    # CONST_Y = 1
    if (x > 320):
        CONST_X = 1
    else:
        CONST_X = -1
    if (y > 240):
        CONST_Y = 1
    else:
        CONST_Y = -1
    real_z = Duycalculate_distance(pixel_width)
    real_x = CONST_X * (real_z * x_coordinate) / 670
    real_y = CONST_Y * (real_z * y_coordinate) / 670
    return (real_x, real_y)

def FromZToDepth(z_robot):
    return (1095 - z_robot)

def calculateNewCenters(a,b,c,a1,b1,c1,Z):
    #deltaC = c1 - c1
    t = (Z-c)/(c1-c)
    x = a + (a1-a)*t
    y = b + (b1-b)*t
    return (x,y)

