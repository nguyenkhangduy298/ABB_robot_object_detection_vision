import cv2
import numpy as np
import xlsxwriter

from Utils.Lib import CAMERA2_PORT, CAMERA1_PORT

CAMERA_PORT1 = CAMERA1_PORT
CAMERA_PORT2 = CAMERA2_PORT

# Package importation

# Filtering
kernel = np.ones((3, 3), np.uint8)

# *************************************************
# ***** Parameters for Distortion Calibration *****
# *************************************************
workbook = xlsxwriter.Workbook('Disparity.xlsx')
worksheet = workbook.add_worksheet()
row_exel = 0
col_exel = 0
data_workbook = ()


def coords_mouse_disp(event, x, y, flags, param):
    global row_exel
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print (x,y,disp[y,x],filteredImg[y,x])
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        # -2E-06x3 + 0.0006x2 - 0.059x + 2.6241
        # 145.47x^2 -301.02x + 200.34
        # Distance = -593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06
        # Distance = 145.47*average**(2) - 301.02*average + 200.34
        Distance = -204.27 * average ** (3) + 557.69 * average ** (2) - 565.101 * average ** (1) + 251.08
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance: ' + str(Distance) + ' m' + " Disparity: " + str(average))
        worksheet.write(row_exel, col_exel, average)
        row_exel += 1


def calculate_distance(x, y):
    # print (x,y,disp[y,x],filteredImg[y,x])
    average = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[y + u, x + v]
    average = average / 9
    # -2E-06x3 + 0.0006x2 - 0.059x + 2.6241
    # 145.47x^2 -301.02x + 200.34
    # Distance = -593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06
    Distance = 145.47 * average ** (2) - 301.02 * average + 200.34
    Distance = np.around(Distance * 0.01, decimals=2)
    # print('Distance: ' + str(Distance) + ' m' + " Disparity: " + str(average))
    return Distance


# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# PicForCalibration/Cam1/1.jpg
PATH_CALIBRATION_IMAGE_CAM1 = "PicForCalibration/Cam1/LeftCam"  # Cam1 is the left one
PATH_CALIBRATION_IMAGE_CAM2 = "PicForCalibration/Cam2/RightCam"  # Cam2 is the right one

# Call all saved images
for i in range(2300,
               2338):  # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t = str(i)
    ChessImaL = cv2.imread(PATH_CALIBRATION_IMAGE_CAM1 + t + '.jpg', 0)  # Left side
    ChessImaR = cv2.imread(PATH_CALIBRATION_IMAGE_CAM2 + t + '.jpg', 0)  # Right side
    cv2.imshow("Window", np.hstack((ChessImaL, ChessImaR)))
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (9, 6),
                                               None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (9, 6), None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)
    cv2.waitKey(1)
cv2.destroyAllWindows()
# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1], None, None)
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                            (wR, hR), 1, (wR, hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1], None, None)
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
                                                           mtxL, distL,
                                                           mtxR, distR,
                                                           ChessImaR.shape[::-1],
                                                           criteria_stereo,
                                                           flags)

print("Stereo Camera calibration value: " + str(retS))

# StereoRectify function
rectify_scale = 0  # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                  ChessImaR.shape[::-1], R, T,
                                                  rectify_scale,
                                                  (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped

# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1],
                                              cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)
# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# minDisparity – Minimum possible disparity value.
# numDisparities – Maximum disparity minus minimum disparity. This parameter must be divisible by 16.
# SADWindowSize – Matched block size. It must be an odd number >=1 .
# disp12MaxDiff – Maximum allowed difference (in integer pixel units) in the left-right disparity check.
# preFilterCap – Truncation value for the prefiltered image pixels.
# uniquenessRatio – Margin in percentage by which the best (minimum) computed cost function value should “win” the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
# speckleWindowSize – Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# speckleRange – Maximum disparity variation within each connected component
# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 0
num_disp = 130 - min_disp

stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp,
                               blockSize=window_size, uniquenessRatio=10,
                               speckleWindowSize=100, speckleRange=32, disp12MaxDiff=5,
                               P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time
stereoL = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time
# # WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
# *************************************
# ***** Prepare for the DNN *****
# *************************************
classes = "YoloFiles/yolo.names"
with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, distance):
    # print("ClassID" + str(class_id))
    label = str(classes[class_id]) + " " + str(distance) + "m"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


weight = "YoloFiles/yolov3_6000.weights"
config = "YoloFiles/yolov3penball.cfg"
scale = 0.00392
net = cv2.dnn.readNet(weight, config)

# *************************************
# ***** Starting the StereoVision *****
# *************************************

# Call the two cameras
CamR = cv2.VideoCapture(CAMERA_PORT2)  # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL = cv2.VideoCapture(CAMERA_PORT1)

while True:
    # Start Reading Camera images
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    # Rectify the images on rotation and alignement
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                          0)  # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    Left_nice_cp = Left_nice.copy()
    Right_nice_cp = Right_nice.copy()

    # Draw Red lines
    for line in range(0, int(Right_nice.shape[
                                 0] / 20)):  # Draw the Lines on the images Then numer of line is defines by the image Size/20
        Left_nice_cp[line * 20, :] = (0, 0, 255)
        Right_nice_cp[line * 20, :] = (0, 0, 255)

    # Convert from color(BGR) to gray
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
    # cv2.imshow("Disparity", disp)
    dispL = disp
    dispR = stereoR.compute(grayR, grayL)

    # USING FILTERED IMAGE
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)

    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp

    cv2.setMouseCallback("Disparity Map", coords_mouse_disp, filteredImg)

    flag_detect = False
    if (flag_detect == True):
        Width = Left_nice.shape[1]
        Height = Left_nice.shape[0]
        # cv2.imshow('frame',image)
        blob = cv2.dnn.blobFromImage(Left_nice, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        distances = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        # Thực hiện xác định bằng HOG và SVM
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)

                    # Distance = calculate_distance(filteredImg[center_x], filteredImg[center_y])
                    Distance = calculate_distance(center_x, center_y)

                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2

                    distances.append(Distance)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # middle_x = (x + w) / 2
            # middly_y = (y + h) / 2

            # Distance = calculate_distance(middle_x, middly_y)
            Distance = distances[i]

            draw_prediction(frameL, class_ids[i], confidences[i],
                            round(x), round(y), round(x + w), round(y + h), Distance)

    # Show the Undistorted images
    cv2.imshow('Both Images', np.hstack([Left_nice_cp, Right_nice_cp]))
    cv2.imshow('Normal', np.hstack([frameL, frameR]))
    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
workbook.close()
