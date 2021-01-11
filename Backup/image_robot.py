# coding=utf-8

import argparse
import datetime
import math
import time
import tkinter as tk

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment

ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

workbook = Workbook()
worksheet = workbook.active
# creating header row for excel file
worksheet["A1"] = "X-Coordinate"
worksheet["A1"].alignment = Alignment(horizontal="center")
worksheet["B1"] = "Y-Coordinate"
worksheet["B1"].alignment = Alignment(horizontal="center")
worksheet["C1"] = "Z-Coordinate"
worksheet["C1"].alignment = Alignment(horizontal="center")
worksheet.column_dimensions['A'].width = 15
worksheet.column_dimensions['B'].width = 15
worksheet.column_dimensions['C'].width = 15
# application
root = tk.Tk()
root.title("Coordinate System")
L1 = tk.Label(root, text="X-coordinate", fg="red", font="Timer").grid(row=0, column=0)
E1 = tk.Entry(root, bd=5)
E1.grid(row=0, column=1)
L2 = tk.Label(root, text="Y-coordinate", fg="red", font="Timer").grid(row=1, column=0)
E2 = tk.Entry(root, bd=5)
E2.grid(row=1, column=1)
L3 = tk.Label(root, text="Z-coordinate", fg="red", font="Timer").grid(row=2, column=0)
E3 = tk.Entry(root, bd=5)
E3.grid(row=2, column=1)


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    print("ClassID" + str(class_id))
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def calculate_distance(pixel_width):
    return (3 * 640) / (2 * pixel_width * 0.4383561643835616)
    # return 441.6/(0.20072727272727273 * pixel_width) # (focal *  realWidth * 640)/(pixelWidth * distance)


# return (2 * 640)/ (2 * pixel_width * 20)

def calculate_distance_focal(KNOWN_WIDTH, pixel_width):
    return (KNOWN_WIDTH * 670) / pixel_width


def calculate_FOV(knownpixel_width):
    knownWidth = 3
    distance = 40
    focal_calculate = 441.6
    div = pixel_width * 40
    fov = focal_calculate / div
    return fov


cam = cv2.VideoCapture(1)
cam.set(3, 640)
cam.set(4, 480)

cv2.namedWindow("Vision system")

img_counter = 0
x_coordinate = 0
y_coordinate = 0

while True:
    ret, image = cam.read()

    if not ret:
        print("failed to grab frame")
        break

    # img = np.zeros((512, 512, 3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    cv2.line(image, (320, 0), (320, 480), (255, 0, 0), 1)
    cv2.line(image, (0, 240), (640, 240), (255, 0, 0), 1)
    cv2.imshow("Vision System", image)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        pixel_width = 0

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # HOG and SVM
        start = time.time()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    x_coordinate = center_x
                    y_coordinate = center_y
                    w = int(detection[2] * Width)
                    pixel_width = w
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
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
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        file_name = str(datetime.datetime.now()) + '.jpg';

        # cv2.waitKey(0)
        path = '/home/sonng9800/CapstoneProject/yolo_beginner-master/result/'

        cv2.putText(image, "%.2fcm" % calculate_distance_focal(3, pixel_width), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (0, 255, 0), 3)
        result_file = "result_processed_0.jpg"
        flag_return = cv2.imwrite(result_file, image)
        if (flag_return):
            print("Saved Ok")
        else:
            print("Save Error")

        print("==============Testing Results====================")

        # return (3 * 640)/ (2 * pixel_width * 20)

        # load the furst image that contains an object that is KNOWN TO BE 2 feet
        # from our camera, then find the paper marker in the image, and initialize
        # the focal length
        # image = cv2.imread(args.image)
        KNOWN_DISTANCE = 90.0
        KNOWN_WIDTH = 3.0
        # focalLength = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
        # print("Focal Length: ",focalLength)

        # image = cv2.imread("opencv_yolo_0.png")

        # marker = find_marker(image)
        # print(pixel_width)
        # print(calculate_distance(pixel_width))

        distance_focal = calculate_distance_focal(KNOWN_WIDTH, pixel_width)
        print("Distance focal:", distance_focal)

        # loop over the images
        # for imagePath in sorted(paths.list_images("images")):
        # load the image, find the marker in the image, then compute the
        # distance to the marker from the camera
        # image = cv2.imread(imagePath)
        # marker = find_marker(image)
        print("FOV: ", calculate_FOV(pixel_width))
        print("Distance to object: ", calculate_distance(pixel_width))

        # cm = distance_to_camera(KNOWN_WIDTH, focalLength, pixel_width)
        #
        # # draw a bounding box around the image and display it
        # box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        # box = np.int0(box)
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        x_coordinate = abs(320 - x_coordinate)
        y_coordinate = abs(240 - y_coordinate)

        # Focal = (PixelW x  Distance) / Width
        # => coordinate = (Distance * PixelCoordinate) / Focal
        real_x = (calculate_distance(pixel_width) * x_coordinate) / 670
        real_y = (calculate_distance(pixel_width) * y_coordinate) / 670
        real_xy = math.pow(real_x, 2) + math.pow(real_y, 2)
        real_z = math.pow(calculate_distance(pixel_width), 2) - real_xy
        # real_z = 0 #math.sqrt(real_z)
        real_z = 80 - calculate_distance_focal(3, pixel_width)
        print(real_y, real_x, real_z)

        xyz30 = [400, 0, 20]  # drop off point Robot Studio
        xyz40 = [400, 0, 40]
        # this is the mod file for robot studio
        file = open("coordinates.mod", "w")
        L10 = "    CONST robtarget Target_10:=[[" + str(round(real_y, 2) * 10 + 300) + "," + str(
            round(real_x, 2) * 10) + "," + str(round(real_z,
                                                     2) * 10 + 20) + "],[0,-0.707106781,0.707106781,0],[0,-2,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n"
        L20 = "    CONST robtarget Target_20:=[[" + str(round(real_y, 2) * 10 + 300) + "," + str(
            round(real_x, 2) * 10) + "," + str(round(real_z,
                                                     2) * 10 + 40) + "],[0,-0.707106781,0.707106781,0],[0,-2,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n"
        L30 = "    CONST robtarget Target_30:=[[" + str(xyz30[0]) + "," + str(xyz30[1]) + "," + str(
            xyz30[2]) + "],[0,-0.707106781,0.707106781,0],[0,-2,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n"
        L40 = "    CONST robtarget Target_40:=[[" + str(xyz40[0]) + "," + str(xyz40[1]) + "," + str(
            xyz40[2]) + "],[0,-0.707106781,0.707106781,0],[0,-2,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];\n"
        L = ["MODULE Path1\n", L10, L20, L30, L40,
             "\tPROC Path_20()\n        Reset gripper;\n        MoveJ Target_30,v200,fine,Servo\WObj:=wobj0;\n        WaitTime\InPos,0.5;\n        Set gripper;\n        WaitTime\InPos,1;\n        MoveJ Target_10,v200,fine,Servo\WObj:=wobj0;\n        WaitTime\InPos,1;\n        Reset gripper;\n        WaitTime\InPos,0.5;\n        Set gripper;\n        MoveJ Target_20,v200,fine,Servo\WObj:=wobj0;\n        MoveJ Target_40,v200,fine,Servo\WObj:=wobj0;\n        MoveJ Target_30,v200,fine,Servo\WObj:=wobj0;\n        WaitTime\InPos,1;\n    ENDPROC\nENDMODULE"]
        file.writelines(L)
        file.close()

        # this is the excel file
        row = ((), (real_y, real_x, real_z))
        for x in row:
            worksheet.append(x)
        workbook.save(filename="Coordinates.xlsx")

        img_counter += 1

cam.release()
cv2.destroyAllWindows()
# root.mainloop()
