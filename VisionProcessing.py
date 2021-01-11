import argparse
import random
import socket

import cv2

from Models.BoundingBoxWidget import BoundingBoxWidget
from Models.Calib import Calib
from Models.YoloObject import Yolo
from Utils.ObjectDetectionTools import ObjectDetection, PrepareClass
from Utils.ObjectProcessing  import *

ap = argparse.ArgumentParser()
# Cameras and Cameras Calibration parametesr
ap.add_argument('-l', '--LeftCam', required=True, help='path to port number from Left Camera') # Port to Left Camera
ap.add_argument('-r', '--RightCam', required=True, help='path to port number from Right Camera') # Port To Right Camera
ap.add_argument('-c', '--CaliPath', required=True, help='Path to Calibration file') # Calibration files for cameras
# Yolo Parameters
ap.add_argument('-cl', '--classYolo', required=True, help='Path to class name file of YoloV3') # Url to model class
ap.add_argument('-cfg', '--cfgYolo', required=True, help='Path to config file of YoloV3') # Url to YOLO Config file
ap.add_argument('-w', '--weightYolo', required=True, help='Path to weight file of YoloV3') # Url To Weight file
# Toggle between AUTO/MANUALLY to detect the object by YOLO or NOT
ap.add_argument('-d', '--DetectAuto', required=False, default= True, help='Path to config files')
ap.add_argument('-tcp', '--tcp', required=False, default= True, action='store_false', help='Integrate with Pose System' )
ap.add_argument('-auto', '--auto', required=False, default= True, action='store_false', help='Running the system auto' )


ap.add_argument('-params', '--Params', required=False, help='Path to config files') # Url To Params File
# TODO: Make a system to query the config file (PARAMS);
#  Params include the mode to be activated in the format: DEBUG = 1/0; DISPARITY = 1/0 -> Display to UI respective UI

args = ap.parse_args()
LeftCameraPort = int(args.LeftCam) # Get Port Number and assign it to correct cap object (Left camera)
RightCameraPort = int(args.RightCam) # Get Port Number and assign it to correct cap object (right camera)
url_calibPath = str(args.CaliPath) # Get Cali Path
calib = Calib(url_calibPath) # Create calib object from calib path
(Left_Stereo_Map, Right_Stereo_Map, _,_,_,_) = Calib.rectifyReturn(calib) # Create Left and Right Stereo map

# Create Yolo Object, if not; ignore this line
YoloObject = Yolo(configUrl=str(args.cfgYolo), classUrl=str(args.classYolo), weightUrl=str(args.weightYolo))

def FindObject(frame, mode="AUTO"):
    if (mode == "AUTO"):
        return ObjectDetection(frame, weight=YoloObject.yoloWeight, yoloCfg=YoloObject.yoloConfigUrl,
                                             confident_threshold=0.05)
    elif (mode == "MANUAL"):
        boundingbox_widget = BoundingBoxWidget(frame)
        a = boundingbox_widget.show_image()
        while True:
            cv2.imshow('image', boundingbox_widget.show_image())
            cv2.putText(a, "Hit Q to Confirm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_4)
            key = cv2.waitKey(1)

            # Close program with keyboard 'q'
            if key == ord('q'):
                cv2.destroyWindow('image')
                break
        return boundingbox_widget.ObjectList

# This function take the array of objects. Set the PickPoint for each Objects
if __name__ == '__main__':
    #setupSystem() # Setup Camera and Calib & Yolo Object

    camL = cv2.VideoCapture(LeftCameraPort); camR = cv2.VideoCapture(RightCameraPort) # Assign and register cam
    # Create StereoSGBM and prepare all parameters TODO: Add this to setup
    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100, speckleRange=32, disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2
                                   # ,mode=cv2.STEREO_SGBM_MODE_HH4
                                   )

    classes = PrepareClass(class_path=YoloObject.yoloClass)
    #COLORS = np.random.uniform(0, 100, size=(len(classes), 3))
    #COLORS = np

    COLORS = [
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ]


    print(COLORS)
    if (args.tcp):
    #Setup TCP/IP:
        TCP_IP = '192.168.125.1'
        TCP_PORT = 1025
        BUFFER_SIZE = 1024

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        print("Connected.")

    AUTO_SYSTEM = args.auto

    while True:
        _, frameLeft = camL.read(); _, frameRight = camR.read() # Frame Left and Right captured fromt he images; no processing yet


        # If detect auto -> Using Yolo to find the object, if not, select the areas by hand
        cpLeft = frameLeft.copy()
        cv2.imshow("Left and Right", np.hstack((frameLeft, frameRight)))
        if ((cv2.waitKey(1) & 0xFF == ord('e')) or AUTO_SYSTEM):

            ObjectsFound = FindObject(frameLeft, mode="AUTO") # Default: AUto mode
            if (len(ObjectsFound) < 0):
                break # Break the loop if no object found
            #Write detected object
            # for object in ObjectsFound:
            #     cv2.imwrite("Object" + str(random.randint(0,9000))+".jpg", object.img_cropped)


            ObjectsWithPickPoint = PickPointsSelect(ObjectsFound= ObjectsFound, LeftOrigin=frameLeft, RightOrigin=frameRight, stereo=stereo,
                                                    min_disp=min_disp, num_disp=num_disp, calib=calib)
            # Select most pickable point to pick first
            print("Found " + str(len(ObjectsWithPickPoint)) + " object(s)" )


            ObjectsWithPickPoint.sort(key = lambda ob: ob.PickedPoint.closeness)

            #Draw the image:
            for stuff in ObjectsWithPickPoint:
                print("=============")
                stuff.giveInfor()
                stuff.PickedPoint.info()
                print("=============")
                draw_prediction(frameLeft,
                                class_id=stuff.Class,
                                confidence=stuff.confidence_level,
                                x=stuff.x_pixel_coordinate,
                                y=stuff.y_pixel_coordinate,
                                x_plus_w=stuff.x_pixel_coordinate + stuff.width,
                                y_plus_h=stuff.y_pixel_coordinate + stuff.height,
                                classes=classes, COLORS=COLORS, distance=stuff.PickedPoint.z,
                                coor1=(stuff.PickedPoint.x, stuff.PickedPoint.y, stuff.PickedPoint.z))

            for stuff in ObjectsFound:
                print("=============")
                stuff.giveInfor()
                stuff.PickedPoint.info()
                print("=============")
                draw_prediction(cpLeft,
                                class_id=stuff.Class,
                                confidence=stuff.confidence_level,
                                x=stuff.x_pixel_coordinate,
                                y=stuff.y_pixel_coordinate,
                                x_plus_w=stuff.x_pixel_coordinate + stuff.width,
                                y_plus_h=stuff.y_pixel_coordinate + stuff.height,
                                classes=classes, COLORS=COLORS, distance=stuff.PickedPoint.z,
                                coor1=(stuff.PickedPoint.x, stuff.PickedPoint.y, stuff.PickedPoint.z))
            while True:
                cv2.imshow("Result", frameLeft)
                cv2.imshow("Object seen", cpLeft)
                cv2.imwrite("Detected.jpg", cpLeft)
                cv2.waitKey(1)

                if (AUTO_SYSTEM):
                    break
                elif cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow("Result")
                    break


            # Result: SortedList -> The closest one is the first index, the last one is the most further away
            if (args.tcp):
                print("BEGIN TCP")
                for object in ObjectsWithPickPoint:
                    (x_robot, y_robot, z_to_pick) = (np.round(object.PickedPoint.x,2),
                                                     np.round(object.PickedPoint.y,2),
                                                     np.round(object.PickedPoint.z,2))
                    (pitch, yaw, roll) = (np.round(object.PickedPoint.pitch, 2),
                                          np.round(object.PickedPoint.yaw, 2),
                                          np.round(object.PickedPoint.roll, 2) )
                    if (z_to_pick < 100):
                        continue
                    robot_coordinate = [x_robot, y_robot, z_to_pick]
                    # robot_rot = [180, 0, 180]
                    robot_rot = [pitch, yaw, roll]
                    command = str(object.Class)+ str(robot_coordinate) + str(robot_rot)
                    print("Robot Coordinate Results:")
                    print("Class: " + str(object.Class))
                    print(robot_coordinate)
                    print(robot_rot)
                    print("=========")

                    receive = ""
                    print("Sutck here")
                    print("Initial Receive:",receive)
                    while receive != "0xFC":
                        print(receive)
                        receive = s.recv(1024).decode('utf-8')
                        print("Second" + str(receive))

                        if (receive == "0xFD"):
                            print("Object delivered/n")
                        elif (receive == "0xFC"):
                            print("Stand by/n")
                            break
                        elif (receive == "0xFF" or receive == "0xFF0xFF"):
                            print("hello")
                            data = command
                            print("Object location request/n")
                            s.send(data.encode())
                        elif (receive == "0xFB"):
                            print("Moving../n")
                        elif (receive == "0xFA"):
                            print("server closed/n")
                            break
                        elif (receive == ""):
                            pass;
                        else:
                            print(".", type(receive))
                    print("Finished Sending")
        continue


        # Calculate X,Y,Z


    camL.release()
    camR.release()
    cv2.destroyAllWindows()