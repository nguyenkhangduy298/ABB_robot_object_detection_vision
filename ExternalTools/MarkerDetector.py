import time

import numpy as np
import cv2
import cv2.aruco as aruco
cap = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0
areas = []


def findCentral(points):
    max = max()


while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)


    frame1 = frame.copy()
    # frame2 = frame.copy()
    frame1 = aruco.drawDetectedMarkers(frame1, corners)
    # for _corner in corners:
    #     area = cv2.contourArea(_corner)
    #     print(_corner)
    #     #_corner = _corner + [100,50]
    #     print(type(_corner[0][0][0]))
    #
    #
    #     cv2.putText(img=frame1, org=(int(_corner[0][0][0]), int(_corner[0][0][1])), text="1",
    #                 fontScale=1, color=(100, 255, 0), thickness=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    #     cv2.putText(img=frame1, org=(int(_corner[0][1][0]), int(_corner[0][1][1])), text="2",
    #                 fontScale=1, color=(100, 255, 0), thickness=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    #     cv2.putText(img=frame1, org=(int(_corner[0][2][0]), int(_corner[0][2][1])), text="3",
    #                 fontScale=1, color=(100, 255, 0), thickness=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    #     cv2.putText(img=frame1, org=(int(_corner[0][3][0]), int(_corner[0][3][1])), text="4",
    #                 fontScale=1, color=(100, 255, 0), thickness=3, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    #     cv2.putText(img=frame1, org=(int(_corner[0][0][0]) + 15  , int(_corner[0][0][1]) + 15), text="Area:" + str(area),
    #                 fontScale=1, color=(100, 255, 0), thickness=3, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    #
    #     max_x = max(_corner[0][0][0],_corner[0][1][0],_corner[0][2][0],_corner[0][3][0])
    #     min_x = min(_corner[0][0][0],_corner[0][1][0],_corner[0][2][0],_corner[0][3][0])
    #     max_y = max(_corner[0][0][1],_corner[0][1][1],_corner[0][2][1],_corner[0][3][1])
    #     min_y = min(_corner[0][0][1],_corner[0][1][1],_corner[0][2][1],_corner[0][3][1])
    #     cen_x = (max_x + min_x)/2; cen_y = (max_y + min_y)/2; cen_x = int(cen_x); cen_y = int(cen_y)
    #     cv2.circle(frame1, center=(cen_x, cen_y), radius=5, color=(20,100,200))

    # frame2 = aruco.drawDetectedMarkers(frame2, rejectedImgPoints)
    # Calculate FPS
    new_frame_time = time.time()
    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    if (new_frame_time - prev_frame_time == 0):
        fps = "100"
    else:
        fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = int(fps)
    fps = str(fps) + " " \
                     "FPS"

    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_4)

    cv2.imshow('Display', frame1)
    # cv2.imshow('Display2', frame2)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()