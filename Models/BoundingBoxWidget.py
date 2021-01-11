import cv2

from Models.ArucoFinder import ArucoFinder
from Models.ObjectFound import *
from Utils.ObjectDetectionTools import cropImage


class BoundingBoxWidget(object):

    ObjectList = []


    def __init__(self, image):
        self.original_image = image
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            x = self.image_coordinates[0][0]
            y =  self.image_coordinates[0][1]
            w = self.image_coordinates[1][0] - self.image_coordinates[0][0]
            h = self.image_coordinates[1][1] - self.image_coordinates[0][1]

            # Create Object from rectangle
            choosenObject = ObjectFound(confidence=100, type="Manual", x=x, y = y,
                                        width= w, height=h, cropped_img=cropImage(frame=self.original_image,x=x, y=y, w=w, h=h))
            markerCorner = ArucoFinder.findCorners(frame=choosenObject.img_cropped, x=x, y=y)
            if (markerCorner is not None):
                choosenObject.MarkerCorners = markerCorner
                self.ObjectList.append(choosenObject)
            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # # Clear drawing boxes on right mouse button click
        # elif event == cv2.EVENT_RBUTTONDOWN:
        #     self.clone = self.original_image.copy()
        #     self.ObjectList = [] #Refresh List


    def show_image(self):
        return self.clone