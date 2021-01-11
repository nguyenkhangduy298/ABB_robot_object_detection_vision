import math
import string
from datetime import datetime
from math import sqrt
from random import random

from Models.PickPoint import PickPoint

MAX_SIZE_CODE_RANDOM_NAME = 6


class ObjectFound:
    Class = "None"
    name = "Noname"

    middle_x = 0
    middle_y = 0

    x_pixel_coordinate = 0  # Coordinate
    y_pixel_coordinate = 0

    # Width and Height in pixel coordinate
    width = 0
    height = 0
    confidence_level = 0
    created_time = None  # May need for later usage

    # Image Frame
    img_cropped = None

    PickedPoint = PickPoint()

    MarkerCorners = None

    def __init__(self, type, x, y, width, height, confidence, cropped_img):
        self.Class = type
        # Coordinates of top, left position
        self.x_pixel_coordinate = x
        self.y_pixel_coordinate = y
        # Size and dimensions
        self.width = width
        self.height = height
        # Centrals Points
        self.middle_y = y + height / 2
        self.middle_x = x + height / 2
        # Confidence Level
        self.confidence_level = confidence
        # Time Created
        self.created_time = datetime.now()
        # Cropped Image from the Yolo
        self.img_cropped = cropped_img

    # Code Generator
    def id_generator(self, size=MAX_SIZE_CODE_RANDOM_NAME, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    # Random name generator: Name is choosen by rule: TYPE + CODE
    def randomNameGenerator(self, type):
        code = self.id_generator()
        return str(type) + str(code)

    # Calculate distance to one other object (need to be sametype)
    def distanceToOtherObject(self, otherObject):
        diff = (self.x_pixel_coordinate - otherObject.x_pixel_coordinate) ** 2 - \
               (self.y_pixel_coordinate - otherObject.y_pixel_coordinate) ** 2
        return sqrt(diff)

    def giveInfor(self):
        print("Object named: " + self.name + " Class: " +
              str(self.Class) + " Coordinate[" + str(self.x_pixel_coordinate) + " ," + str(
            self.y_pixel_coordinate) + "]"
              + " Size: " + str(self.width) + "x" + str(self.height))

    def measureToOrigin(self):
        print("Begin")
        print(PickPoint.x)
        if (PickPoint.x != 0) & (PickPoint.y != 0) & (PickPoint.z != 0 ):
            print("True")
            distance = int(math.sqrt(PickPoint.x ** 2 + PickPoint.y ** 2))

            return distance
        return 9999

