import math


class PickPoint:
    x = 0
    y = 0
    z = 0

    pitch = 0
    yaw = 0
    roll = 0


    closeness = 0

    def __init__(self, x = 0, y = 0, z = 0, pitch = 0, yaw = 0, roll = 0):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        self.closeness = math.sqrt(self.x ** 2 + self.y ** 2)

    def info(self):
        print("X,Y,Z: {} ; {} ; {}".format(self.x, self.y, self.z))
        print("Pitch,Yaw,Roll: {} ; {} ; {}".format(self.pitch, self.yaw, self.roll))