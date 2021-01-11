import cv2
import numpy as np

from Models.ArucoFinder import ArucoFinder
from Models.ObjectFound import ObjectFound

# Return of Classes obj Objects
def PrepareClass(class_path):
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Get Final Result, (outputlayer)
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Setup
def PrepareNet(weight_path, cfg_path):
    return cv2.dnn.readNet(weight_path, cfg_path)

# Function to crop image, return new img from original image
def cropImage(x, y, w, h, frame):
    return frame[y:y + h, x:x + w]

# Find object inside an image, create a list of object then return it.
def ObjectDetection(frame, weight, yoloCfg, scale=0.00392, confident_threshold=0.5, nms_threshold=0.4):
    ObjectDetectedList = []; Width = frame.shape[1]; Height = frame.shape[0]
    net = PrepareNet(weight_path=weight, cfg_path=yoloCfg) #TODO Refactor to not call this function in more than once in the program

    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # If confidence > conf_threshold
            if confidence > confident_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)

                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confident_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        image_cropped = cropImage(int(x),int(y),int(w),int(h),frame) # Get the bounding box of object from the images
        #markerCorner = ArucoFinder.findCorners(frame,x,y) # Only get the marker with highest area found (Most likely to be picked able)
        foundObject = ObjectFound(class_ids[i], x=x, y=y, width=w, height=h, confidence=confidences[i], cropped_img=image_cropped)

        # if markerCorner.all() != None:
        #     foundObject.MarkerCorners = markerCorner #Set marker corner for the object found
        ObjectDetectedList.append(foundObject) # Only get the marker where marker is found (Pick-able)
            # TODO: This is due to handle cylinder; May upgrade this part and find a better solution

    return ObjectDetectedList


