import argparse
import os
import time
import ntpath

import cv2
import numpy

from Models.YoloObject import Yolo
from Utils.ObjectDetectionTools import PrepareClass

ap = argparse.ArgumentParser()
# Yolo Parameters
ap.add_argument('-f', '--ImageFolder', required=False, help='Left Camera: Port')
ap.add_argument('-cl', '--classYolo', required=True, help='Path to class name file of YoloV3') # Url to model class
ap.add_argument('-cfg', '--cfgYolo', required=True, help='Path to config file of YoloV3') # Url to YOLO Config file
ap.add_argument('-w', '--weightYolo', required=True, help='Path to weight file of YoloV3') # Url To Weight file

args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) +" "+ str(numpy.round(confidence, 2))

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def createStringLabel(real_width, real_height, x,y,w,h, label):
    return str(label) + " " + str(x/real_width) + " "+ str(y/real_height) +" "+ str(w/real_width) + " " + str(h/real_height)+"\n"


def load_images_from_folder(folder, net):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,filename))
        if image is not None:

            # Detect object
            Width = image.shape[1]
            Height = image.shape[0]
            print("BEFORE" + str(Width) + " " + str(Height))
            scale = 0.00392
            blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.3
            nms_threshold = 0.1

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = numpy.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            write_file_path = os.path.join(folder,filename).replace(".png", ".txt")
            write_p = open(write_file_path, "w+")
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
                write_p.write(createStringLabel(x=x+w/2, y=y+h/2, w=w, h=h, label=class_ids[i], real_height=Height, real_width=Width))
            write_p.close()
            cv2.imshow("Test", image)
            print("After" + str(image.shape))


            cv2.waitKey(1)
    return images

directory = str(args.ImageFolder)

# Prepare the model
if __name__ == '__main__':
    YoloObject = Yolo(configUrl=str(args.cfgYolo), classUrl=str(args.classYolo), weightUrl=str(args.weightYolo))

    classes = PrepareClass(class_path=YoloObject.yoloClass)
    COLORS = numpy.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(YoloObject.yoloWeight, YoloObject.yoloConfigUrl)
    load_images_from_folder(directory, net)

