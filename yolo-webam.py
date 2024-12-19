import capture
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import urllib.request


cv2.namedWindow("preview")

phone = "https://10.91.213.119:8080//video" #phone
#capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("image.jpg")
capture.open(phone) #phone

capture.set(3, 1280)
capture.set(4, 720)


model = YOLO("../Yolo-Weights/yolov8l.pt")
#model = YOLO("../yolotrain/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

storeObjects =[]

while True:
    success, index = capture.read()
    results = model(index, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(index, (x1, y1, w, h))

            #confidene
            conf = math.ceil((box.conf[0]*100))

            #classname
            cls = int(box.cls[0])
            if conf > 0.1:
                cvzone.putTextRect(index, f'{classNames[cls]} {conf}{"%"}', (max(0, x1), max(38, y1)), scale=1.5, thickness=2)
            Objects = classNames[cls]
            if Objects in storeObjects:
                pass
            else:
                storeObjects.append(Objects)

    if index is not None:
        #cv2.imshow("preview", cv2.resize(index,(1280, 720))) #from phone
        cv2.imshow("preview", index)
    key = cv2.waitKey(1)
    if key == 27:  # exit when ESC
        break
print("The detected objects were:", storeObjects)
capture.release()
cv2.destroyWindow("preview")