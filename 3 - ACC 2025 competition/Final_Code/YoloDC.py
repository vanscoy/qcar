##############################################
# This is the main driver for the YOLO algorithm. This file is built upon Joseph Redmon and Syed Abdul Gaffar implementations, edited and derived by Frederick Levins.
# This YOLOv3-tiny detects real-time objects for the Quanser QCar 1 and has eight essential useable classes for detection but could involve more without limiters.
# Weights, Config, and Text derived from COCO dataset, above authors, and YOLO resources; IT IS NOT CUSTOM TRAINED.
##############################################

import cv2
import numpy as np
import time
import threading
from Quanser.q_essential import Camera2D


# Main Yolo Class
class YOLODetector:

    # Initialize YOLO model with given configuration, weights, and classes.
    def __init__(self, config_path, weights_path, classes_path, confThresh=0.55, nmsThresh=0.4):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.confThresh = confThresh
        self.nmsThresh = nmsThresh

        # To handle redundant and over created bounding boxes
        self.lastDetectedObjs = []
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Assign random colors for visualization
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Selected classes (only process these)
        self.selectedClasses = ["car", "bus",  "traffic light", "truck", "motorbike",
                                    "stop sign", "bicycle", "traffic sign", "train", "dog", "cat", "parking meter", "person"]
        
        # find selected classes
        self.selectedIndices = [self.classes.index(cls) for cls in self.selectedClasses if cls in self.classes]

    # Check if a similar object was detected previously in a close location.
    def similarPrev(self, obj, threshold_px=30):
        for prev in self.lastDetectedObjs:
            if obj["class"] != prev["class"]:
                continue
            
            # Distance between centers of bounding boxes
            dx = obj["center"][0] - prev["center"][0]
            dy = obj["center"][1] - prev["center"][1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < threshold_px:
                return True
        return False
    
    # Retrieve the output layer names from the YOLO model.
    def getOutLay(self):
        layerNames = self.net.getLayerNames()
        return [layerNames[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    # Draw bounding box and label on the image.
    def drawPred(self, img, classIds, confidence, x, y, x_plus_w, y_plus_h):

        label = f"{self.classes[classIds]}: {confidence:.2f}"
        color = tuple(map(int, self.colors[classIds]))
        
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Process a frame and return the image with YOLO detections.
    def procFrame(self, frame):

          # Start time for YOLO processing, used for testing purposes
        startYoloTime = time.time()

        Height, Width = frame.shape[:2]
        scale = 0.00392

        # Convert frame to YOLO-compatible format
        blob = cv2.dnn.blobFromImage(frame, scale, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutLay())

        # Variables to store stats
        classIds = []
        confidences = []
        boxes = []
        detectedObjs = []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classIdLoc = np.argmax(scores)
                confidence = scores[classIdLoc]

                if confidence > self.confThresh and classIdLoc in self.selectedIndices:
                    cenX = int(detection[0] * Width)
                    cenY = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)

                    x = cenX - w // 2
                    y = cenY - h // 2

                    # Important needed attributes of detected objects for usage
                    currentObj = {
                        "class": self.classes[classIdLoc],
                        "confidence": round(confidence, 2),
                        "position": (x, y, w, h),
                        "center": (cenX, cenY)
                    }
                    if self.similarPrev(currentObj):
                         continue
                    classIds.append(classIdLoc)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

                    # for info dump
                    detectedObjs.append({
                    "class": self.classes[classIdLoc],
                    "confidence": round(confidence, 2),
                    "position": (x, y, w, h),
                    "center": (cenX, cenY)
                })

        # Apply Non-Maximum Suppression further to help handle over reading
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThresh, self.nmsThresh)

        # Draw detections
        for i in indices:
            x, y, w, h = boxes[i]
            self.drawPred(frame, classIds[i], confidences[i], x, y, x + w, y + h)
            # print(confidences[i]) test for confidence output

        self.lastDetectedObjs = detectedObjs

        # End time for YOLO processing
        endYoloTime = time.time()

        # Overall testing time, print if refining optimization 
        yoloProcTime = endYoloTime - startYoloTime

         # Return processed frame and detected object with attributes
        return frame, detectedObjs

