##############################################
# This is the main driving code for node detection involving using openCV blue color masking. 
# Within the track, each node was designated and designed to be read through frames for the robot to localize itself.
# This code can be equally used for red and blue color masking, but blue was chosen in the end with a higher success rate and better real life outcomes.
# Zach Copenhaver wrote the code for blue tape masking, Frederick Levins wrote the code for red tape masking with overall support for both from Jakob Feltz.
##############################################

import cv2
import numpy as np
import threading
from PIL import Image

class BlueTapeDetector:
    def __init__(self):
        self.blueDetect = False
        self.boundBox = []
        self.lock = threading.Lock()

    #Detects blue tape using HSV filtering and draws bounding boxes.
    def detectBlueT(self, frame):
        if frame is None or frame.size == 0:
            
            # Prevent processing invalid frames
            return None 
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # Mask limits
        lowerBlue1 = np.array([0, 125, 60])
        upperBlue1 = np.array([10, 255, 255])
        lowerBlue2 = np.array([160, 110, 80])
        upperBlue2 = np.array([185, 255, 255])

        
        # Create masks Blue Masks
        mask1 = cv2.inRange(hsv, lowerBlue1, upperBlue1)
        mask2 = cv2.inRange(hsv, lowerBlue2, upperBlue2)

        # Blue Masks
        mask1 = cv2.inRange(hsv, lowerBlue1, upperBlue1)
        mask2 = cv2.inRange(hsv, lowerBlue2, upperBlue2)

        # Combine masks 
        mask =  cv2.bitwise_or(mask1, mask2)


        # Reduce noise
        kernel = np.ones((5, 5), np.uint8) # test at 3
        
        # Can increase iterations for more than a singular pass 
        # down = less loss of details, up = remove more noise and close more gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 1) # remove small noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 1) # fill holes
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find mask apparence
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        boundBox = []

        # Drawing bounding boxes for apparent masks
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Minimum size threshold
                detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                # only detect in lower half
                if y > 200:
                    boundBox.append((x, y, x + w, y + h))

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update shared variables safely
        with self.lock:
            self.blueDetect = detected
            self.boundBox = boundBox

        # Return frame with bounding boxes
        return frame, boundBox, self.blueDetect 

    #Returns whether blue tape is detected
    def isBlueDetected(self):
        with self.lock:
            return self.blueDetect

    def findBlue(self, image):
        if image is None or image.size == 0:
            return None  # Prevent processing invalid frames
        croppedLeftX = 100 #0 # maybe crop right a bit more
        croppedRightX = 540 #640 
        croppedTopY = 240
        croppedBottomY = 480
        hsv = cv2.cvtColor(image[croppedTopY:croppedBottomY,croppedLeftX:croppedRightX], cv2.COLOR_BGR2HSV) # try image[240:,240:400]
        frame = image.copy()
        found = False
        blue = [255,0,0]
        lower, upper = self.getLimits(blue)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8) # test at 3
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 1) # remove small noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 1) # fill holes
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        maskp = Image.fromarray(mask)
        bbox = maskp.getbbox()
        if bbox is not None:
            
            # change to match change in screen
            bbox = (bbox[0]+croppedLeftX, bbox[1]+croppedTopY, bbox[2]+croppedLeftX, bbox[3]+croppedTopY)
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0,255,0), 5)
            
            # convert bbox to (x1, y1, w, h) to match the rest of the code
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox = (x1, y1, w, h)
            # print(f"🔵 Blue Tape Detected at: x={x1}, y={y1}, w={w}, h={h}")
            found = True
        else:
            bbox = []
            found = False
        
        # Update shared variables safely
        with self.lock:
            self.blueDetect = found
            self.boundBox = bbox
        return frame, mask, bbox, found

    def getLimits(self, color):
        c = np.uint8([[color]])
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        lowerLimit = hsvC[0][0][0] - 40, 100, 100
        upperLimit = hsvC[0][0][0] + 40, 255, 255
        lowerLimit = np.array(lowerLimit, dtype=np.uint8)
        upperLimit = np.array(upperLimit, dtype=np.uint8)
        return lowerLimit, upperLimit                    

# Continuously detects blue tape and displays output
def blueTapeProc(detector, camera, camLock):
    try:
        while True:
            with camLock:
                camera.read()
                frame = camera.image_data.copy()
                frame = frame[200:400, :]

            if frame is None or frame.size == 0:
                print("Blue Tape Error: No valid frame.")
                continue

            procFrame = detector.detectBlueT(frame)

            return procFrame
    except Exception as e:
        print(f"Blue Tape Thread Error: {e}")