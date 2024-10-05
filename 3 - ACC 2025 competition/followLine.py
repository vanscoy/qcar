from Quanser.q_essential import Camera2D
from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
from matplotlib import pyplot as plt
import time
import struct
import numpy as np 
import cv2
import math
import os
import sys
import utils

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 30.0
sampleTime = 1/sampleRate
simulationTime = 10.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
counter = 0
imageWidth = 640
imageHeight = 480
croppedImageHeight = int(imageHeight/2)
#cameraID = '3'
angle = 0
max_distance = 5
robot_pos = np.array([0.0, 0.0, 0.0])

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the CSI cameras
#myCam = Camera2D(camera_id=cameraID, frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
rightCam = Camera2D(camera_id="0", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
backCam = Camera2D(camera_id="1", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
leftCam = Camera2D(camera_id="2", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
frontCam = Camera2D(camera_id="3", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)

myCar = QCar()
gpad = gamepadViaTarget(1) 

# function to detect objects based on HSV color
# draws contour boxes around object in original bgr image instead of in HSV image to help with comparing to grayscale
# uses erosion and dilation
# so far only detects one color per call
def detectHSV(image, color):
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0,0,190), (255,65,255))
    yellow_mask = cv2.inRange(hsv, (15,0,0), (36, 255, 255))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked,0,0,0,0
    """

    if color == 'white':
        lower = np.array([0,0,200])
        upper = np.array([180,25,255])
    elif color == 'yellow':
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])

    # Need to set to HSV for proper color usage 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # for number of masks:
    # Create a mask set specific to white color bounds
    # can we make this function detect multiple colors per call?
    mask = cv2.inRange(hsv, lower, upper)
    
    _,contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    
    # Draw bounding boxes around detected green objects
    for contour in contours:
        area = cv2.minAreaRect(contour)
        areaTest = cv2.contourArea(contour)
        if areaTest >= 100: # causes issues on turn. results in 0'd circle 
            points = cv2.boxPoints(area)
            points = np.int0(points)
            cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
            x,y,w,h = cv2.boundingRect(contour)
    #maxArea = max(contours, key = cv2.contourArea)
    #x,y,w,h = cv2.boundingRect(maxArea)
        
    return image,x,y,w,h

def combineFeeds(leftCam, backCam, rightCam, frontCam):
    # defining barriers to display between the camera feeds
    horizontalBlank = np.zeros((20, 2*imageWidth+60, 3), dtype=np.uint8)
    verticalBlank = np.zeros((croppedImageHeight, 20, 3), dtype=np.uint8)

    # combine all images into one array
    allCams = np.concatenate(
        (horizontalBlank,
            np.concatenate(
                (verticalBlank, leftCam, verticalBlank, frontCam, verticalBlank),
                axis = 1),
            horizontalBlank,
            np.concatenate(
                (verticalBlank, backCam, verticalBlank, rightCam, verticalBlank),
                axis = 1),
            horizontalBlank),
        axis=0)
    
    return allCams

new = gpad.read()
try:
    # B button to cancel
    #prev_center_xf = int(imageWidth/2)
    #prev_center_yf = int(croppedImageHeight/2)
    while gpad.B != 1:
        start = time.time()

        #leftCam.read()
        #backCam.read()
        #rightCam.read()
        frontCam.read()

        counter += 1

        # Cropping camera feeds
        # half height for all cams
        #left = leftCam.image_data[croppedImageHeight:480, :]
        #back = backCam.image_data[croppedImageHeight:480, :]
        #right = rightCam.image_data[croppedImageHeight:480, :]
        front = frontCam.image_data[croppedImageHeight:480, :]

        #leftHSV, xl, yl, wl, hl = detectHSV(left,'white')
        #leftHSV, xl, yl, wl, hl = detectHSV(left,'yellow')
        #backHSV, xb, yb, wb, hb = detectHSV(back,'white')
        #backHSV, xb, yb, wb, hb = detectHSV(back,'yellow')
        #rightHSV, xr, yr, wr, hr = detectHSV(right,'white')
        #rightHSV, xr, yr, wr, hr = detectHSV(right,'yellow')
        frontHSV, xfw, yfw, wfw, hfw = detectHSV(front,'white')
        frontHSV, xfy, yfy, wfy, hfy = detectHSV(front,'yellow')

        """center_xf = int((xfw+wfw + xfy) / 2)
        center_yf = int(((yfy+hfy) + yfw) / 2)

        center_x_diff = abs(prev_center_xf - center_xf)
        center_y_diff = abs(prev_center_yf - center_yf)

        if center_x_diff > 200:
            center_xf = prev_center_xf
            center_yf = prev_center_yf
        elif center_y_diff > 200:
            center_xf = prev_center_xf
            center_yf = prev_center_yf"""

        #if center_xf != 0 & center_yf != 0:
        #print(center_xf)
        #print(center_yf)
        center_xfw = int(xfw + (wfw / 2))
        center_yfw = int(yfw + (hfw / 2))
        frontHSV = cv2.circle(frontHSV, (center_xfw, center_yfw), 5, (0,0,255), -1)

        center_xfy = int(xfy + (wfy / 2))
        center_yfy = int(yfy + (hfy / 2))
        frontHSV = cv2.circle(frontHSV, (center_xfy, center_yfy), 5, (0,0,255), -1)

        center_xfc = int((center_xfw+center_xfy-100)/2)
        center_yfc = int((center_yfw+center_yfy)/2)
        frontHSV = cv2.circle(frontHSV, (center_xfc, center_yfc), 5, (0,0,255), -1)

        frontHSV = cv2.line(frontHSV, (int(imageWidth/2),0), (int(imageWidth/2),croppedImageHeight), (255,0,0), 5)
        #center_box = np.array([int(imageWidth/2-30), 0, int(imageWidth/2+30), int(croppedImageHeight)])
        #frontHSV = cv2.rectangle(frontHSV, (center_box[0], center_box[1]), (center_box[2], center_box[3]), (255,0,0), 5, cv2.FILLED, 0)

        #hsvObjects = combineFeeds(leftHSV, backHSV, rightHSV, frontHSV)

        cv2.imshow('HSV Objects', frontHSV)

        # return points of bounding box
        # draw another box in the middle of camera
        # check if points returned are within the middle box
        # if right of middle box then turn right
        # if left of middle box then turn left
        # https://stackoverflow.com/questions/29739411/what-does-cv2-cv-boxpointsrect-return

        # attempt at controls for SLAM
        angle = 0
        if center_xfc >= imageWidth/2:
            # to the right
            angle = -.2
        elif center_xfc < imageWidth/2:
            angle = .2

        print(center_xfc)
        print(center_yfc)
        print(angle)
        print(imageWidth/2)
        print()

        ## Movement and Gamepadxit
        # right trigger for speed
        mtr_cmd = np.array([.05, angle]) # need to replace with varius input on encoders and speeds
        #mtr_cmd = np.array([.075*gpad.RT, angle])
        #mtr_cmd = np.array([.25*(1-abs(.5*gpad.LLA), .25*gpad.LLA]) - Autonomous Code
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        new = gpad.read()

        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)

        #prev_center_xf = center_xf
        #prev_center_yf = center_yf

        end = time.time()
        
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
	print("User interrupted!")
finally:
    leftCam.terminate()
    backCam.terminate()
    rightCam.terminate()
    frontCam.terminate()
    gpad.terminate()
    myCar.terminate()
    plt.close() 
