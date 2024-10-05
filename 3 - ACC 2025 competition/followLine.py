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
    if color == 'white':
        lower = np.array([0,0,127])
        upper = np.array([180,25,255])
    elif color == 'red':                    # detects tan instead
        lower = np.array([0, 100, 100])
        upper = np.array([20, 255, 255])
    elif color == 'yellow':                 # widen range
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif color == 'green':                  # range too dark
        lower = np.array([40, 30, 100])
        upper = np.array([80, 255, 255])
    elif color == 'blue':                   # range too dark
        lower = np.array([100, 150, 50])
        upper = np.array([140, 255, 255])
    else:
        print('Pick a different color')
        return image

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
    while gpad.B != 1:
        start = time.time()

        leftCam.read()
        backCam.read()
        rightCam.read()
        frontCam.read()

        counter += 1

        # Cropping camera feeds
        # half height for all cams
        left = leftCam.image_data[croppedImageHeight:480, :].copy()
        back = backCam.image_data[croppedImageHeight:480, :].copy()
        right = rightCam.image_data[croppedImageHeight:480, :].copy()
        front = frontCam.image_data[croppedImageHeight:480, :].copy()

        leftHSV, xl, yl, wl, hl = detectHSV(left, 'yellow')
        backHSV, xb, yb, wb, hb = detectHSV(back, 'yellow')
        rightHSV, xr, yr, wr, hr = detectHSV(right, 'yellow')
        frontHSV, xf, yf, wf, hf = detectHSV(front, 'yellow')

        # red box
        mid_w = 70
        mid_h = croppedImageHeight
        mid_x = int(imageWidth/2 - mid_w/2)
        mid_y = 0
        front = cv2.rectangle(front, (mid_x, mid_y), (int(mid_x+mid_w), int(mid_y+mid_h)), (0, 0, 255), 2, cv2.FILLED, 0)

        center_xf = int(xf + (wf / 2))
        center_yf = int(yf + (hf / 2))

        #if center_xf != 0 & center_yf != 0:
        #print(center_xf)
        #print(center_yf)
        frontHSV = cv2.circle(frontHSV, (center_xf, center_yf), 10, (0,0,255), -1)

        hsvObjects = combineFeeds(leftHSV, backHSV, rightHSV, frontHSV)

        """if front_cam_box_points[1][1] >= int(mid_x)+70:
            print('Too far right')
        elif front_cam_box_points[3][1] <= mid_x:
            print('Too far left')
        else:
            print('centered')"""

        cv2.imshow('HSV Objects', hsvObjects) # Error created here

        # return points of bounding box
        # draw another box in the middle of camera
        # check if points returned are within the middle box
        # if right of middle box then turn right
        # if left of middle box then turn left
        # https://stackoverflow.com/questions/29739411/what-does-cv2-cv-boxpointsrect-return

        #if os.geteuid() != 0:
        #    args = ['sudo', sys.executable] + sys.argv + [os.environ]
        #    os.execlpe('sudo', *args)


        # attempt at controls for SLAM
        angle = 0
        if center_xf >= imageWidth/2:
            # to the right
            angle = -.2
        elif center_xf < imageWidth/2:
            angle = .2

        print(center_xf)
        print(center_yf)
        print(angle)
        print(imageWidth/2)
        print()

        ## Movement and Gamepadxit
        # right trigger for speed
        '''.075*gpad.RT''' # code to use right trigger for manual testing
        mtr_cmd = np.array([.05, angle]) # need to replace with varius input on encoders and speeds
        print(gpad.RT)
        #mtr_cmd = np.array([.25*(1-abs(.5*gpad.LLA), .25*gpad.LLA]) - Autonomous Code
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        new = gpad.read()

        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)

    

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
