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
def detectGrayscale(image):
    # convert bgr image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # simple threshold
    # everything with a value above 100 becomes 255 (white)

    # Gaussian blur to reduce noise -- smoothes out grayscale fcn prior to threshold; can change sizes dependent on needs
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

    ret, binaryImage = cv2.threshold(blurredImage, 100, 255, cv2.THRESH_BINARY) # blurredImage if using 
    # adaptive threshold; should help with lighting
    # binaryImage = cv2.adaptiveThreshold(blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 20) 
    # Otsu's method could also be helpful helps with varying illumniation levels of colors
    #ret, binaryImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8) # kernel can change sizes, larger remove more noise but costwrothy and chain effect

    # lose small holes in the foreground objects and connect nearby objects
    morphedImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
    # removes small noise points by eroding and then dilating the image
    morphedImage = cv2.morphologyEx(morphedImage, cv2.MORPH_OPEN, kernel)

    return morphedImage

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
        print('-----------------------------------------------------------------')
        frontCam.read()
        counter += 1
        front = frontCam.image_data[croppedImageHeight:480, :]
        binaryf = detectGrayscale(front)

        maxY5 = 0
        maxY160 = 0
        maxY320 = 0
        maxY480 = 0
        maxY634 = 0
        for i in range(0,croppedImageHeight-1):
            white5 = binaryf[i][5]
            if white5 == 255:
                maxY5 = i
            white160 = binaryf[i][160]
            if white160 == 255:
                maxY160 = i
            white320 = binaryf[i][320]
            if white320 == 255:
                maxY320 = i
            white480 = binaryf[i][480]
            if white480 == 255:
                maxY480 = i
            white634 = binaryf[i][634]
            if white634 == 255:
                maxY634 = i
        #print(maxYl)
        #print(maxYr)
        print('Max on Col 5 =', maxY5)
        print('Max on Col 160 =', maxY160)
        print('Max on Col 320 =', maxY320)
        print('Max on Col 480 =', maxY480)
        print('Max on Col 634 =', maxY634)
        
        
        cv2.imshow('Binary Front Image', binaryf)

        # attempt at controls for SLAM
        angle = 0
        # check if the left side is significantly lower than the right and vice versa
        highThresh = 160
        lowThresh = 150
        #"""
        turn = 'r'
        """
        if maxY5 >= maxY160 and maxY160 >= maxY320:
            turn = 'r'
        elif maxY634 >= maxY480 and maxY480 >= maxY320:
            turn = 'l'
        #"""
        if (maxY5 - maxY160) < 70 and (maxY5 - maxY160) >= 40 and maxY5 >= 120:
            turn = 'r'
            print('thing1')
        elif (maxY634 - maxY480) < 70 and (maxY634 - maxY480) >= 30 and maxY634 >= 120:
            turn = 'l'
            print('thing2')
        print('Turn Direction =', turn)
        #right turn and straight
        if turn == 'r':
            if maxY5 <= 120 or maxY5 >= highThresh:
                angle = -.2*(abs(maxY5 - highThresh))*.1
            elif maxY5 < lowThresh:
                angle = .2*(lowThresh - maxY5)*.1
        # left turn
        elif turn == 'l':
            if maxY634 <= 120 or maxY634 >= highThresh:
                angle = .2*(maxY634 - highThresh)*.1
            elif maxY634 < lowThresh:
                angle = -.2*(lowThresh - maxY634)*.1
        #"""
        print('Angle =',angle)

        ## Movement and Gamepadxit
        # right trigger for speed
        #speed = 0.066*gpad.RT
        speed = 0.066
        #mtr_cmd = np.array([.066, angle]) # need to replace with varius input on encoders and speeds
        mtr_cmd = np.array([speed, angle])
        print('Speed =',speed)
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
