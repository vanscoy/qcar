from Quanser.q_essential import Camera3D
from Quanser.q_ui import gamepadViaTarget
import Quanser.q_interpretation as qi 
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
myCam1 = Camera3D(mode='RGB', frame_width_RGB=imageWidth, frame_height_RGB=imageHeight)
max_distance = 2
min_distance = 1
begin = True

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

    _,contours,_ = cv2.findContours(morphedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # morphed Image if using morph ops

    # draw contours
    #Fl - https://www.labelvisor.com/challenges-and-solutions-in-implementing-tight-bounding-boxes/ introduce convex hull bounding
    # set below but not above 
    for contour in contours:

            area = cv2.minAreaRect(contour)
            points = cv2.boxPoints(area)
            points = np.int0(points)

    return binaryImage

new = gpad.read()
try:
    # A button to start
    while begin:
        print('-----------------------------------------------------------------')
        new = gpad.read()
        if gpad.A == 1:
            begin = False
            speed = 0.074
    # B button to cancel
    #prev_center_xf = int(imageWidth/2)
    #prev_center_yf = int(croppedImageHeight/2)
    while gpad.B != 1:
        start = time.time()
        print('-----------------------------------------------------------------')
        myCam1.read_RGB()
        counter += 1

        front = myCam1.image_buffer_RGB[croppedImageHeight:480, :]
        binaryf = detectGrayscale(front)

        max_y_white_l = 0
        max_y_white_r = 0
        for i in range(0,croppedImageHeight-1):
            whiteLeft = binaryf[i][5]
            whiteRight = binaryf[i][634]
            if whiteLeft == 255:
                max_y_white_l = i
                #print(i)
            if whiteRight == 255:
                max_y_white_r = i
        #print('----------------------------------------------------------------')
        print(max_y_white_l)
        print(max_y_white_r)
        
        cv2.imshow('Binary Front Image', binaryf)

        # attempt at controls for SLAM
        angle = 0
        # check if the left side is significantly lower than the right and vice versa
        highThresh = 150
        lowThresh = 130

        """yDiff = abs(max_y_white_l - max_y_white_r)
        print(yDiff)
        # straight section
        if yDiff < 20:
            if max_y_white_r >= max_y_white_l:
                #turn = 'left'
                angle = max_y_white_r/400
            elif max_y_white_r < max_y_white_l:
                #turn = 'right'
                angle = -1*max_y_white_l/400
        # curve
        else:
            """
        # this causes errors because it crosses the yellow line
        # use right line if no errors from lighting
        """
        if max_y_white_r < croppedImageHeight/2:
            if max_y_white_r >= max_y_white_l:
                #turn = 'left'
                angle = .2*(max_y_white_r - highThresh)*.1
            elif max_y_white_r < max_y_white_l:
                #turn = 'right'
                angle = -.2*(lowThresh - max_y_white_r)*.1
        # use left line if no errors from lighting
        elif max_y_white_l < croppedImageHeight/2:
            if max_y_white_r < max_y_white_l:
                #turn = 'left'
                angle = .2*(max_y_white_l - highThresh)*.1
            elif max_y_white_r >= max_y_white_l:
                #turn = 'right
                angle = -.2*(lowThresh - max_y_white_l)*.1
        #"""
        """
        if max_y_white_r >= max_y_white_l:
            #turn = 'left'
            angle = max_y_white_r/400
        elif max_y_white_r < max_y_white_l:
            #turn = 'right'
            angle = -1*max_y_white_l/400
        #"""
        #"""
        turn = 'r'
        if turn == 'r':
            if max_y_white_l >= highThresh:
                angle = -.2*(max_y_white_l - highThresh)*.1
            elif max_y_white_l < lowThresh:
                angle = .2*(lowThresh - max_y_white_l)*.1
        elif turn == 'l':
            if max_y_white_r >= highThresh:
                angle = .2*(max_y_white_r - highThresh)*.1
            elif max_y_white_r < lowThresh:
                angle = -.2*(lowThresh - max_y_white_r)*.1
        #"""
        print(angle)
        """for i in range(0, imageWidth-1):
            if binaryf[120][i] == 255:
                print(i)"""

        ## Movement and Gamepadxit
        # right trigger for speed
        #mtr_cmd = np.array([.066, angle]) # need to replace with varius input on encoders and speeds
        mtr_cmd = np.array([speed, angle])
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
    myCam1.terminate()
    gpad.terminate()
    myCar.terminate()
    plt.close() 
