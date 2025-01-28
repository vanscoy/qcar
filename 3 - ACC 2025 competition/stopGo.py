from Quanser.q_essential import Camera2D
from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
from matplotlib import pyplot as plt
from speedCalc import *
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
speed = speedCalc(robot_pos, myCar)
mtrSpeed = .066

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

def stopGo():
    # Josh - If stop sign detected (Uncomment when Stop-Go image detection implemented)
    #if [stopLight] >= [stopValue]:
    #    while (speed.encoder_speed() != 0)  
    #        slowSpeed -= .001
    #        mtr_cmd = np.array([slowSpeed, angle]
    #    time.sleep(3)
    #    mtr_cmd = np.array([.066, angle]
    #    time.sleep(2)
    return 0

new = gpad.read()
try:
    # B button to cancel
    #prev_center_xf = int(imageWidth/2)
    #prev_center_yf = int(croppedImageHeight/2)
    #left = leftCam.image_data[croppedImageHeight:480, :]
    #back = backCam.image_data[croppedImageHeight:480, :]
    #right = rightCam.image_data[croppedImageHeight:480, :]
    front = frontCam.image_data[croppedImageHeight:480, :]
    while gpad.B != 1:
        start = time.time()
        print('-----------------------------------------------------------------')
        #leftCam.read()
        #backCam.read()
        #rightCam.read()
        frontCam.read()
        counter += 1
        #binaryl = detectGrayscale(left)
        #binaryb = detectGrayscale(back)
        #binaryr = detectGrayscale(right)
        binaryf = detectGrayscale(front)
        #print(time.time() - start)

        lowest_white_l = 0
        lowest_white_r = 0
        for i in range(0,croppedImageHeight-1):
            whiteLeft = binaryf[i][5]
            whiteRight = binaryf[i][634]
            if whiteLeft == 255:
                lowest_white_l = i
                #print(i)
            if whiteRight == 255:
                lowest_white_r = i
        #print('----------------------------------------------------------------')
        #print(lowest_white_l)
        #print(lowest_white_r)
        print(speed.encoder_speed()) # Speed Units: m/s
        print(angle) # Speed Units: m/s
        #print(time.time() - start)
        # Reference: Motor speed is in Watts?

        #print(binaryf[int(croppedImageHeight*3/4)])
        #cv2.imshow('Contour Image', front)
        #cv2.imshow('Binary Left Image', binaryl)
        #cv2.imshow('Binary Right Image', binaryr)
        cv2.imshow('Binary Front Image', binaryf)
        #allCams = combineFeeds(left, back, right, front)
        #cv2.imshow('All Feeds', allCams)
        #print(time.time() - start)

        # attempt at controls for SLAM
        angle = 0
        # check if the left side is significantly lower than the right and vice versa
        direction = 'l'
        '''if direction == 'r':
            if lowest_white_l >= 90:
                # to the right
                angle = -.2*(lowest_white_l/10)
            elif lowest_white_l < 70:
                angle = .2*(lowest_white_l/10)
        elif direction == 'l':
            if lowest_white_r >= 90:
                # to the right
                angle = .2*(lowest_white_r/10)
            elif lowest_white_r < 70:
                angle = -.2**(lowest_white_r/10)'''

        if lowest_white_l >= 90:
            # to the right
            angle = -.025*(lowest_white_l/10)
        elif lowest_white_l < 70:
            angle = .025*(lowest_white_l/10)

        ## Movement and Gamepadxit
        # right trigger for speed
        mtr_cmd = np.array([mtrSpeed, angle]) # need to replace with varius input on encoders and speeds
        #mtr_cmd = np.array([.075*gpad.RT, angle])
        #mtr_cmd = np.array([.25*(1-abs(.5*gpad.LLA), .25*gpad.LLA]) - Autonomous Code

        # stopGo() # Stop sign functionality

        if (speed.encoder_speed() < .43 and angle == 0):
            print("speed")
            mtrSpeed = mtrSpeed + .002

        if (speed.encoder_speed() > .43 and angle == 0):
            print("slow")
            mtrSpeed = mtrSpeed - .002

        if (speed.encoder_speed() < .43 and angle != 0):
            print("speed angle")
            mtrSpeed = mtrSpeed + .003

        if (speed.encoder_speed() > .43 and angle != 0):
            print("slow angle")
            mtrSpeed = mtrSpeed - .003


        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        new = gpad.read()

        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)

        #prev_center_xf = center_xf
        #prev_center_yf = center_yf

        end = time.time()
        #print(end - start)
        
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
        cv2.waitKey(msSleepTime)
        #print(time.time() - start)

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
