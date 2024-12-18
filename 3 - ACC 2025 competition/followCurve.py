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
import datetime

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
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
angleRad = 0
max_distance = 5
mtrSpeed = 0.066
robot_pos = np.array([0.0, 0.0, 0.0])
isColor = False
manual = False
path = './outputVideos/CSI_Front_Camera/'
filename = 'output'
filetype = '.avi'
LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
frameTime = 0

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Object Initializations
frontCam = Camera2D(camera_id="3", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
myCar = QCar()
gpad = gamepadViaTarget(1)
speed = speedCalc(robot_pos, myCar)
new = gpad.read()
frameTimeList = list()
frameList = list()
prevFrames = list()


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
### Functions

# converts an BGR image to a binary image
# applies erosion and dilation to reduce error
# kernel is (5,5)
def BGR2Binary(image):
    image2 = image.copy()
    # convert bgr image to grayscale
    grayImage = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to reduce noise -- smoothes out grayscale fcn prior to threshold; can change sizes dependent on needs
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # simple threshold: every pixel with value > 100 becomes 255 (white)
    ret, binaryImage = cv2.threshold(blurredImage, 100, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to remove noise
    # kernel can change sizes, larger remove more noise but costworthy and chain effect
    kernel = np.ones((5, 5), np.uint8)
    # lose small holes in the foreground objects and connect nearby objects
    morphedImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
    # removes small noise points by eroding and then dilating the image
    morphedImage = cv2.morphologyEx(morphedImage, cv2.MORPH_OPEN, kernel)

    return morphedImage

# Function to find the highest y value of the white pixels in each given column
# image is a binary image as a 2d integer array where each element is 0 or 255
# cols is an integer array contain each column we want to search on
def findLowestWhite(image, cols):
    print('Columns to search on:', cols)
    maxY = [0] * len(cols)
    # iterate through each pixel in a column
    for y in range(0, croppedImageHeight):
        for x in range(0, len(cols)):
            if image[y][cols[x]] == 255:
                maxY[x] = y
    print('Max Y values:        ', maxY)
    return maxY

# Function to determine which direction the car should turn on a curve
# returns 's' = straight, 'l' = left, 'r' = right
def findDirection(maxY):
    turn = 's'
    if maxY[0] >= maxY[1] and maxY[1] >= maxY[2]:
        turn = 'r'
    elif maxY[4] >= maxY[3] and maxY[3] >= maxY[2]:
        turn = 'l'
    if (maxY[0] - maxY[1]) < 70 and (maxY[0] - maxY[1]) >= 40 and maxY[0] >= 120:
        turn = 'r'
    elif (maxY[4] - maxY[3]) < 70 and (maxY[4] - maxY[3]) >= 30 and maxY[4] >= 120:
        turn = 'l'
    if maxY[0] >= 150 and maxY[0] <= 160 and maxY[4] >= 150 and maxY[4] <= 160:
        turn = 's'
    print('Turn Direction =', turn)
    return turn

# Function to determine the angle that the car should turn at
# adjusts angle based on how far off it is
def setAngle(maxY, section):
    angleRad = 0
    highThresh = 170
    lowThresh = 145
    if section == 'straight':
        if maxY[0] >= highThresh:
            angleRad = -.02*(abs(maxY[0] - highThresh))
        elif maxY[0] < lowThresh:
            angleRad = .02*(lowThresh - maxY[0])
    angleDeg = math.degrees(angleRad)
    print('Angle (Radians)=',angleRad)
    print('Angle (Degrees)=',angleDeg)
    return angleRad, angleDeg


# Function to set and change the speed of the car
# angle is a float. Sets speed based on sharpness of turn
# prev_angle was the angle last iteration
# prev_speed was the speed last iteration
# elapsed_time is the time between last iteration and this iteration
# manual is a boolean that checks if we are controlling the car speed with the gamepad
# speed mod is a modifier for the speed value from the gamepad. Defaults to 0.066
def setSpeed(angle, prev_mtrSpeed, counter, turn, manual, speed_mod=0.066):
    mtrSpeed = 0.066
    msSpeed = speed.encoder_speed()
    angleMag = abs(angle)
    if manual == True:
        new = gpad.read()
        mtrSpeed = speed_mod*gpad.RT
    elif section == 'straight':
        mtrSpeed = 0.075
    print('Speed (m/s) =',msSpeed)
    print('mtrSpeed =',mtrSpeed)
    return msSpeed, mtrSpeed

# Function to determine which section of the track the car is on
def findSection(maxY, cols, curr_section):
    section = curr_section
    if maxY[0] >= 145 and maxY[0] <= 170 and maxY[int(len(maxY)-1)] >= 145 and maxY[int(len(maxY)-1)] <= 170:
        section = 'straight'
    print('Section =', section)
    return section

def videoName(path, filename, filetype):
    now = datetime.datetime.now()
    name = path + filename + '_' + str(now) + filetype
    name = name.replace(' ', '_')
    return name

# usually around 4-5
# a few outliers mess this up, especially at start; maybe reject outliers
# frameTimeList[0] is a huge outlier
# seems to take about 4-5 cycles for frame time to stabilize
# maybe graph frame time
# I could simply hardcode framerate for the video but I would like to be able to use this in the loop as well
def frameRate(frameTimeList):
    meanFrameTime = (sum(frameTimeList) - frameTimeList[0]) / len(frameTimeList)
    fps = 1/meanFrameTime
    return fps

# frames is an list of frames that will be used to make the video
# filename is the name of the video (including the relative path to video) and ends with .avi
# fps is the framerate as a float
# isColor is a boolean that tells us if the video will be in color
# maybe add time and date to filename
# add terminal output to frames
def saveVideo(frames, filename, fps, isColor):
    makeVid = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (imageWidth,croppedImageHeight), isColor)
    for i in range(0,len(frames)):
        makeVid.write(frames[i])
    makeVid.release()

# maxSize is the maximum number of frames we want to save
def saveAsPrevFrame(prevData, maxSize):
    prevFrames.append(prevData)
    if len(prevFrames) > maxSize:
        prevFrames.pop(0) # pop the first frame to save memory

# takes the index of the data that you want to read and how many frames ago you read it from and returns said data
# prevFrame[frameNum] = [binaryf.copy(), maxY, section, angleRad, mtrSpeed, counter, current, batteryVoltage, encoderCounts, frameTime]
def readPrevFrame(dataNum, framesAgo):
    if len(prevFrames) < framesAgo:
        return
    size = len(prevFrames)
    frameNum = size - framesAgo
    return prevFrames[frameNum][dataNum]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
### Main

try:
    frontCam.read()
    # B button to cancel
    while gpad.B != 1:
        print('-----------------------------------------------------------------')
        startFrame = time.time()
        if counter >= 1:
            frontCam.read()
        front = frontCam.image_data[croppedImageHeight:480, :]
        binaryf = BGR2Binary(front)
        cv2.imshow('Binary Front Image', binaryf)
        new = gpad.read()

        # x = 5, and x = 634 are near the edge of the screen but not exactly on the edge
        cols = [5, 160, 320, 480, 634]
        maxY = findLowestWhite(binaryf, cols)
        curr_section = readPrevFrame(2,1)
        section = findSection(maxY, cols, curr_section)
        angleRad, angleDeg = setAngle(maxY, section)
        msSpeed, mtrSpeed = setSpeed(angleRad, readPrevFrame(4,1), counter, section, manual)
        print('Counter:', counter)
        new = gpad.read()

        #mtrSpeed = 0.066
        # activate controls
        mtr_cmd = np.array([mtrSpeed, angleRad])
        #mtr_cmd = np.array([mtrSpeed, .25*gpad.LLA])
        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)
        new = gpad.read()
        
        # wait statement
        endCompute = time.time()
        computationTime = endCompute - startFrame
        sleepTime = sampleTime - ( computationTime % sampleTime )
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
        cv2.waitKey(msSleepTime)
        endFrame = time.time()
        new = gpad.read()

        # save the previous 10 frames
        prevData = [binaryf.copy(), maxY, section, angleRad, mtrSpeed, counter, current, batteryVoltage, encoderCounts, frameTime]
        saveAsPrevFrame(prevData, 10)
        counter += 1
        frameList.append(binaryf.copy())
        frameTime = endFrame - startFrame
        print('Frame Time =',frameTime)
        frameTimeList.append(frameTime)

except KeyboardInterrupt:
	print("User interrupted!")
finally:
    frontCam.terminate()
    gpad.terminate()
    myCar.terminate()
    fps = frameRate(frameTimeList)
    name = videoName(path, filename, filetype)
    print('FPS = ', fps)
    print('Filename = ', name)
    saveVideo(frameList, name, fps, isColor)
    plt.close()
