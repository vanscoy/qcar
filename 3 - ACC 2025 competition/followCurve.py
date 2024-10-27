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
#cameraID = '3'
angle = 0
max_distance = 5
robot_pos = np.array([0.0, 0.0, 0.0])

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the CSI cameras
frontCam = Camera2D(camera_id="3", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)

myCar = QCar()
gpad = gamepadViaTarget(1)

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
    # kernel can change sizes, larger remove more noise but costwrothy and chain effect
    kernel = np.ones((5, 5), np.uint8)
    # lose small holes in the foreground objects and connect nearby objects
    morphedImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
    # removes small noise points by eroding and then dilating the image
    morphedImage = cv2.morphologyEx(morphedImage, cv2.MORPH_OPEN, kernel)

    return morphedImage

# function to format the display of multiple camera feeds
# currently set for display of all 4 cams
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
def setAngle(maxY, turn):
    angle = 0
    highThresh = 170
    lowThresh = 145
    #right turn and straight
    if turn == 'r':
        if maxY[0] <= 120 or maxY[0] >= highThresh:
            angle = -.02*(abs(maxY[0] - highThresh))
        elif maxY[0] < lowThresh:
            angle = .02*(lowThresh - maxY[0])
    # left turn
    elif turn == 'l':
        if maxY[4] <= 120 or maxY[4] >= highThresh:
            angle = .02*(maxY[4] - highThresh)
        elif maxY[4] < lowThresh:
            angle = -.02*(lowThresh - maxY[4])
    elif turn == 's':
        if maxY[0] <= 120 or maxY[0] >= highThresh:
            angle = -.02*(abs(maxY[0] - highThresh))
        elif maxY[0] < lowThresh:
            angle = .02*(lowThresh - maxY[0])
    if angle > 0.5:
        angle = 0.4
    elif angle < -0.5:
        angle = -0.4
    print('Angle =',angle)
    return angle

# Function to set and change the speed of the car
# angle is a float. Sets speed based on sharpness of turn
# prev_angle was the angle last iteration
# prev_speed was the speed last iteration
# elapsed_time is the time between last iteration and this iteration
# manual is a boolean that checks if we are controlling the car speed with the gamepad
# speed mod is a modifier for the speed value from the gamepad. Defaults to 0.066
def setSpeed(angle, prev_angle, prev_speed, counter, turn, manual, speed_mod=0.066):
    speed = 0.08 #0.075
    angleMag = abs(angle)
    if manual == True:
        new = gpad.read()
        speed = speed_mod*gpad.RT
    elif turn == 's':
        # keep speed for 4 iterations so that the speed change can actually be in effect
        if counter % 4 != 0:
            speed = prev_speed
        elif angleMag >= 0.2:
            speed = speed * 0.8
        elif angleMag >= 0.1:
            speed = speed * 0.9
        else:
            speed = speed * 1.1
    else:
        if angleMag >= 0.35:
            speed = speed * 0.95
        else:
            speed = speed
        '''
        # keep speed for 4 iterations so that the speed change can actually be in effect
        if counter % 4 != 0:
            speed = prev_speed
        elif prev_angle * angle < 0:
            if angleMag >= 0.35:
                speed = speed * 0.8
            elif angleMag >= 0.1:
                speed = speed * 0.9
            else:
                speed = speed * 1.05
        else:
            if angleMag >= 0.35:
                speed = speed * 0.9
            elif angleMag >= 0.1:
                speed = speed * 0.95
            else:
                speed = speed * 1.1
        '''
    print('Speed =',speed)
    return speed

# Function to determine which section of the track the car is on
def findSection(maxY, cols, curr_section):
    section = curr_section
    if maxY[0] >= 145 and maxY[0] <= 170 and maxY[int(len(maxY)-1)] >= 145 and maxY[int(len(maxY)-1)] <= 170:
        section = 'straight'
    return section

def leftCurveAngle():
    return angle

def rightCurveAngle():
    return angle

def straightAngle():
    return angle

def curveSpeed():
    return speed

def straightSpeed():
    return speed

def driveInLane(image):
    return

def addTerminalToVideo(image, testData, frameNum):
    text = 'Columns to search on:\t' + str(testData[frameNum][0]) + '\n'
    text += 'Max Y values:\t\t' + str(testData[frameNum][1]) + '\n'
    text += 'Turn Direction = ' + str(testData[frameNum][2]) + '\n'
    text += 'Angle = ' + str(testData[frameNum][3]) + '\n'
    text += 'Speed = ' + str(testData[frameNum][4]) + '\n'
    text += 'Counter = ' + str(testData[frameNum][5]) + '\n'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    thickness = 1
    color = (255,255,255)
    origin = (0, croppedImageHeight + 100)
    cv2.putText(image, text, origin, font, fontScale, color, thickness, cv2.LINE_AA)
    return

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
def saveVideo(frames, filename, fps, isColor, testData):
    makeVid = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (imageWidth,croppedImageHeight+100), isColor)
    for i in range(0,len(frames)):
        #addTerminalToVideo(frames[i], testData, i)
        makeVid.write(frames[i])
    makeVid.release()

new = gpad.read()
frameTimeList = list()
frameList = list()
testData = list([])
isColor = False
path = './outputVideos/CSI_Front_Camera/'
filename = 'output'
filetype = '.avi'
try:
    LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    angle = 0
    speed = 0.066
    manual = False
    prev_speed = 0
    prev_angle = 0
    frameTime = 0
    # B button to cancel
    while gpad.B != 1:
        print('-----------------------------------------------------------------')
        startFrame = time.time()
        frontCam.read()
        counter += 1
        front = frontCam.image_data[croppedImageHeight:480, :]
        binaryf = BGR2Binary(front)
        cv2.imshow('Binary Front Image', binaryf)
        new = gpad.read()

        # x = 5, and x = 634 are near the edge of the screen but not exactly on the edge
        cols = [5, 160, 320, 480, 634]
        maxY = findLowestWhite(binaryf, cols)
        turn = findDirection(maxY)
        angle = setAngle(maxY, turn)
        speed = setSpeed(angle, prev_angle, prev_speed, counter, turn, manual)
        print('Counter:', counter)
        new = gpad.read()

        #speed = 0.066
        # activate controls
        mtr_cmd = np.array([speed, angle])
        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)
        prev_angle = angle
        prev_speed = speed
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

        # Data for testing and output
        testData.append([cols, maxY, turn, angle, speed, counter])

        '''text = 'Columns to search on:\t' + str(testData[counter][0]) + '\n'
        text += 'Max Y values:\t\t\t' + str(testData[counter][1]) + '\n'
        text += 'Turn Direction = ' + str(testData[counter][2]) + '\n'
        text += 'Angle = ' + str(testData[counter][3]) + '\n'
        text += 'Speed = ' + str(testData[counter][4]) + '\n'
        text += 'Counter = ' + str(testData[counter][5]) + '\n'
        print(text)'''
        frameList.append(binaryf)
        frameTime = endFrame - startFrame
        frameTimeList.append(frameTime)

except KeyboardInterrupt:
	print("User interrupted!")
finally:
    frontCam.terminate()
    gpad.terminate()
    myCar.terminate()
    fps = frameRate(frameTimeList)
    name = videoName(path, filename, filetype)
    print(name)
    saveVideo(frameList, name, fps, isColor, testData)
    plt.close() 
