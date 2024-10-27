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

# Next steps, refine process and how to continue from stop sign --> create more legitamte coloring and sizing 
# AI involvment? --> clean up code, figure out best way to run live

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
#speed = 0.066

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
'''
def detectHSV(image, color):
    
    mtr_cmd = np.array([speed, angle])
    if color == 'red':                    # detects tan instead
        lower = np.array([0, 100, 100])
        upper = np.array([20, 255, 255])
    elif color == 'yellow':                 # widen range
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif color == 'green':                  # range too dark
        lower = np.array([40, 30, 100])
        upper = np.array([80, 255, 255])
        print('')
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
    
    # Draw bounding boxes around detected green objects
    for contour in contours:
        area = cv2.minAreaRect(contour)
        points = cv2.boxPoints(area)
        points = np.int0(points)
        cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
        speed = 2

    return image'''

# function for red stop sign
def detectStopSign(image, showImage=False):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    redMask = cv2.inRange(hsv_image, lower_red, upper_red)

    kernel = np.ones((15, 15), np.uint8)  # Increase kernel size
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_CLOSE, kernel)
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)

    '''lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)'''

    # Combine masks
    #redMask = cv2.bitwise_or(mask1, mask2)
    '''
    _,contours,_ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected green objects
    for contour in contours:
        area = cv2.minAreaRect(contour)
        points = cv2.boxPoints(area)
        points = np.int0(points)
        cv2.drawContours(image, [points], 0, (0, 255, 0), 2) '''

        
    # Function to filter contours based on area
    def get_largest_contour_area(mask):
        _,contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            #points = cv2.boxPoints(area)
            #points = np.int0(points)
            #cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
            if area > largest_area:
                largest_area = area
                rect = cv2.minAreaRect(contour)
                points = cv2.boxPoints(rect)
                points = np.int0(points)
            # draw onto original image
                cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
        return largest_area

    red_area = get_largest_contour_area(redMask)

    area_threshold = 500  # Set a minimum area threshold

    
    # Morphological operations to remove noise
    #kernel = np.ones((5, 5), np.uint8)
    #morphed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    #morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_OPEN, kernel)

    #cv2.imshow('StopSign', redMask)
     # Show the images if requested
    if showImage:
        cv2.imshow('Stop Sign Bouding Box', image)
        #cv2.imshow('Stop Sign Red', redMask)
        #cv2.waitKey(1)  # Wait for a brief moment

    # Check for non-zero areas in the morphed mask
    #if np.count_nonzero(redMask) > 0:
    
    if red_area > area_threshold:
        return 1  # stop sign detected
    else:
        return 0  # No stop sign detected

    #return morphed_mask

# Function for stop lights
def detectStopLight(image, showImage = False):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Define range for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Define range for green color in HSV
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)


    kernel = np.ones((15, 15), np.uint8)  # Increase kernel size
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Function to filter contours based on area
    def get_largest_contour_area(mask):
        _,contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            #points = cv2.boxPoints(area)
            #points = np.int0(points)
            #cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
            if area > largest_area:
                largest_area = area
                rect = cv2.minAreaRect(contour)
                points = cv2.boxPoints(rect)
                points = np.int0(points)
            # draw onto original image
                cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
        return largest_area
    

    red_area = get_largest_contour_area(red_mask)
    yellow_area = get_largest_contour_area(yellow_mask)
    green_area = get_largest_contour_area(green_mask)

    area_threshold = 500  # Set a minimum area threshold

    if showImage:
        cv2.imshow('Bouding Boxes SL', image)
        #cv2.imshow('Red Mask SL', red_mask)
        #cv2.imshow('Yellow Mask SL', yellow_mask)
        #cv2.imshow('Green Mask SL', green_mask)
        #cv2.waitKey(1)  # Wait for a brief moment

    if red_area > area_threshold:
        return 1  # Red light detected
    elif yellow_area > area_threshold:
        return 2  # Yellow light detected
    elif green_area > area_threshold:
        return 3  # Green light detected
    else:
        return 0  # No stop light detected




    # Combine masks
    #combined_mask = cv2.bitwise_or(red_mask, yellow_mask)
    #combined_mask = cv2.bitwise_or(combined_mask, green_mask)

    # Morphological operations to remove noise
    #kernel = np.ones((5, 5), np.uint8)
    #morphed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    #morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_OPEN, kernel)

    #_,contours,_ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected green objects
    '''for contour in contours:
        area = cv2.minAreaRect(contour)
        points = cv2.boxPoints(area)
        points = np.int0(points)
        cv2.drawContours(image, [points], 0, (0, 255, 0), 2)
    # Morphological operations to remove noise'''

    '''
    #return morphed_mask if need image
    if np.count_nonzero(red_mask) > 0:
        return 1  # Red light detected
    elif np.count_nonzero(yellow_mask) > 0:
        return 2  # Yellow light detected
    elif np.count_nonzero(green_mask) > 0:
        return 3  # Green light detected
    else:
        return 0  # No stop light detected
    '''




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
        backCam.read()
        counter += 1
        front = frontCam.image_data[croppedImageHeight:480, :]
        backHSV = backCam.image_data[croppedImageHeight:480, :].copy()
        binaryf = detectGrayscale(front)
        #maskSL = detectStopLight(frontHSV) 
        #maskSS = detectStopSign(front) #stop sign 
        detectedSL = detectStopLight(front, showImage = True)
        detectedSS = detectStopSign(front, showImage=True)

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
        
        #hsvObjects = combineFeeds(maskSS, maskSL)
        
        #cv2.imshow('Binary Front Image', binaryf)
        #cv2.imshow('Envriomental Camera front', maskSL)
        #cv2.imshow('Envriomental Camera front', hsvObjects)

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
        #
        print('Angle =',angle)

        ## Movement and Gamepadxit
        # right trigger for speed
        #speed = 0.066*gpad.RT
        #speed = 0.066
        #mtr_cmd = np.array([.066, angle]) # need to replace with varius input on encoders and speeds
        
        # Stop sign check
        if detectedSS == 0:
            speed = 0.066
            print('Stop Sign not detected!')
            #cv2.waitKey(1)
        elif detectedSS == 1:
            speed = 0
            print('Stop Sign detected!')
            #cv2.waitKey(1)
            #speed = 0.066
            #wait(1)

        # stop light check
        if detectedSL == 0:
            speed = 0.066
            print('No Stop Light')
        elif detectedSL == 1:
            print('Stop light red detected')
            speed = 0
            #cv2.waitKey(1)
            #speed = 0.066
        elif detectedSL == 2:
            print('Stop light yellow detected')
            speed = 0.06
            
        elif detectedSL == 3:
            print('Stop light green detected')
            speed = 0.066

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
