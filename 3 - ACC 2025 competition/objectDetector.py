from Quanser.q_essential import Camera2D
import time
import struct
import numpy as np 
import cv2
import math

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

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the CSI cameras
#myCam = Camera2D(camera_id=cameraID, frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
rightCam = Camera2D(camera_id="0", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
backCam = Camera2D(camera_id="1", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
leftCam = Camera2D(camera_id="2", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
frontCam = Camera2D(camera_id="3", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

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
    
    # Filter out unnecessary imaging noise occuring from varying objects in enviroment
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    _,contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected green objects
    for contour in contours:
        area = cv2.minAreaRect(contour)
        points = cv2.boxPoints(area)
        points = np.int0(points)
        cv2.drawContours(image, [points], 0, (0, 255, 0), 2)

    return image

# function to detect objects based on their brightness in a grayscale format
# draws contour boxes around object in original bgr image instead of in grayscale image to help with comparing to HSV
# does not use erosion or dilation
def detectGrayscale(image):
    # copy of image to be drawn on
    imageContours = image.copy()
    # convert bgr image to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # simple threshold
    # everything with a value above 100 becomes 255 (white)
    ret, binaryImage = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
    # adaptive threshold; should help with lighting
    # binaryImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 211, 30)
    _,contours,_ = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    for contour in contours:
            # this method changes less, but only draws flat rectangles
            """x1,y1,w,h = cv2.boundingRect(contour)
            cv2.rectangle(contourImage, (x1, y1), (x1+w, y1+h), (0,0,255), 2)"""

            # this method changes more often, but can draw rectangles at an angle
            area = cv2.minAreaRect(contour)
            points = cv2.boxPoints(area)
            points = np.int0(points)
            # draw onto original image
            cv2.drawContours(imageContours, [points], 0, (0, 255, 0), 2)

    return imageContours

# function to format how all four cameras are displayed at once
# each camera should not interfere with the others; this function merely organizes them
# # top left = left cam, top right = front cam, bottom left = back cam, bottom right = right cam
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

#Main Loop
try:
    while True:
        
        # Start timing this iteration
        start = time.time()

        # Capture RGB Image from CSI
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

        # detects objects based on brightness in a grayscale format
        grayscaleObjects = combineFeeds(detectGrayscale(left),
                                        detectGrayscale(back),
                                        detectGrayscale(right),
                                        detectGrayscale(front))
        
        # detect objects based on color in HSV format
        # detects white and yellow and shows on the same image
        # needs x function calls per cam to detect x number of colors
        """leftHSV = detectHSV(left, 'white')
        leftHSV = detectHSV(left, 'yellow')
        backHSV = detectHSV(back, 'white')
        backHSV = detectHSV(back, 'yellow')
        rightHSV = detectHSV(right, 'white')
        rightHSV = detectHSV(right, 'yellow')
        frontHSV = detectHSV(front, 'white')
        frontHSV = detectHSV(front, 'yellow')
        hsvObjects = combineFeeds(leftHSV, backHSV, rightHSV, frontHSV)"""

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Display the four images
        # HSV
        #cv2.imshow('HSV Objects', hsvObjects)
        # Grayscale
        cv2.imshow('Grayscale Objects', grayscaleObjects)
        
        # Pause/sleep for sleepTime in milliseconds
        # show only every 20th frame on VNC
        if counter % 20 == 0:
            msSleepTime = int(20000*sleepTime)
            if msSleepTime <= 0:
                msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
            cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Terminate all webcam objects    
    leftCam.terminate()
    backCam.terminate()
    rightCam.terminate()
    frontCam.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
