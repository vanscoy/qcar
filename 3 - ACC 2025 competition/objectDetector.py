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
#cameraID = '3'

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the CSI cameras
#myCam = Camera2D(camera_id=cameraID, frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
rightCam = Camera2D(camera_id="0", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
backCam = Camera2D(camera_id="1", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
leftCam = Camera2D(camera_id="2", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)
frontCam = Camera2D(camera_id="3", frame_width=imageWidth, frame_height=imageHeight, frame_rate=sampleRate)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

# function to detect color
def detect_color(image, color):
    if color == 'white':
        lower = np.array([0,0,127])
        upper = np.array([180,25,255])
        #frame = detect_white(image, lower, upper)
        #return frame
    elif color == 'red':
        lower = np.array([0, 100, 100])
        upper = np.array([20, 255, 255])
        #return detect_white(image, lower, upper)
    elif color == 'yellow':
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
        #frame = detect_white(image, lower, upper)
        #return frame
    elif color == 'green':
        lower = np.array([40, 30, 100])
        upper = np.array([80, 255, 255])
        #return detect_white(image, lower, upper)
    elif color == 'blue':
        lower = np.array([100, 150, 50])
        upper = np.array([140, 255, 255])
        #return detect_white(image, lower, upper)
    else:
        print('Pick a different color')
        return image

    ty = 315
    tp = 300

    # Need to set to HSV for proper color usage 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # for number of masks:
    # Create a mask set specific to white color bounds
    mask = cv2.inRange(hsv, lower, upper)
    
    # Filter out unnecessary imaging noise occuring from varying objects in enviroment
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    _,contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 4000
    
    # Initalize temps for checking previously set position of box
    prev_ty, prev_tp = 0, 0
    
    # Draw bounding boxes around detected green objects
    for contour in contours:
        area = cv2.minAreaRect(contour)
        points = cv2.boxPoints(area)
        points = np.int0(points)
        cv2.drawContours(image, [points], 0, (0, 255, 0), 2)

    return image

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

        # Morpho stuff; maybe need gradescale
        #binr = cv2.threshold(myCam.image_data, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        #kernel = np.ones((3,3), np.uint8)
        #opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations = 1)
        counter += 1

        # int(imageHeight/2):480 height for all cams
        leftCropped = leftCam.image_data[int(imageHeight/2):480, 0:640]
        backCropped = backCam.image_data[int(imageHeight/2):480, 0:640]
        rightCropped = rightCam.image_data[int(imageHeight/2):480, 0:640]
        frontCropped = frontCam.image_data[int(imageHeight/2):480, 0:640]

        horizontalBlank = np.zeros((20, 2*imageWidth+60, 3), dtype=np.uint8)
        verticalBlank = np.zeros((int(imageHeight/2), 20, 3), dtype=np.uint8)

        # note: adjust ranges; lighting matters
        detect_color(frontCropped, 'white') 
        #red_boxes = detect_color(myCam.image_data, 'red') # detects tan instead
        #yellow_boxes = detect_color(myCam.image_data, 'yellow') # widen range
        #green_boxes = detect_color(myCam.image_data, 'green') # range too dark
        #blue_boxes = detect_color(myCam.image_data, 'blue') # range too dark

        allCams = np.concatenate((horizontalBlank,
                                  np.concatenate((verticalBlank,
                                                  leftCropped,
                                                  verticalBlank,
                                                  frontCropped,
                                                  verticalBlank),
                                                  axis = 1),
                                    horizontalBlank,
                                    np.concatenate((verticalBlank,
                                                    backCropped,
                                                    verticalBlank,
                                                    rightCropped,
                                                    verticalBlank),
                                                    axis = 1),
                                    horizontalBlank),
                                    axis=0)

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Display the four images
        # note: might still need grayscale; boxes drawn on both frames
        #cv2.imshow('Left', leftCropped)
        #cv2.imshow('Back', backCropped)
        #cv2.imshow('Right', rightCropped)
        #cv2.imshow('Front', frontCropped)
        cv2.imshow('All Cams', allCams)
        #cv2.imshow('White', white_boxes)
        #cv2.imshow('Yellow', yellow_boxes)
        # morpho stuff
        #cv2.imshow(opening, cmap='gray')
        
        # Pause/sleep for sleepTime in milliseconds
        # show only every 10th frame on VNC
        if counter % 10 == 0:
            msSleepTime = int(10000*sleepTime)
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
