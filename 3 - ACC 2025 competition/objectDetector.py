from Quanser.q_essential import Camera2D
import time
import struct
import numpy as np 
import cv2
import math
from matplotlib import pyplot as plt

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
#grayTimeArr = np.zeros([1,1000], dtype=float)
#hsvTimeArr = np.zeros([1,1000], dtype=float)
grayTimeArr = []
hsvTimeArr = []
frames = []

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
            # this method changes less, but only draws flat rectangles
            """x1,y1,w,h = cv2.boundingRect(contour)
            cv2.rectangle(contourImage, (x1, y1), (x1+w, y1+h), (0,0,255), 2)"""

            """# espsilon param to adjust/control estimate, work to find diff values if needed give 0.01 from overflow
            epsilonParam = 0.01 * cv2.arcLength(contour, True)
            # estimation of contour to further simplify and reduce needed points in RBB 
            estimate = cv2.approxPolyDP(contour, epsilonParam, True)

            #FL - Reduces incorrect input of less than the five required points for minAreaRect
            if len(contours) >= 5 :
            # this method changes more often, but can draw rectangles at an angle
            # Intrduces convex hull for tightest possible fitting still using RBB and min

                # Convex Hull Bounding Boxes --- Created by connecting the outermost points of a set of objects, useful for objects with complex shapes
                hull = cv2.convexHull(estimate) # convex hull usage with estimate rather than orginal -- cleaner/quicker 
                area = cv2.minAreaRect(hull) 
                points = cv2.boxPoints(area)
                points = np.int0(points)

                # draw onto original image
                cv2.drawContours(imageContours, [points], 0, (0, 255, 0), 2)"""

            area = cv2.minAreaRect(contour)
            points = cv2.boxPoints(area)
            points = np.int0(points)
            # draw onto original image
            cv2.drawContours(imageContours, [points], 0, (0, 255, 0), 2)

    return imageContours

# function to convert an image to grayscale
def toGray(image, kernel_size=5):
    # converts a copy of the original image to grayscale
    image2 = image.copy()
    # convert bgr image to grayscale
    grayImage = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to reduce noise -- smoothes out grayscale fcn prior to threshold can change sizes dependent on needs
    blurredImage = cv2.GaussianBlur(grayImage, (kernel_size, kernel_size), 0)
    return blurredImage

def cannyLines(image, low_threshold, high_threshold):
    grayImage = toGray(image, 5)
    edges = cv2.Canny(grayImage, low_threshold, high_threshold)

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

def findLines(image):
    image2 = image.copy()
    grayImage = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(grayImage)[0]
    drawn_img = lsd.drawSegments(image2, lines)
    return drawn_img

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
        #"""
        grayAllTimeStart = time.time()
        grayscaleObjects = combineFeeds(detectGrayscale(left),
                                        detectGrayscale(back),
                                        detectGrayscale(right),
                                        detectGrayscale(front))
        grayAllTimeEnd = time.time()
        #"""
        grayAllTime = grayAllTimeEnd - grayAllTimeStart
        #grayTimeArr[0,int(counter)] = grayAllTime
        grayTimeArr.append(grayAllTime)
        frames.append(counter)
        
        # detect objects based on color in HSV format
        # detects white and yellow and shows on the same image
        # needs x function calls per cam to detect x number of colors
        #"""
        hsvAllTimeStart = time.time()
        leftHSV = detectHSV(left, 'white')
        leftHSV = detectHSV(left, 'yellow')
        backHSV = detectHSV(back, 'white')
        backHSV = detectHSV(back, 'yellow')
        rightHSV = detectHSV(right, 'white')
        rightHSV = detectHSV(right, 'yellow')
        frontHSV = detectHSV(front, 'white')
        frontHSV = detectHSV(front, 'yellow')
        hsvObjects = combineFeeds(leftHSV, backHSV, rightHSV, frontHSV)
        hsvAllTimeEnd = time.time()
        hsvAllTime = hsvAllTimeEnd - hsvAllTimeStart
        #hsvTimeArr[0,int(counter)] = hsvAllTime
        hsvTimeArr.append(hsvAllTime)
        #"""

        # test only the front cam
        #grayscaleObjects = detectGrayscale(front)
        #hsvObjects = detectHSV(front, 'white')

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Display the four images
        # HSV
        cv2.imshow('HSV Objects', hsvObjects)
        # Grayscale
        cv2.imshow('Grayscale Objects', grayscaleObjects)

        # Pause/sleep for sleepTime in milliseconds
        # show only every 20th frame on VNC
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

    """plt.plot(frames,grayTimeArr,'o',color='r')
    plt.plot(frames,hsvTimeArr,'o',color='b')
    plt.title('Computation time IP for each frame')
    plt.xlabel('Frame')
    plt.ylabel('Computation Time')
    plt.legend(['Gray Time', 'HSV Time'])
    plt.savefig("IP_runtime_10-6-24.jpg")
    plt.show()"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
