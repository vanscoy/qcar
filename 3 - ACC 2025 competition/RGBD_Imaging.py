from Quanser.q_essential import Camera3D
import Quanser.q_interpretation as qi 
import time
import struct
import numpy as np 
import cv2

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 30.0
sampleTime = 1/sampleRate
simulationTime = 30.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
counter = 0
imageWidth = 1280
imageHeight = 720

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Initialize the RealSense camera for RGB and Depth data
myCam1 = Camera3D(mode='RGB&DEPTH', frame_width_RGB=imageWidth, frame_height_RGB=imageHeight)
counter = 0
# max_distance_view = 5
max_distance = 3 # pixels in RGB image farther than this will appear white
## Distance = 3 ~ 7-8 feet from front of the vehicle
min_distance = 1 # pixels in RGB image closer than this will appear black
## Distance = 1 ~ 2.5 feet from front of the vehicle

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

## Main Loop
flag = True
try:
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = time.time()

        # Read the RGB and Depth data (latter in meters)
        myCam1.read_RGB()
        myCam1.read_depth(dataMode='m')
        
        # Threshold the depth image based on min and max distance set above, and cast it to uint8 (to be used as a mask later)
        binary_now = qi.binary_thresholding(myCam1.image_buffer_depth_m, min_distance, max_distance).astype(np.uint8)
        
        # Initialize binary_before to keep a 1 step time history of the binary to do a temporal difference filter later. 
        # At the first time step, flag = True. Initialize binary_before and then set flag = False to not do this again.
        if flag:
            binary_before = binary_now
            flag = False
        
        # clean  =  closing filter applied ON ( binary_now BITWISE AND ( BITWISE NOT of ( the ABSOLUTE of ( difference between binary now and before ) ) ) )
        binary_clean = qi.image_filtering_close(cv2.bitwise_and( cv2.bitwise_not(np.abs(binary_now - binary_before)/255), binary_now/255 ), dilate=3, erode=1, total=1)

        # grab a smaller chunk of the depth data and scale it back to full resolution to account for field-of-view differences and physical distance between the RGB/Depth cameras.
        binary_clean = cv2.resize(binary_clean[81:618, 108:1132], (1280, 720)).astype(np.uint8)

        # Apply the binary_clean mask to the RGB image captured, and then display it.
        masked_RGB = cv2.bitwise_and(myCam1.image_buffer_RGB, myCam1.image_buffer_RGB, mask=binary_clean)
        cv2.imshow('Original', cv2.resize(masked_RGB, (640, 360)))
        
        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)
        binary_before = binary_now

except KeyboardInterrupt:
    print("User interrupted!")

finally:    
    # Terminate RealSense camera object
    myCam1.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 