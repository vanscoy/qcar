from Quanser.q_essential import Camera3D
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
simulationTime = 20.0
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
max_distance = 5 # maximum depth distance
                 # distances beyond will display as white pixels 

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    while elapsed_time() < simulationTime:
        # Start timing this iteration
        start = time.time()

        # Capture RGB and depth data
        myCam1.read_RGB()
        myCam1.read_depth(dataMode='m') # for data in meters... 
        # myCam1.read_depth(dataMode='px') # for data in pixel range 0-255

        counter += 1
        
        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Display the two images
        cv2.imshow('My RGB', myCam1.image_buffer_RGB)
        cv2.imshow('My Depth', myCam1.image_buffer_depth_m/max_distance) # by default, ranges between 0 to 1 meters will show up in grayscale. Beyond 1 m, everything will look white. 
                                                            # apply a gain of 0.5 to improve range to 2 meters, 0.33 for 3 meters and so on
        # cv2.imshow('My Depth', 30*myCam1.image_buffer_depth_px) # apply a gain of 30 to visualize data if using pixels

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:    
    # Terminate RealSense camera object
    myCam1.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
