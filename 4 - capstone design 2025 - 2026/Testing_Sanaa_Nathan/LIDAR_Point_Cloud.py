from Quanser.q_essential import LIDAR
from Quanser.q_misc import Utilities
import time
import struct
import numpy as np 
import cv2

saturate = Utilities.saturate
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Timing Parameters and methods 
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 30
sampleTime = 1/sampleRate
simulationTime = 30.0
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Additional parameters and buffers
counter = 0
gain = 50 # pixels per meter
dim = 8 * gain # 8 meters width, or 400 pixels side length
decay = 0.9 # 90% decay rate on old map data
map = np.zeros((dim, dim), dtype=np.float32) # map object 

# LIDAR initialization and measurement buffers
myLidar = LIDAR(num_measurements=7200, max_distance=3.9)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Main Loop
try:
    while elapsed_time() < simulationTime:
        # Decay existing map
        map = decay*map

        # Start timing this iteration
        start = time.time()

        # Capture LIDAR data
        myLidar.read()

        # convert distances and angles to XY contour
        x = myLidar.distances*np.cos(myLidar.angles)
        y = myLidar.distances*np.sin(myLidar.angles)
        
        # convert XY contour to pixels contour and update those pixels in the map
        pX = (dim/2 - x*gain).astype(np.uint16)
        pY = (dim/2 - y*gain).astype(np.uint16)
        map[pX, pY] = 1

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Display the map at full resolution
        cv2.imshow('Map', map)
        
        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
        cv2.waitKey(msSleepTime)


except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Terminate the LIDAR object
    myLidar.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 