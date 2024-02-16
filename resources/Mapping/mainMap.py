from Quanser.q_essential import LIDAR
import numpy as np
import random
import utils
import cv2
import time
import math
from GridMap import *
from ParticleFilter import *
# from SingleBotLaser2D import *
import copy

num_measurements=720
max_distance=5
myLidar = LIDAR(num_measurements, max_distance)

startTime = time.time()
sampleTime = 1/30
def elapsed_time():
    return time.time() - startTime

#
def SensorMapping(m, bot_pos, angles, dists):
    for i in range(num_measurements):
        if dists[i] > max_distance or dists[i] < .01:
            continue
        theta = bot_pos[2] + angles[i]
        m.GridMapLine(
        int(bot_pos[0]), 
        int(bot_pos[0]+dists[i]*np.cos(theta)),
        int(bot_pos[1]),
        int(bot_pos[1]+dists[i]*np.sin(theta))
        )
        #print(bot_pos[0]+dists[i]*np.cos(theta))
        #print(bot_pos[1]+dists[i]*np.sin(theta))
        
        
        
        

def AdaptiveGetMap(gmap):
    mimg = gmap.GetMapProb(
        gmap.boundary[0]-20, gmap.boundary[1]+20, 
        gmap.boundary[2]-20, gmap.boundary[3]+20 )
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg


if __name__ == '__main__':
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
    # Initialize GridMap
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [.4, -.4, 5.0, -5.0] 
    m = GridMap(map_param, gsize=.1)
    
    # TODO change env to Qcar friendly
    bot_pos = np.array([10.0, 10.0, 0.0])
    
    counter = 0
    while elapsed_time() < 30.0:
        myLidar.read()
        start = time.time()
        if myLidar.distances.any() !=0:
            #print(myLidar.distances)
            SensorMapping(m, bot_pos, myLidar.angles, myLidar.distances)
    
            mimg = AdaptiveGetMap(m)
            cv2.imshow('map',mimg)
            cv2.waitKey(1)

            #pf = ParticleFilter(bot_pos.copy(), bot_param, copy.deepcopy(m), 10)
        
        counter += 1
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Pause/sleep and print out the current timestamp
        time.sleep(sleepTime)


myLidar.terminate()