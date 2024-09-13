from Quanser.q_essential import LIDAR
from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
from speedCalc import *
import numpy as np
import utils
import cv2
import time
from GridMap import *
from ParticleFilter import *
import copy

# num_measurements is how many positions the lidar scans for
# max_distance = maximum lidar scan in meters
# mapUnits is used to convert the lidar distances from meters to different measurements (10 makes it dm)
# robot_pos is the robots coordinates and pose [x, y, theta]
num_measurements=360
max_distance= 2
mapUnits=20
num_particles = 5
robot_pos = np.array([0.0, 0.0, 0.0])

# Object initialization
myLidar = LIDAR(num_measurements, max_distance)
myCar = QCar()
gpad = gamepadViaTarget(1)
mySpeed = speedCalc(robot_pos, myCar, time.time())

# timing variables and helper function
startTime = time.time()
sampleTime = 1/50
def elapsed_time():
    return time.time() - startTime

# Changes the given map (m) based on robots position (bot_pos) and lidar scan results (angles and dists)
def SensorMapping(m, bot_pos, angles, dists): 
    for i in range(num_measurements):
        if dists[i] >= (max_distance) * mapUnits:
            continue
            """
            theta = bot_pos[2] + angles[i]
            m.EmptyMapLine(
            int(bot_pos[0]), 
            int(bot_pos[0]+dists[i]*np.cos(theta)),
            int(bot_pos[1]),
            int(bot_pos[1]+dists[i]*np.sin(theta))
            )
            """
            
        if dists[i] < .05:
            continue
        theta = bot_pos[2] - angles[i]
        m.GridMapLine(
        int(bot_pos[0]), 
        int(bot_pos[0]+dists[i]*np.cos(theta)),
        int(bot_pos[1]),
        int(bot_pos[1]+dists[i]*np.sin(theta))
        )

# Makes an image based on the given gridmap (gmap)
def AdaptiveGetMap(gmap):
    
    mimg = gmap.GetMapProb(
        gmap.boundary[0]-20, gmap.boundary[1]+20, 
        gmap.boundary[2]-20, gmap.boundary[3]+20  )
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)

    return mimg

def DrawParticle(img, plist, scale=1.0):
    for p in plist:
        cv2.circle(img, (int(p.gmap.center[0] + scale*p.pos[0]), int(p.gmap.center[1] + scale*p.pos[1])), int(2), (0,200,0), -1)
    return img

new = gpad.read()

if __name__ == '__main__':
    cv2.namedWindow('particle_map', cv2.WINDOW_AUTOSIZE)
    # Initialize GridMap
    # lo_occ, lo_free, lo_max, lo_min
    map_param = [.4, -.4, 5.0, -5.0] 
    m = GridMap(map_param, gsize=1)
    myLidar.read()
    encoder_Dist = 0
    print("Start: "+ str(elapsed_time()))
    # Makes a rudimentary map of the starting area
    # Cannot move the car while making this map
    counter = 0
    while (elapsed_time() < 5.0):
        myLidar.read()
    
    # Initialize the particle filter based on the map of the starting area
    SensorMapping(m, robot_pos, myLidar.angles, myLidar.distances * mapUnits)
    mimg = AdaptiveGetMap(m)
    pf = ParticleFilter(robot_pos.copy(), num_measurements, max_distance, mapUnits, copy.deepcopy(m), num_particles)

    counter = 0
    image_counter = 0
    # Main loop
    # Iterates until B is pressed on the gamepad
    while gpad.B != 1:
        
        new = gpad.read()
        start = time.time()

        # RT controls speed of the car, Left stick controls truning left or right
        mtr_cmd = np.array([.07*gpad.RT, (gpad.left - gpad.right) * .3])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        

        myCar.read_write_std(mtr_cmd, LEDs)
        #if counter % 10 == 0:
        encoder_Dist = mySpeed.encoder_dist()
        robot_pos = utils.posUpdate(robot_pos, mtr_cmd[1], mapUnits, encoder_Dist)
        myLidar.read()

        # Only update the particle filter when the car moves
        if (encoder_Dist > 0):
            #myCar.read_write_std((0,0), LEDs)
            print("Start PF: "+ str(elapsed_time()))
            pf.Feed( robot_pos[2], mtr_cmd[1], encoder_Dist, myLidar.angles, myLidar.distances * mapUnits)
            pf.Resampling(num_measurements, myLidar.angles, myLidar.distances * mapUnits)
            image_counter += 1
            print("End PF: "+ str(elapsed_time()))
                
                
        # Finds the most probable particle        
        mid = np.argmax(pf.weights)

        print(pf.particle_list[mid].pos)
        # Get an image from the most probable particle map
        imgp0 = AdaptiveGetMap(pf.particle_list[mid].gmap)
        imgp0 = DrawParticle(imgp0, pf.particle_list)
        
        cv2.imwrite('map' + str(counter) + '.jpg', imgp0)
        cv2.imshow('particle_map',imgp0)
        
        

        counter += 1
        end = time.time()
        
        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )
        
        # Pause/sleep

        print("End: "+ time.asctime( time.localtime(time.time()) ))
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1 # this check prevents an indefinite sleep as cv2.waitKey waits indefinitely if input is 0
        cv2.waitKey(msSleepTime)


myLidar.terminate()
gpad.terminate()
myCar.terminate()
