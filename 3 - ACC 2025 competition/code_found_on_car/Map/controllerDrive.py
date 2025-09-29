from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D, LIDAR
from Quanser.q_interpretation import *
from speedCalc import *
import time
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import cv2
import utils

## Timing Parameters
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

## Miscellaneous parameters
sampleRate = 1000
sampleTime = 1/sampleRate
counter = 0
imageWidth = 1280
imageHeight = 720
max_distance = 5
robot_pos = np.array([0.0, 0.0, 0.0])

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Object Initialization
myCar = QCar()
myCam1 = Camera3D(mode='RGB&DEPTH', frame_width_RGB=imageWidth, frame_height_RGB=imageHeight)
myLidar = LIDAR(num_measurements=7200, max_distance=1.5)
mySpeed = speedCalc(robot_pos, myCar, time.time())
h = plt.polar()
plt.show(block=False)

## Gamepad Initialization
# use ls -l /dev/input/by-id to find the Logitech Gamepad
# F710 event number, add 1 to it, and use result as the controller number below
# if all else fails start with 1
gpad = gamepadViaTarget(1) 

## Movement instructions
# B to quit
# RT fro motor speed
# left joystick to move left and right (fixed angle)
new = gpad.read()
try:
    while gpad.B != 1:
        start = time.time()
        counter += 1
        ## Movement and Gamepad
        mtr_cmd = np.array([.05*gpad.RT, (gpad.left - gpad.right) * .3])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        new = gpad.read()

        os.system('clear')
		
        print("Right Trigger:\t\t{0:.2f}\nleft:\t\t\t{1:.0f}\nRight:\t\t\t{2:.0f}\nMotor Throttle:\t\t\t{3:4.2f}".format(gpad.RT, gpad.left, gpad.right, mtr_cmd[0]))

        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)

       
        # End timing this iteration
        end = time.time()
        robot_pos = utils.posUpdate(robot_pos, mtr_cmd[1], mySpeed.encoder_dist())
        print(robot_pos)

        #gyro = myCar.read_gyroscope()
        #velocity = myCar.read_accelerometer()
        #print(velocity)
        #print(gyro)
        ## Camera
        #myCam1.read_RGB()
        #myCam1.read_depth(dataMode='m') # for data in meters... 

        

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        #cv2.imshow('My RGB', myCam1.image_buffer_RGB)
        #cv2.imshow('My Depth', myCam1.image_buffer_depth_m/max_distance)

        # Pause/sleep for sleepTime in milliseconds
        #cv2.waitKey(1)

        ## LIDAR
        plt.cla()

        myLidar.read()
        if counter % 10 == 1:
            plt.scatter(myLidar.angles, myLidar.distances, marker='.')
            plt.draw()
            plt.pause(0.0001)
        


except KeyboardInterrupt:
	print("User interrupted!")
finally:
    gpad.terminate()
    myCar.terminate()
    plt.close() 
    myLidar.terminate()