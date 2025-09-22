from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
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
angle = 0
imageWidth = 1280
imageHeight = 720
max_distance = 5
robot_pos = np.array([0.0, 0.0, 0.0])

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Object Initialization
myCar = QCar()

## Gamepad Initialization
# use ls -l /dev/input/by-id to find the Logitech Gamepad
# F710 event number, add 1 to it, and use result as the controller number below
# if all else fails start with 1
gpad = gamepadViaTarget(1) 

## Movement instructions
# B to quit
# right joystick for motor speed, forwards and backwards
# left joystick to move left and right 
new = gpad.read()
try:
    while gpad.B != 1:
        start = time.time()
        counter += 1
        ## Movement and Gamepadxit
        mtr_cmd = np.array([.1*gpad.RLO, angle])
        #mtr_cmd = np.array([.25*(1-abs(.5*gpad.LLA), .25*gpad.LLA]) - Autonomous Code
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    ## Adjust tracker to increase or decrease angle
    ## Y - close to white
    ## X - middle of the lane
    ## A - close to yellow
    ## RIGHT - close to white
    ## UP - middle of the lane
    ## LEFT - close to yellow
    ## RB - set to straight
        if gpad.Y:
            angle = 0.18
        elif gpad.X:
            angle = 0.2
        elif gpad.A:
            angle = 0.22
        if gpad.up:
            angle = -0.33
        elif gpad.left:
            angle = -0.29
        elif gpad.right:
            angle = -0.36
        elif gpad.LB:
            angle = 0
	## Adjust LEDs based off steering
        if mtr_cmd[1] > 0.3:
            LEDs[0] = 1
            LEDs[2] = 1
        elif mtr_cmd[1] < -0.3:
            LEDs[1] = 1
            LEDs[3] = 1
        if mtr_cmd[0] < 0:
            LEDs[5] = 1

        new = gpad.read()
        os.system('clear')
		
        print("Angle:\t\t{0:.2f}\nRight Stick:\t\t\t{1:.2f}\nLeft Stick:\t\t\t{2:.2f}\nMotor Throttle:\t\t\t{3:4.2f}".format(30*mtr_cmd[1], gpad.RLO, gpad.LLA, mtr_cmd[0]))

        current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)

        # End timing this iteration
        end = time.time()
 

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        #cv2.imshow('My RGB', myCam1.image_buffer_RGB)
        #cv2.imshow('My Depth', myCam1.image_buffer_depth_m/max_distance)

        # Pause/sleep for sleepTime in milliseconds
        #cv2.waitKey(1)


        


except KeyboardInterrupt:
	print("User interrupted!")
finally:
    gpad.terminate()
    myCar.terminate()
    plt.close() 
