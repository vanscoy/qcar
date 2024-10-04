from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
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
angle = 0
speed = 0
distance = 0
imageWidth = 1280
imageHeight = 720
max_distance = 5
robot_pos = np.array([0.0, 0.0, 0.0])
begin = True

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
## Object Initialization
myCar = QCar()
mySpeed = speedCalc(robot_pos, myCar, time.time())

## Gamepad Initialization
# use ls -l /dev/input/by-id to find the Logitech Gamepad
# F710 event number, add 1 to it, and use result as the controller number below
# if all else fails start with 1
gpad = gamepadViaTarget(1) 

## Movement instructions
# B to quit
# A to start
new = gpad.read()
try:
    while begin:
        new = gpad.read()
        if gpad.A == 1:
            begin = False
            speed = 0.1

    while gpad.B != 1:
            start = time.time()
            counter += 1
            ## Movement and Gamepadxit
            mtr_cmd = np.array([speed, angle])
            #mtr_cmd = np.array([.25*(1-abs(.5*gpad.LLA), .25*gpad.LLA]) - Autonomous Code
            LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
	## Adjust LEDs based off steering
            if mtr_cmd[1] > 0.3:
                LEDs[0] = 1
                LEDs[2] = 1
            elif mtr_cmd[1] < -0.3:
                LEDs[1] = 1
                LEDs[3] = 1
            if mtr_cmd[0] < 0:
                LEDs[5] = 1
            ## 122 in = 0.1  = 3.1 m 
            ## turn is 2 m = 0.0645
            if distance > 0.125:
                angle = 0.195
                ## speed = 0
            if distance > 0.194:
                angle = 0.21
            if distance > 0.254:
                angle = 0
                ## speed = 0
            if distance > 0.36:
                angle = 0
                ## speed = 0

            distance += mySpeed.encoder_dist()

            new = gpad.read()
            os.system('clear')
		
            print("Angle:\t\t{0:.2f}\nRight Stick:\t\t\t{1:.2f}\nLeft Stick:\t\t\t{2:.2f}\nMotor Throttle:\t\t\t{3:4.2f}".format(30*mtr_cmd[1], gpad.RLO, gpad.LLA, mtr_cmd[0]))

            current, batteryVoltage, encoderCounts = myCar.read_write_std(mtr_cmd, LEDs)

            # End timing this iteration
            end = time.time()
            robot_pos = utils.posUpdate(robot_pos, end - start, mtr_cmd[1], mySpeed.encoder_speed())
            print(robot_pos)
            print(distance)
 

            # Calculate the computation time, and the time that the thread should pause/sleep for
            computationTime = end - start
            sleepTime = sampleTime - ( computationTime % sampleTime )



        
except KeyboardInterrupt:
	print("User interrupted!")
finally:
    gpad.terminate()
    myCar.terminate()
    plt.close() 
