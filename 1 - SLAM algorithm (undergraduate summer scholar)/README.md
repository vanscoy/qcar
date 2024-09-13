# Qcar SLAM
python code for use with the Quanser Q car.

[GitHub repository](https://github.com/DanielMiamiU/Qcar)

## SLAM Overview
SLAM stands for Simultaneous Localization and Mapping. SLAM algorithms are notorious for being a "chicken and egg" problem, knowing your position in a map that is empty is very difficult.  

## Implementation
We are solving the problem with a grid based fastSLAM algorithm. The map that is being created by this code is a grid of probabilities representing the probability of an object in that grid cell. A particle filter is used to represent the motion of the car. A particle filter uses several praticles, which are a representation of the cars position. These particles also have a probability assigned to them, representing the probability the particle represents the actual position of the car. As the car moves, encoder data is transcribed to distance and sent to the particle filter. The particle filter moves the particles and uses a random number to simulate error. LIDAR data is used to build a map of the surroundings and also determine which particles are closest to the real position.

## Running
To run the program, SSH into the robot and run sudo python3 mainMap.py. The robot saves the maps as a png every iteration of the loop and these can be retrieved from the robot in the same folder as the programs. Right trigger controlls the speed of the robot, left stick controlls the steering, B button quits the program. There is a 5 second delay to start in order to throw out bad LIDAR data where the robot will not move.


![QPMT0CTJASK88T44](https://github.com/DanielMiamiU/Qcar/assets/118695934/3fecaee0-9adf-4c7c-839b-83f0e08f69e6)


![IMG_3693](https://github.com/DanielMiamiU/Qcar/assets/118695934/a79d6516-02a3-4496-8bfb-5306524690a5)


Code is appropriated from a simulation for use in the real world on the Qcar, github for the simluation is below.
https://github.com/toolbuddy/2D-Grid-SLAM/tree/master

