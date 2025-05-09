#########################################################################################################################
# This is the driver for the controls of the car, created and edited by Zach Copenhaver. 
# This class converts an image into a binary mask and processes it to approximate the current location of the
# car in relation to the road lines. It defines how the car should execute different motions in different directions and 
# on different parts of the road. It also contains functions to define how the sequence of controls for each edge.
#########################################################################################################################

import cv2
import numpy as np
import statistics
from Quanser.q_ui import gamepadViaTarget
from Quanser.q_misc import Calculus
from speedCalc import *

# x values of image to search on
cols = np.array([100, 500, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400])

# Main controls class
class LineFollower:
    # initialize controls with the starting edge
    def __init__(self, stateState):
        self.gpad = gamepadViaTarget(1)
        self.stateCount = 0
        self.prevNode = stateState
        self.battery = 0
        self.prevData = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.adjusting = False

    # Convert the given image to a binary mask using a given threshold
    def bgr2Binary(self, image, thresh):
        """Convert BGR image to binary using thresholding and morphological operations."""
        if image is None or image.size == 0:
            print("❌ Error: Empty image received.")
            return None  # or return an empty binary image if needed
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    # Take certain columns of the image as vectors. 
    # Return an array containing the index of the last white pixel (1 or 255) in each column.
    def findLastWhite(self, image):
        """Finds the lowest white pixel in each specified column."""
        maxY = np.zeros(len(cols))
        for y in range(0, image.shape[0]):
            for x, col in enumerate(cols):
                if image[y][col] == 255:
                    maxY[x] = y
        return maxY

    # Set the speed of the car
    def getSpeed(self, angle, manual):
        speed = 0.09
        k_ff, k_p, k_i = 0.1, 0.1, 0.1
        speed_thresh = 0
        if manual == True:
            # drive car manually with gamepad
            new = self.gpad.read()
            return 0.066*self.gpad.RT
        speed_thresh = 0.4
        difference = speed_thresh - speed
        integrator_1 = Calculus().integrator_variable(.001)
        next(integrator_1)
        speed = k_ff * speed_thresh + (k_p * difference + integrator_1.send((k_i*difference, .001)))
        #     ------------------- + --------------------------------------------------------------
        #     feedfroward term    +                     feedback term
        # The speed of the car sometimes changes between runs even though the program is sending 
        # the same value to the motor.
        # Adjust for this by manually setting the speed.
        # speed = .066 # this speed seems to work the most consistently
        return speed
    
    # distance is the value we use to turn. It represents how far the car is from the line it is following
    # Left: direction = 1; Right: direction = -1
    def getAngle(self, distance, direction, high, low):
        angle = 0
        # car is close to line so turn away from line
        if distance >= high:
            angle = .02*direction*(abs(distance - high))
        # car is far from line so turn towards line
        elif distance < low:
            angle = -.02*direction*(abs(low - distance))
        angle = np.clip(angle, -0.5, 0.5)
        return angle
    
    # Left turn / curve. Follow the right line.
    # Also works for straights when right line is visible.
    def left(self, maxY, high=155, low=130): # def left(self, maxY, high, low)
        distance = maxY[1]
        direction = 1
        angle = self.getAngle(distance, direction, high, low)
        speed = self.getSpeed(angle, False)
        return speed, angle
    
    # Right turn / curve. Follow the left line.
    def right(self, maxY, high=165, low=140):
        distance = maxY[0]
        direction = -1
        angle = self.getAngle(distance, direction, high, low)
        speed = self.getSpeed(angle, False)
        return speed, angle
    
    # Left turn following a dotted line
    # If there is a gap in the line (i.e. the index is too low), assume a hardcoded value
    def leftDotted(self, maxY):
        if maxY[1] <= 120:
            maxY[1] = 160
        speed, angle = self.left(maxY)
        return speed, angle

    # Right turn following a dotted line
    # If there is a gap in the line (i.e. the index is too low), assume a hardcoded value
    def rightDotted(self, maxY):
        if maxY[0] <= 130:
            maxY[0] = 170
        speed, angle = self.right(maxY)
        return speed, angle

    # Left turn through intersection. Switch to left camera and follow the far line
    def leftNoLine(self, maxY, direction, high, low):
        angle = 0
        mean = statistics.mean(maxY[2:])
        var = statistics.variance(maxY[2:])
        # if too much variance then filter out both high and low noise
        if var > 100:
            # find outliers
            out_index = list()
            no_outs = list()
            for i in range(len(maxY[2:])):
                if maxY[i] > 75 or maxY[i] < 20:
                    out_index.append(i)
                else:
                    no_outs.append(maxY[i])
            # find mean without outliers
            if len(no_outs) < 1:
                no_outs.append(mean)
            mean = statistics.mean(no_outs)
        distance = mean
        angle = self.getAngle(distance, direction, high, low)
        speed = self.getSpeed(angle, False)
        return speed, angle

    # Right turn through intersection. Switch to right camera and follow the close line
    def rightNoLine(self, maxY, direction=1, high=155, low=135):
        angle = 0
        mean = statistics.mean(maxY[2:])
        var = statistics.variance(maxY[2:])
        # if too much variance then filter out both high and low noise
        if var > 100:
            # find outliers
            out_index = list()
            no_outs = list()
            for i in range(len(maxY[2:])):
                if maxY[i] > 190:
                    out_index.append(i)
                else:
                    no_outs.append(maxY[i])
            # find mean without outliers
            if len(no_outs) < 1:
                no_outs.append(mean)
            mean = statistics.mean(no_outs)
        # maxY is lower when further away and higher when closer
        distance = mean
        angle = self.getAngle(distance, direction, high, low)
        speed = self.getSpeed(angle, False)
        return speed, angle
    
    # Drive straight without following a line
    def straight(self):
        angle = 0
        speed = self.getSpeed(angle, False)
        return speed, angle
    
    # Do not move
    def stop(self):
        angle = 0
        speed = 0
        return speed, angle
    
    # Use the current node to determine what edge, or state, the car is in.
    def getEdge(self, path, pathPos, currNode):
        if path[-1] == currNode:
            return "z"
        edge = str(path[pathPos]) + str(path[pathPos+1])
        return edge

    # Execute a state based on the edge.
    def next_path(self, edge, maxY):
        # each of the edge functions will the sequence and duration of each motion function
        if edge == "12": return self.edge12(maxY) 
        if edge == "23": return self.edge23(maxY) 
        if edge == "34": return self.edge34(maxY) 
        if edge == "316": return self.edge316(maxY)
        if edge == "45": return self.edge45(maxY)
        if edge == "56": return self.edge56(maxY)
        if edge == "522": return self.edge522(maxY)
        if edge == "67": return self.edge67(maxY)
        if edge == "78": return self.edge78(maxY)
        if edge == "71": return self.edge71(maxY)
        if edge == "89": return self.edge89(maxY)
        if edge == "812": return self.edge812(maxY)
        if edge == "818": return self.edge818(maxY)
        if edge == "93": return self.edge93(maxY)
        if edge == "1012": return self.edge1012(maxY)
        if edge == "1018": return self.edge1018(maxY)
        if edge == "1023": return self.edge1023(maxY)
        if edge == "1213": return self.edge1213(maxY)
        if edge == "124": return self.edge124(maxY)
        if edge == "1310": return self.edge1310(maxY)
        if edge == "1513": return self.edge1513(maxY) 
        if edge == "1516": return self.edge1516(maxY)
        if edge == "1618": return self.edge1618(maxY)
        if edge == "169": return self.edge169(maxY)
        if edge == "1623": return self.edge1623(maxY)
        if edge == "1819": return self.edge1819(maxY)
        if edge == "186": return self.edge186(maxY)
        if edge == "1915": return self.edge1915(maxY)
        if edge == "2022": return self.edge2022(maxY)
        if edge == "2019": return self.edge2019(maxY)
        if edge == "2223": return self.edge2223(maxY)
        if edge == "229": return self.edge229(maxY)
        if edge == "2212": return self.edge2212(maxY)
        if edge == "2320": return self.edge2320(maxY)
        return self.end()
    
    # define the binary threshold used for each camera
    def getBinaryThresh(self, camID):
        if camID == 0: return 130
        elif camID == 1: return 130
        elif camID == 2: return 100
        elif camID == 3: return 130
        else: return 130
    
    # Main control function that processes the image and executes the state.
    def get_control(self, frame, dataList, count, currNode, nextNode, path, pathPos, camID):
        """Processes camera input and outputs (speed, angle) for QCar."""
        if frame is None or frame.size == 0:
            print("❌ LineFollower: Received empty frame!")
            return np.zeros((1,data.length())), frame  # Skip control if no valid image. orginally returned OG values

        # convert image to a binary mask
        binary = self.bgr2Binary(frame, self.getBinaryThresh(camID))
        if binary is None:
            return np.zeros((1,data.length())), frame  # Skip control if binary conversion failed
        
        # obtain the variables of the previous frame
        if count >= 1:
            self.prevData = dataList[-1]
            self.battery = dataList[-1][-1]
        else:
            self.battery = 12

        # increment a counter each frame
        self.stateCount += 1

        # process image and obtain approximation of distance to line
        maxY = self.findLastWhite(binary)

        # reset counter at the start of each state
        if self.prevNode != currNode:
            self.stateCount = 0

        # Determine and execute controls for current state
        edge = self.getEdge(path, pathPos, currNode)
        speed, angle = self.next_path(edge, maxY)

        # Stop car if at the end of the path
        if pathPos >= len(path):
            speed, angle = self.stop()

        # save variables for this frame
        self.prevNode = currNode
        lineMean = statistics.mean(maxY[2:])
        data = [count, speed, angle, maxY, edge, self.prevNode, currNode, nextNode, self.stateCount, lineMean]
        return data, binary
    
    # Functions for each state:

    # left
    def edge12(self, maxY):
        # follow white line
        speed, angle = self.left(maxY)
        return speed, angle

    # left with line
    def edge23(self, maxY):
        # stop at dotted line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow white line
            speed, angle = self.left(maxY)
        return speed, angle
    
    # straight with line
    def edge34(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow white line
            speed, angle = self.left(maxY)
        return speed, angle
    
    # Left no line
    def edge316(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow white line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=53)
        return speed, angle
    
    # left with line
    def edge45(self, maxY):
        # sometimes the car does not complete the turn before changing nodes so use large turns to adjust heading
        if maxY[1] == 0:
            self.adjusting = True
        if self.adjusting == True and self.stateCount <= 20:
            angle = 0.5
            speed = self.getSpeed(angle, False)
        else:
            # follow white line
            speed, angle = self.left(maxY)
            self.adjusting == False
        return speed, angle
    
    # straight with line
    def edge56(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.left(maxY)
        return speed, angle
    
    # Left no line
    def edge522(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=57)
        return speed, angle
    
    # left with line
    def edge67(self, maxY):
        # follow white line
        speed, angle = self.left(maxY)
        return speed, angle
    
    # Left dotted turn
    def edge78(self, maxY):
        # follow line
        speed, angle = self.leftDotted(maxY)
        return speed, angle
    
    # left with line (straight with line)
    def edge71(self, maxY):
        # follow white line
        speed, angle = self.left(maxY)
        return speed, angle
    
    # right no line
    def edge89(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # straight no line
    def edge812(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        # if the straight white line to the left is seen (i.e. if car is in the lane), follow it to correct heading
        elif maxY[0] > 120:
            speed, angle = self.right(maxY)
        # if the curved white line to the right is seen, follow it
        elif maxY[1] > 120 and self.stateCount > 40:
            speed, angle = self.left(maxY)
        # when no line is seen, adjust heading every ~5 sec by turning 0.05 rad left
        elif self.stateCount % 5 == 0:
            angle = 0.2
            speed = self.getSpeed(angle, False)
        # when no line is seen, drive straight
        else:
            speed, angle = self.straight()
        return speed, angle
    
    # Left no line
    def edge818(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=53)
        return speed, angle
    
    # Left dotted turn
    def edge93(self, maxY):
        # sometimes the car does not complete the turn before changing nodes so use large turns to adjust heading
        if maxY[1] == 0:
            self.adjusting = True
        if self.adjusting == True and self.stateCount <= 5:
            angle = 0.5
            speed = self.getSpeed(angle, False)
        # correct alignment while following line
        elif self.stateCount <= 10:
            speed, angle = self.left(maxY)
            self.adjusting = False
        # follow line assuming line is dotted
        else:
            speed, angle = self.leftDotted(maxY)
            self.adjusting = False
        return speed, angle
    
    # right no line
    def edge1012(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # straight no line
    def edge1018(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        # briefly follow line to adjust heading and complete previous turn
        elif self.stateCount <= 16:
            speed, angle = self.left(maxY)
        # if the straight white line to the left is seen (i.e. if car is in the lane), follow it to correct heading
        elif maxY[0] > 120:
            speed, angle = self.right(maxY)
        # if the curved white line to the right is seen, follow it
        elif maxY[1] > 120:
            speed, angle = self.left(maxY)
        # when no line is seen, adjust heading every ~5 sec by turning 0.05 rad left
        elif self.stateCount % 50 == 0:
            angle = -0.025
            speed = self.getSpeed(angle, False)
        # when no line is seen, drive straight
        else:
            speed, angle = self.straight()
        return speed, angle
    
    # Left no line
    def edge1023(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        elif self.stateCount <= 30:
            # follow line at a different threshold then later
            speed, angle = self.leftNoLine(maxY, direction=-1, high=55, low=35)
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=53)
        return speed, angle
    
    # right no line
    def edge1213(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # Left no line
    def edge124(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=70, low=53)
        return speed, angle
    
    # right dotted line
    def edge1310(self, maxY):
        # follow line
        speed, angle = self.right(maxY)
        return speed, angle
    
    # straight no line
    def edge1513(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        # follow line at start to complete previous turn
        elif self.stateCount <= 16:
            speed, angle = self.left(maxY)
            angle = -0.22
        # if the yellow line to the left is seen, follow it
        elif maxY[0] > 120 and self.stateCount > 30:
            speed, angle = self.right(maxY)
        # if the curved white line to the right is seen, follow it
        elif maxY[1] > 120 and self.stateCount > 30:
            speed, angle = self.left(maxY)
        # when no line is seen, adjust heading every 5 counts by turning 0.05 rad left
        elif self.stateCount % 17 == 0:
            angle = -0.225
            speed = self.getSpeed(angle, False)
        # when no line is seen, drive straight
        else:
            speed, angle = self.straight()
        return speed, angle
    
    # right no line
    def edge1516(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # right no line
    def edge1618(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # Left no line
    def edge169(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=53)
        return speed, angle
    
    # straight no line
    def edge1623(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        # if the straight white line to the left is seen (i.e. if car is in the lane), follow it to correct heading
        elif maxY[0] > 120:
            speed, angle = self.left(maxY)
        # if the curved white line to the right is seen, follow it
        elif maxY[1] > 120:
            speed, angle = self.right(maxY)
        # when no line is seen, adjust heading every ~5 sec by turning 0.05 rad left
        elif self.stateCount % 40 == 0:
            angle = -0.025
            speed = self.getSpeed(angle, False)
        # when no line is seen, drive straight
        else:
            speed, angle = self.straight()
        return speed, angle
    
    # right no line
    def edge1819(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # Left no line
    def edge186(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=53)
        return speed, angle
    
    # right with line
    def edge1915(self, maxY):
        # follow line
        speed, angle = self.right(maxY)
        return speed, angle
    
    # right no line
    def edge2022(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # straight no line
    def edge2019(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        # follow line at start to complete previous turn
        elif self.stateCount <= 19:
            speed, angle = self.left(maxY)
            angle = -0.22
        # if the yellow line to the left is seen, follow it
        elif maxY[0] > 120 and self.stateCount > 30:
            speed, angle = self.right(maxY)
        # if the curved white line to the right is seen, follow it
        elif maxY[1] > 120 and self.stateCount > 30:
            speed, angle = self.left(maxY)
        # when no line is seen, adjust heading every 5 counts by turning 0.05 rad left
        elif self.stateCount % 13 == 0:
            angle = -0.22
            speed = self.getSpeed(angle, False)
        # when no line is seen, drive straight
        else:
            speed, angle = self.straight()
        return speed, angle
    
    # right no line
    def edge2223(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.rightNoLine(maxY)
        return speed, angle
    
    # straight no line
    def edge229(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        # if the straight white line to the left is seen (i.e. if car is in the lane), follow it to correct heading
        elif maxY[0] > 120:
            speed, angle = self.right(maxY)
        # if the curved white line to the right is seen, follow it
        elif maxY[1] > 120:
            speed, angle = self.left(maxY)
        # when no line is seen, adjust heading every ~5 sec by turning 0.05 rad left
        elif self.stateCount % 50 == 0:
            angle = -0.025
            speed = self.getSpeed(angle, False)
        # when no line is seen, drive straight
        else:
            speed, angle = self.straight()
        return speed, angle
    
    # Left no line
    def edge2212(self, maxY):
        # stop at line
        if self.stateCount <= 10:
            speed, angle = self.stop()
        else:
            # follow line
            speed, angle = self.leftNoLine(maxY, direction=-1, high=75, low=57)
        return speed, angle
    
    # right dotted
    def edge2320(self, maxY):
        # follow line
        speed, angle = self.right(maxY)
        return speed, angle
    
    def end(self):
        return 0,0

    def cleanup(self):
        """Cleans up camera resources."""
        self.camera.terminate() # this would currently throw an error
