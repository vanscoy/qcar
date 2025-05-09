##############################################
# This is the completed main file for Miami University (OH) Senior Design Capstone: 2025 Quanser QCar Self Driving Competiton
# This project was completed by Zach Copenhaver, Frederick Levins, Jakob Feltz, and Josh Strong.
# This project was advised by Dr. Bryan Van Scoy and Dr. Dave Hartup.
# Please see comments, code, gitHub, report, and files for more information into functionality and citations.
# lfDC.py = Line Following Controls File, YoloDC.py = main driver for YOLOv3-tiny real time object detection, CaStar.py = A* Algorithim, rTDC = main driver for node detection 
# lfDC was headed by Zach Copenhaver, CaStar by Jakob Feltz, and YOLODc by Frederick Levins. rTDC was codesigned by Zach and Frederick with Zach's final blue mask design being used.
# This project was fully completed and live demonstrated at the Millet Hall 2025 Engineering Design expo.
# Control-C is the functionality to exit code if during concurrent running.
# This was an overall team project with all four members contributing valuable and important information and insight. Without everyone, it would not have worked!
# Thank you for viewing our project!
#
# **Important note: This code requires XServer connection for frame viewing and overall usage and a Quanser QCar 1, will not work otherwise.**
# **Important note: The Quanser folder must be in the active directory
# **Important note: Due to time constraints and deadlines, this code could be further optimzed and improved upon.**
##############################################


import os
import cv2
import numpy as np
import threading
import time
import sys
from Quanser.q_essential import Camera2D
from Quanser.product_QCar import QCar
from Quanser.q_ui import gamepadViaTarget
from Quanser.q_misc import Calculus, Utilities
from YoloDC import YOLODetector
from lfDC import LineFollower
from rTDC import BlueTapeDetector
from speedCalc import *
from responseNode import * 
from CaStar import *
from log import Log

# Restart NVArgus daemon before using the camera
os.system("sudo systemctl restart nvargus-daemon")

# Paths to YOLO model files
CONFIG_PATH = '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/Modular/yolov3-tiny.cfg'
WEIGHTS_PATH = '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/Modular/yolov3-tiny.weights'
CLASSES_PATH = '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/Modular/yolov3.txt'


# --------------------------
# MultiFeedDisplay class
# This class combines the output 
# if wanting only one output adjust winN with proper choice
# --------------------------
class MultiFeedDisplay:
    def __init__(self, winN="Multi Feed Display", targetSize=(320, 240)):
        self.winN = winN
        self.targetSize = targetSize

    # Function to display feeds in a combined XServer output
    def dispFeeds(self, blueF, lineF, yoloF):

        blueD = self.ensureFrame(blueF)
        lineD = self.ensureFrame(lineF)
        yoloD = self.ensureFrame(yoloF)
        
        combined = np.hstack((blueD, lineD, yoloD))
        cv2.imshow(self.winN, combined)
        return combined

    # Ensure a frame is read in and not empty
    def ensureFrame(self, frame):
        if frame is None:
            return np.zeros((self.targetSize[1], self.targetSize[0], 3), dtype=np.uint8)
        return cv2.resize(frame, self.targetSize)

    def cleanup(self):
        cv2.destroyAllWindows()


# --------------------------
# QCarController class
# Main class for functionality and interaction amongst working algorithms.
# --------------------------

# Initialize important variables for various usages
class QCarController:
    def __init__(self, path):
        self.imageWidth = 640
        self.imageHeight = 480
        self.sampleRate = 30.0

        # Create a single camera instance and a lock for safe access
        self.frontCam = Camera2D(camera_id="3", 
                                  frame_width=self.imageWidth, 
                                  frame_height=self.imageHeight, 
                                  frame_rate=self.sampleRate)
        
        # use the right camera to turn right when there is no line to follow
        self.rightCam = Camera2D(camera_id="0", 
                                  frame_width=self.imageWidth, 
                                  frame_height=self.imageHeight, 
                                  frame_rate=self.sampleRate)
        
        # use the right camera to turn left when there is no line to follow
        self.leftCam = Camera2D(camera_id="2", 
                                  frame_width=self.imageWidth, 
                                  frame_height=self.imageHeight, 
                                  frame_rate=self.sampleRate)
        self.camLock = threading.Lock()
        self.sharedF = None  # Updated continuously by camRead thread
        self.sharedFL = None
        self.sharedFR = None

        # objects
        self.car = QCar()
        self.gamepad = gamepadViaTarget(1)
        self.yoloDetect = YOLODetector(CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH)
        self.lineFollower = LineFollower(self.car, '12')
        self.blueTapeDetector = BlueTapeDetector()
        self.log = Log(True)

        # control variables
        self.controlCount = 0
        self.startTime = time.time()

        # logging variables
        self.controlsFrameList = list()
        self.tapeFrameList = list()
        self.multiFeedFrameList = list()
        self.stopSignFrameList = list()
        self.dataList = list()
        self.dataFields = ['count','speed','angle','max_y','edge','prev_node', 'currNode','nextNode','state_count','lineMean','bat_volts']
        
        # Processed outputs for display
        self.yoloF = None
        self.lineF = None
        self.blueF = None

        # stored detected objects for output
        self.detectedObjs = []

        # create events
        self.camE = threading.Event()
        self.yoloE = threading.Event()
        self.lineE = threading.Event()
        self.tapeE = threading.Event()

        # Use a single stopEvent for clean shutdown
        self.stopEvent = threading.Event()

        # A* variables 
        self.pathPos = 0
        self.path = path
        self.currNode = self.path[self.pathPos]
        self.nextNode = self.path[self.pathPos+1]
        self.mySpeed = speedCalc(np.array([0.0, 0.0, 0.0]), self.car, 0)
        self.currentDist = 0
        self.totalDist = 0
        self.detecting = False
        self.startNode = 0
        self.goal = 0

        # Node variables 
        self.bTDetect = False
        self.bTPrevDetect = False
        self.countNode = 0

        # Yolo Variables 
        self.stopSignSeen = False
        self.stopSignTime = 0
        self.stopSignCount = 0

    # Main camera concurrent read funnction and updates
    def camRead(self):
        while not self.stopEvent.is_set():
            with self.camLock:
                self.frontCam.read()
                frame = self.frontCam.image_data
                if frame is not None and frame.size != 0:
                    self.sharedF = frame.copy()


    # Main call for yolo algorithm
    def procY(self):
        # Processes the shared frame using YOLO detection.
        while not self.stopEvent.is_set():
            # If no frame yet, wait briefly
            if self.sharedF is None:
                continue
            with self.camLock:
                frameC = self.sharedF.copy()
            processFrame, detectedObjs = self.yoloDetect.procFrame(frameC)
            
            self.stopSignFrameList.append(processFrame)
            self.yoloF = processFrame
            self.detectedObjs = detectedObjs

            # Main Functionality of detecting stop sign and stop at detection
            for obj in self.detectedObjs:
                if obj["class"].lower() == 'stop sign':
                    print("You are at a stop sign!")
                    self.yoloE.set()
                    self.lineE.set()
                    self.yoloE.wait(timeout = 6.5)
                    time.sleep(6.5)
                    self.yoloE.clear()

    # This function determines when we use each camera
    def chooseCamFeed(self, edge):
        if edge == "12": return self.frontLF()
        if edge == "23": return self.frontLF()
        if edge == "34": return self.frontLF()
        if edge == "316": return self.leftLF()
        if edge == "45": return self.frontLF()
        if edge == "56": return self.frontLF()
        if edge == "522": return self.leftLF()
        if edge == "67": return self.frontLF()
        if edge == "78": return self.frontLF()
        if edge == "71": return self.frontLF()
        if edge == "89": return self.rightLF()
        if edge == "812": return self.frontLF()
        if edge == "818": return self.leftLF()
        if edge == "93": return self.frontLF()
        if edge == "1012": return self.rightLF()
        if edge == "1018": return self.frontLF()
        if edge == "1023": return self.leftLF()
        if edge == "1213": return self.rightLF()
        if edge == "124": return self.leftLF()
        if edge == "1310": return self.frontLF()
        if edge == "1513": return self.rightLF()
        if edge == "1516": return self.rightLF()
        if edge == "1618": return self.rightLF()
        if edge == "169": return self.leftLF()
        if edge == "1623": return self.frontLF()
        if edge == "1819": return self.rightLF()
        if edge == "186": return self.leftLF()
        if edge == "1915": return self.frontLF()
        if edge == "2022": return self.rightLF()
        if edge == "2019": return self.rightLF()
        if edge == "2223": return self.rightLF()
        if edge == "229": return self.frontLF()
        if edge == "2212": return self.leftLF()
        if edge == "2320": return self.frontLF()
        return self.frontLF()
    
    # the following 3 functions choose which camera we read from for specific states
    def rightLF(self):
        self.rightCam.read()
        frame = self.rightCam.image_data
        frame = frame[280:,:].copy()
        camID = 0
        return frame, camID
    
    def leftLF(self):
        self.leftCam.read()
        frame = self.leftCam.image_data
        frame = frame[240:320,:].copy()
        camID = 2
        return frame, camID
    
    def frontLF(self):
        self.frontCam.read()
        frame = self.frontCam.image_data
        frame = frame[200:400,:].copy()
        camID = 3
        return frame, camID

    # Line following alg but truly overall controls alg
    def procLF(self):

        #Processes the shared frame using line following logic.
        while not self.stopEvent.is_set() :
            if self.sharedF is None:
                time.sleep(0.01)
                continue
            with self.camLock:

                # grab shared frame and copy to inherent process of line following
                frameC = self.sharedF[200:400, :].copy()
                currNode = self.currNode
                pathPos = self.pathPos

                # Check nodes and update path
                if self.pathPos >= len(self.path) - 1:
                    nextNode = self.nextNode
                else:
                    nextNode = self.path[pathPos+1]
                    self.nextNode = nextNode
                path = self.path
                edge = self.lineFollower.getEdge(path, pathPos, currNode)

                # get camera feed
                frameC, camID = self.chooseCamFeed(edge)
            
            # If thread open, being main line following control functionality
            if self.lineE.is_set():
                # this block executes when the stop sign is detected; The car will pause
                binary = self.lineFollower.bgr2Binary(frameC, 130)
                data = [0]*len(self.dataFields)
                mtr_current, bat_voltage, mtr_encoder = self.car.read_write_std(np.array([data[1], data[2]]), np.array([0, 0, 0, 0, 0, 0, 0, 0]))
                data.append(bat_voltage)
                self.lineE.wait(timeout=1.5)
                time.sleep(1.5)
                self.lineE.clear()
            else:
                data, binary = self.lineFollower.getControl(frameC, self.dataList, self.controlCount, currNode, nextNode, path, pathPos, camID)
                mtr_current, bat_voltage, mtr_encoder = self.car.read_write_std(np.array([data[1], data[2]]), np.array([0, 0, 0, 0, 0, 0, 0, 0]))
                data.append(bat_voltage)
            
            # Convert binary image to BGR for display purposes
            if binary is not None:
                lineDlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            else:
                lineDlay = np.zeros((self.imageHeight, self.imageWidth, 3), dtype=np.uint8)
            self.lineF = lineDlay

            # append frame and data to list for logging
            self.controlsFrameList.append(binary)
            self.dataList.append(data)
            self.controlCount += 1

            time.sleep(0.01)

    # Read in blue tape on track : represents nodes for robot localization
    def procRT(self):
        while not self.stopEvent.is_set():
            if self.sharedF is None:
                time.sleep(0.01)
                continue
            with self.camLock:
                frameC = self.sharedF.copy()
                prev_node = self.lineFollower.prevNode
                currNode = self.currNode

            # Solve error of overreading nodes
            if prev_node != currNode:
                self.tapeE.set()
                self.tapeE.wait(timeout=3)
                time.sleep(3)
                self.tapeE.clear()
            processFrame, mask, bounding_boxes, blue_detected = self.blueTapeDetector.find_blue(frameC)
            self.blueF = processFrame
            self.updNNode()
            self.tapeFrameList.append(processFrame)
            time.sleep(0.01)

    # Update next node using blue tape detection
    def updNNode(self):

        # Safely copy detected bounding boxes
        with self.blueTapeDetector.lock:
            boxes = self.blueTapeDetector.bounding_boxes
        if boxes:

            # Choose the largest bounding box (by area)
            cenX = (boxes[0] + boxes[2]) / 2

            # Map the center x-coordinate to a node:
            if boxes[2]*boxes[3] >= 500:
                self.nextNode = self.checkRTNN(cenX) 
            return self.nextNode
        else:
            if self.detecting == True:
                self.detecting = False
                self.pathPos = self.pathPos + 1

                # Safety statement to keep running even when current node is path's end
                if self.pathPos+1 > len(self.path):
                    self.pathPos = 0
                self.currNode = self.path[self.pathPos]
            return None

    # If tape is the next node, then it returns back what should be next in the path
    def checkRTNN(self, cx):
        if (cx < 2 * self.imageWidth / 3):
            self.detecting = True
            if self.pathPos+1 >= len(self.path):
                    return self.path[1]
            self.countNode += 1
            self.nextNode = self.path[self.pathPos+1]
            return self.path[self.pathPos+1]
        var = self.currNode
        return self.nextNode 

    # Starting function for threading and start of processes
    def start(self):

        # Create threads for camera reading and processing modules
        camT = threading.Thread(target=self.camRead, daemon=True)
        yoloT = threading.Thread(target=self.procY, daemon=True)
        lineT = threading.Thread(target=self.procLF, daemon=True)
        bluetT = threading.Thread(target=self.procRT, daemon=True)
        

        # Start all threads
        camT.start()
        yoloT.start()
        lineT.start()
        bluetT.start()

        display = MultiFeedDisplay()
        try:

            # Main loop: combine and display processed feeds
            while not self.stopEvent.is_set():

                # Periodically update next node based on blue tape detection if not reached
                if self.pathPos >= len(self.path) - 1:
                    print(f"Goal node {self.path[self.pathPos]} reached.")
                    return True
                
                # Display combined feeds: blue tape, line following, and YOLO output
                mFDFrame = display.dispFeeds(self.blueF, self.lineF, self.yoloF)
                self.multiFeedFrameList.append(mFDFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stopEvent.set()
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stopEvent.set()
            return False
        finally:
            self.stopEvent.set()
            camT.join()
            yoloT.join()
            lineT.join()
            bluetT.join()
            self.car.read_write_std(np.array([0, 0]), np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            self.car.terminate()
            self.frontCam.terminate()
            self.rightCam.terminate()
            self.leftCam.terminate()
            self.gamepad.terminate()
            display.cleanup()

            # Save logs of the videos and control data for testing purposes.
            
            # Control logs.
            # self.log.save_video(self.controlsFrameList, False, 640, 80, 5, '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/outputVideos/CSI_Left_Camera/')
            # self.log.save_video(self.controlsFrameList, False, 640, 200, 5, '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/outputVideos/CSI_Front_Camera/')
            # self.log.save_data(self.dataFields, self.dataList)
            # rt video log
            # self.log.save_video(self.tapeFrameList, True, 640, 480, 5, '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/outputVideos/tape/')
            # stop sign video log
            # self.log.save_video(self.stopSignFrameList, True, 640, 480, 5, '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/outputVideos/stopSign/')
            # multi feed display video log
            # self.log.save_video(self.multiFeedFrameList, True, 960, 240, 8, '/home/nvidia/Documents/Python/qcar/3 - ACC 2025 competition/outputVideos/multiFeedDisplay/')
            # print("Saved logs for line following")

            cv2.destroyAllWindows()
            print("\nThanks for trying out our project! Hope you enjoyed.")

# Main call function
if __name__ == "__main__":

    # Command Line Interface Tool
    print("\nWelcome to the 2024/2025 autonomous QCAR capstone!\n\nThis is a project team compiled of:\nFrederick Levins\nZach Copenhaver\nJakob Felts\nJosh Strong\n\nAdvised by: \nDr. Dave Hartup\nDr. Bryan Van Scoy\n\nUse this CLI tool to interact with the car. \nEnjoy!")
    while True:
        print("\nEnter the starting point and the ending point. Please remember the ending point for continuation!\n")
        path = genPath()

        # hardcoded path for largest test case of all turns, straights, and object detection
        #path = [1, 2, 3, 16, 18, 19, 15, 13, 10, 23, 20, 19, 15, 16, 18, 6, 7]

        if path:
            controller = QCarController(path)
            print("Current Path:")
            print(path)
            time.sleep(2)
            success = controller.start()

            if success:
                print("\nWelcome to the 2024/2025 autonomous QCAR capstone!\n\nThis is a project team compiled of:\nFrederick Levins\nZach Copenhaver\nJakob Felts\nJosh Strong\n\nAdvised by: \nDr. Dave Hartup\nDr. Bryan Van Scoy\n\nUse this CLI tool to interact with the car. \nEnjoy!")
                print("\nUpon a consecutive runs, please input start cell number as previous goal cell")
                userInput = input("Goal reached! Input a new path? (y/n): ").strip().lower()
                if userInput != 'y':
                    print("Thanks for trying out our project! Hope you enjoyed.")
                    break
            else:
                print("Controller interrupted or failed. Exiting.")
                break


