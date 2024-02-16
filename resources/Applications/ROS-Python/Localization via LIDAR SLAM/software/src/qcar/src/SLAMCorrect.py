#!/usr/bin/env python3
from __future__ import division, print_function, absolute_import

import roslib
import rospy
import numpy as np
import message_filters
from pyquaternion import Quaternion
from geometry_msgs.msg import PoseStamped
import time



class PoseCorrect(object):
    def __init__(self):
        
        # Publish frequency is set to the speed at which the slam pose from Hector is published
        publish_freq = 20 
        self.SLAM_Pose_ = rospy.Subscriber('/slam_out_pose', PoseStamped, self.PoseMsg)        
        self.SLAM_Pose_corrected = rospy.Publisher('/slam_pose_corrected', PoseStamped,queue_size = 100 )
        
        # Initialize cashed variables 
        self._SLAM_pose_msg = PoseStamped()
        self.Pose_Rotated = Quaternion()
        
        # Set up a timer for publishing the corrected pose 
        self.publish_timer = rospy.Timer(rospy.Duration(1.0/publish_freq), self.PosePublisher)



    def PoseMsg(self, slam_pose):
        # ============== Variable Initialization ==============
        # Variable used to store the original pose from hector slam 
        pose_orig = Quaternion()
        pose_orig[0] = slam_pose.pose.orientation.x
        pose_orig[1] = slam_pose.pose.orientation.y
        pose_orig[2] = slam_pose.pose.orientation.z
        pose_orig[3] = slam_pose.pose.orientation.w

        self._SLAM_pose_msg = slam_pose
        
        # Rotate slam pose by 90 deg CCW to correct the RPLiDAR pose error 
        q_rotate = Quaternion(axis = [1, 0, 0], angle = -np.pi/4)

        # create a vector to store the normalised pose from Hector SLAM 
        slam_pose_normal = pose_orig.normalised
        self.Pose_Rotated = q_rotate.rotate(slam_pose_normal)


    def PosePublisher(self,event):
        # Variable used for publishing rotated pose 
        SLAM_Pose_rotated = PoseStamped()

        # Correct the x,y,z to match the QCar vehicle frame 
        SLAM_Pose_rotated.pose.position.x = self._SLAM_pose_msg.pose.position.x 
        SLAM_Pose_rotated.pose.position.y = self._SLAM_pose_msg.pose.position.y
        SLAM_Pose_rotated.pose.position.z = self._SLAM_pose_msg.pose.position.z

        # Define new quaternion orientation for corrected pose 
        SLAM_Pose_rotated.pose.orientation.x = self.Pose_Rotated[0]
        SLAM_Pose_rotated.pose.orientation.y = self.Pose_Rotated[1]
        SLAM_Pose_rotated.pose.orientation.z = self.Pose_Rotated[2]
        SLAM_Pose_rotated.pose.orientation.w = self.Pose_Rotated[3]

        # Passing time and frame id from slam_out_pose to the corrected node.
        SLAM_Pose_rotated.header.frame_id = self._SLAM_pose_msg.header.frame_id
        SLAM_Pose_rotated.header.stamp = self._SLAM_pose_msg.header.stamp
                    
        self.SLAM_Pose_corrected.publish(SLAM_Pose_rotated)
            
        
     
        

if __name__ == '__main__':
    rospy.init_node('PoseCorrect_node',disable_signals = True)
    r = PoseCorrect()
    rospy.spin()