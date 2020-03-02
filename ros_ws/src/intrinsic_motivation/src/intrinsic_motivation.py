#!/usr/bin/env python
#from _ast import alias
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from icub_drivers.msg import Commands, JointPositions
import data_keys, utils
import time
import yarp
import cv2
import random
from cv_bridge import CvBridge
bridge = CvBridge()

import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"

#robot_ip = "192.168.26.135"
robot_ip = "localhost"
#robot_port = 9559

class IntrinsicMotivation():
	def __init__(self):
		# flag to check for generating new motor commands
		self.is_moving = False

		# create ros node
		rospy.init_node('icub_intrinsic_motivation', anonymous=True)

		# create the publishers for sending motor commands
		self.cmd_pub = rospy.Publisher('/icubRos/commands/move_to_joint_pos', JointPositions, queue_size=10)
		
		# subscribe to the topics
		joint_speed_sub = rospy.Subscriber('/icubRos/sensors/joint_speeds', JointPositions, self.joint_speed_callback, queue_size=10)
		# only for head joints at the moment
		self.joint_limits = utils.get_joint_pos_limits_dumb()
		#print (str(self.joint_limits))
#		rospy.spin()
		self.motor_babbling()

	# only for head joints at the moment
	def gen_random_cmd_msg(self):
		cmd=[]
		for j in range(len(self.joint_limits)):
			cmd.append(random.uniform(self.joint_limits[j][0], self.joint_limits[j][1]))
		cmd_msg = JointPositions()
		cmd_msg.header.stamp = rospy.Time.now()
		cmd_msg.head = np.asarray(cmd)
		#print (str(cmd))
		return cmd_msg


	def joint_speed_callback(self, speed_msg):
		# do it for all the joint groups
		speeds = speed_msg.head
		#print (str(speeds))
		if sum(speeds[0:2])< data_keys.SPEED_THRESHOLD / 3:
			self.is_moving=False
			print ("not moving")
		else:
			self.is_moving = True
			print ("moving")

	def motor_babbling(self):
		while True:
			if not self.is_moving:
				cmd_msg = self.gen_random_cmd_msg()
				print ("sending command")
				self.cmd_pub.publish(cmd_msg)
				time.sleep(1.5)

	def __del__(self): 
		pass

if __name__=="__main__":
	rospy.loginfo("intrinsic_motivation")
	try:
		intrMot = IntrinsicMotivation() 
		
	except rospy.ROSInterruptException:
		pass
