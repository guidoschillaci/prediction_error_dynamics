#!/usr/bin/env python
import numpy as np
import rospy
from icub_drivers.msg import Commands, JointPositions
from std_msgs.msg import Empty
import yarp
import sys
import data_keys
import time
#import signal

import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"


class MotorDriver:

	def __init__(self):
		self.robot_ip = "localhost" #"192.168.26.135"

		# Initialise YARP
		yarp.Network.init()
		rospy.loginfo("motor_driver connected to yarp")

		# create ros node
		rospy.init_node('icub_motors_driver', anonymous=True)

		# create the subscribers
		target_pos_sub = rospy.Subscriber('/icubRos/commands/move_to_joint_pos', JointPositions, self.move_to_joint_pos_callback, queue_size=10)

		self.props = []
		self.joint_drivers = [] 
		# encoders for each joint group,e.g. head, left_arm, etc.
		self.pos_control = []
		# number of joints in each joint group
		self.num_joint = [] 
		for j in range(len(data_keys.JointNames)):	
			self.props.append(yarp.Property())
			self.props[-1].put("device", "remote_controlboard")
			self.props[-1].put("local", "/client_motor/"+data_keys.JointNames[j])
			self.props[-1].put("remote", "/icubSim/"+data_keys.JointNames[j])

			self.joint_drivers.append(yarp.PolyDriver(self.props[-1]))
			self.pos_control.append(self.joint_drivers[-1].viewIPositionControl())

		rospy.spin()


	def get_num_joints(self, group_id):
		return self.joint_drivers[group_id].viewIPositionControl().getAxes()

	def move_to_joint_pos_callback(self, msg, verbose=True):
		self.cmd_msg = msg

		if verbose:
			start_time = time.time()

		for jn in range(len(data_keys.JointNames)):
			cmd = data_keys.get_joint_values_from_msg(self.cmd_msg, data_keys.JointNames[jn])
			
			if verbose:
				print ("sending cmd ", str(cmd))
			cmd_yarp = yarp.Vector(self.get_num_joints(jn))
			for j in range(self.get_num_joints(jn)):
				cmd_yarp.set(j, cmd[j])
			# send command
			self.pos_control[jn].setControlMode(yarp.Vocab_encode('pos'))
			self.pos_control[jn].positionMove(cmd_yarp.data())

		if verbose:
			end_time = time.time()
			elapsed = end_time - start_time
			print ("sending command took (seconds): ", elapsed)



if __name__=="__main__":
	rospy.loginfo("motor_driver started")
	try:
		motorDriver = MotorDriver()
		
	except rospy.ROSInterruptException:
		pass

