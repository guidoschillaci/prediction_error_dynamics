#!/usr/bin/env python
#from _ast import alias
import numpy as np
import rospy
from icub_drivers.msg import JointPositions
import data_keys
import time
import yarp
import threading
lock = threading.Lock()
import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"

#robot_ip = "192.168.26.135"
robot_ip = "localhost"
#robot_port = 9559

##use this when using gazebo. Use a better way to check this
if robot_ip =="localhost":
	rospy.set_param("use_sim_time", 'false')
		
class JointReader():

	def __init__(self):
		# Initialise YARP
		yarp.Network.init()

		# create ros node
		rospy.init_node('icub_sensors_driver', anonymous=True)
		
		# create the publishers
		self.joint_pos_pub = rospy.Publisher('/icubRos/sensors/joint_positions', JointPositions, queue_size=10)
		self.joint_speed_pub = rospy.Publisher('/icubRos/sensors/joint_speeds', JointPositions, queue_size=10)


		self.props = []
		self.joint_drivers = [] 
		# encoders for each joint group,e.g. head, left_arm, etc.
		self.encoders = []
		# number of joints in each joint group
		self.num_joint = [] 
		for j in range(len(data_keys.JointNames)):	
			self.props.append(yarp.Property())
			self.props[-1].put("device", "remote_controlboard")
			self.props[-1].put("local", "/client/"+data_keys.JointNames[j])
			self.props[-1].put("remote", "/icubSim/"+data_keys.JointNames[j])

			self.joint_drivers.append(yarp.PolyDriver(self.props[-1]))
			self.encoders.append(self.joint_drivers[-1].viewIEncoders())

		# initialise ROS messages
		self.joint_pos_msg = JointPositions()
		self.joint_speed_msg = JointPositions()
		self.current_time = time.time()
		rate = 10.0 #hertz
		#rate = rospy.Rate(10) # for some reasons, using rospy rate slows it down a lot (check why)
		while not rospy.is_shutdown():
			self.read_and_publish_in_ros(verbose=False)
			time.sleep(1/rate)
			#rate.sleep()
			#rospy.sleep(1/rate)

	def get_num_joints(self, group_id):
		return self.joint_drivers[group_id].viewIPositionControl().getAxes()

	def read_and_publish_in_ros(self, verbose=False):
		if verbose:
			start_time = time.time()
		# set ROS timestamp
		self.joint_pos_msg.header.stamp = rospy.Time.now()
		# lock to this thread	
		lock.acquire()
		for j in range(len(data_keys.JointNames)):
			self.current_position = yarp.Vector(self.get_num_joints(j))
			self.current_speed = yarp.Vector(self.get_num_joints(j))
			# get data from encoders
			self.encoders[j].getEncoders(self.current_position.data())
			self.encoders[j].getEncoderSpeeds(self.current_speed.data())
			# copy values into ROS msg
			self.joint_pos_msg = data_keys.set_joint_pos_msg_value(self.joint_pos_msg, data_keys.JointNames[j], self.current_position.toString(-1,1), self.get_num_joints(j))
			self.joint_speed_msg = data_keys.set_joint_pos_msg_value(self.joint_speed_msg, data_keys.JointNames[j], self.current_speed.toString(-1,1), self.get_num_joints(j))
		# release lock
		lock.release()

		if verbose:
			end_time = time.time()
			elapsed = end_time - start_time
			print ("reading time (seconds): ", elapsed, " time_btw ", end_time-self.current_time, " time now ", end_time, " ros_stamp ", self.joint_pos_msg.header.stamp)
			self.current_time=end_time
		# publish ROS message
		self.joint_pos_pub.publish(self.joint_pos_msg)
		self.joint_speed_pub.publish(self.joint_speed_msg)


	def __del__(self): 
		# Cleanup
		#self.input_port_arm.close()
		pass



if __name__=="__main__":
	rospy.loginfo("main_icub_sensor_driver")
	try:
		jointReader = JointReader() 
	except rospy.ROSInterruptException:
		pass
