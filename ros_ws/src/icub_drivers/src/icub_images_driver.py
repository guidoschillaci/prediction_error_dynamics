#!/usr/bin/env python
#from _ast import alias
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
import data_keys
import time
import yarp
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"

#robot_ip = "192.168.26.135"
robot_ip = "localhost"
#robot_port = 9559

##use this when using gazebo. Use a better way to check this
if robot_ip =="localhost":
	rospy.set_param("use_sim_time", 'false')
		
class ImageReader():

	def __init__(self, width=640, height=480):
		# Initialise YARP
		yarp.Network.init()

		# create ros node
		rospy.init_node('icub_images_driver', anonymous=True)
		
		# create the publishers
		self.image_pub = rospy.Publisher('/icubRos/sensors/camera_left', Image, queue_size=10)

		# Create a port and connect it to the iCub simulator virtual camera
		self.input_port_cam = yarp.Port()
		self.input_port_cam.open("/icubRos/camera_left")
		yarp.Network.connect("/icubSim/cam/left", "/icubRos/camera_left")

		# prepare image
		self.yarp_img_in = yarp.ImageRgb()
		self.yarp_img_in.resize(width,height)
		self.img_array = np.zeros((height, width, 3), dtype=np.uint8)
		# yarp image will be available in self.img_array
		self.yarp_img_in.setExternal(self.img_array, width, height)


		# create time variable for calculating reading time
		self.current_time = time.time()
		print ("Starting reading")
		rate = rospy.Rate(10) 
		rate = 10.0
		while not rospy.is_shutdown():
		#while True:
			self.read_and_publish_in_ros()
			#rate.sleep()
			time.sleep(1/rate)

	def read_and_publish_in_ros(self, verbose = False):
		if verbose: # start reading time
			start_time = time.time()
		# read image
		self.input_port_cam.read(self.yarp_img_in)
		# scale down img_array and convert it to cv2 image
		self.image = cv2.resize(self.img_array,(64, 64), interpolation = cv2.INTER_LINEAR)

		#cv2.imshow("camera left", self.image)
		#cv2.waitkey(0)
		# make a ROS image message from the cv2 image
		self.image_msg = bridge.cv2_to_imgmsg(self.image, 'bgr8')
		# set current timestamp
		self.image_msg.header.stamp = rospy.Time.now()
		# publish image
		self.image_pub.publish(self.image_msg)
		if verbose: # calculate reading time 
			end_time = time.time()
			elapsed = end_time - start_time
			print ("Image Reading time (seconds): ", elapsed, " time_btw ", end_time-self.current_time, " time now ", end_time, " ros_stamp ", self.image_msg.header.stamp)
			self.current_time=end_time
	

	def __del__(self):
		# Cleanup
		#self.input_port_arm.close()
		pass



if __name__=="__main__":
	rospy.loginfo("main_icub_image_driver")
	try:
		imageReader = ImageReader() 
	except rospy.ROSInterruptException:
		pass
