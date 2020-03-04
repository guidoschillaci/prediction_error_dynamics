# Use this simple simulator with the visuo-motor dataset https://zenodo.org/record/3552827#.Xk21zBNKiRt
# Authors:
# Antonio Pico Villalpando, Humboldt-Universitaet zu Berlin
# Guido Schillaci, Scuola Superiore Sant'Anna

import sys
import os
import cv2
import numpy as np
import re
import gzip 
import pickle
from utils import Position

class Cam_sim():
	def __init__(self,imagesPath):
		self.imagesPath = imagesPath
		if self.imagesPath[-1] != '/':
			self.imagesPath += '/'
		self.load_images_in_mem()

	def load_images_in_mem(self):
		print('loading simulator images in memory')
		self.images = [] * 158
		for i in range(len(self.images)):
			self.images[i] = [] * 158

		print ('images sim size ', np.asarray(self.images).shape)

		with gzip.open(self.imagesPath, 'rb') as memory_file:
			memories = pickle.load(memory_file)
			print ('loading images...')
			count = 0
			x_vector = []
			y_vector = []
			for memory in memories:
				image = memory['image']
				# image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
				# image = cv2.resize(image, (64, 64))
				cmd = memory['position']
				self.images[cmd.x][cmd.y] = image
				#x_vector.append(float(cmd.x))
				#y_vector.append(float(cmd.y))
				#title = './romi_data/x' + str(cmd.x) + '_y' + str(cmd.y) + '.jpeg'
				#cv2.imwrite(title, image)
			#print ('x len', len(np.asarray(x_vector)), 'x max ', np.max(np.asarray(x_vector)), ' x_min ',np.min(np.asarray(x_vector)))
			#print ('y len', len(np.asarray(y_vector)), 'y max ', np.max(np.asarray(y_vector)), ' y_min ',np.min(np.asarray(y_vector)))
			#horiz = np.max(np.asarray(x_vector)) / 5 + 1
			#vert = np.max(np.asarray(y_vector)) / 5 + 1
			#print ('hori ', horiz, ' vert ', vert, ' prod ', (horiz * vert))
			print('Simulator images loaded')

	def round2mul(self,number, multiple):
		half_mult = multiple/2.
		result = np.floor( (number + half_mult) / multiple  ) * multiple
		return result.astype(np.int)
	
	def get_trajectory(self,start,end):
		trajectory =  np.array(self.get_line(start, end))
		return trajectory



	def get_trajectory_names(self,start,end):
		trajectory = self.get_trajectory(start,end)
		t_rounded = self.round2mul(trajectory,5) #there is only images every 5 mm, use closer image to real coordinate
		t_images = []
		for i in t_rounded:
			img_name = self.imagesPath + "x{:03d}_y{:03d}.jpeg".format(i[0],i[1])
			t_images.append(img_name)
		return t_images

	def get_line(self,start, end):
		# Setup initial conditions
		x1, y1 = start
		x2, y2 = end
		dx = x2 - x1
		dy = y2 - y1

		# Determine how steep the line is
		is_steep = abs(dy) > abs(dx)

		# Rotate line
		if is_steep:
			x1, y1 = y1, x1
			x2, y2 = y2, x2

		# Swap start and end points if necessary and store swap state
		swapped = False
		if x1 > x2:
			x1, x2 = x2, x1
			y1, y2 = y2, y1
			swapped = True

		# Recalculate differentials
		dx = x2 - x1
		dy = y2 - y1

		# Calculate error
		error = int(dx / 2.0)
		ystep = 1 if y1 < y2 else -1

		# Iterate over bounding box generating points between start and end
		y = y1
		points = []
		for x in range(x1, x2 + 1):
			coord = (y, x) if is_steep else (x, y)
			points.append(coord)
			error -= abs(dy)
			if error < 0:
				y += ystep
				error += dx

		# Reverse the list if the coordinates were swapped
		if swapped:
			points.reverse()
		return points



def extract_images(file_name):
	with gzip.open(file_name, 'rb') as memory_file:
		memories = pickle.load(memory_file)
		print ('extracting images...')
		count = 0
		x_vector=[]
		y_vector=[]
		for memory in memories:
			image = memory['image']
			#image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
			image = cv2.resize(image, (64, 64))
			cmd = memory['position']
			x_vector.append(float(cmd.x))
			y_vector.append(float(cmd.y))
			title = './romi_data/x'+str(cmd.x)+'_y'+str(cmd.y)+'.jpeg'
			cv2.imwrite(title,image)
		print ('x len', len(np.asarray(x_vector)), 'x max ', np.max(np.asarray(x_vector)), ' x_min ', np.min(np.asarray(x_vector)))
		print ('y len', len(np.asarray(y_vector)), 'y max ', np.max(np.asarray(y_vector)), ' y_min ', np.min(np.asarray(y_vector)))
		horiz = np.max(np.asarray(x_vector)) / 5 +1
		vert = np.max(np.asarray(y_vector)) / 5 +1
		print ('hori ', horiz, ' vert ', vert, ' prod ', (horiz*vert))

if __name__ == '__main__':

	path="./romi_data"
	channels =1
	compressed_dataset = path+'/compressed_dataset.pkl'
	if os.path.isfile(compressed_dataset):
		print ('compressed dataset already exists')
		extract_images(compressed_dataset)

	else:
		print ('creating compressed dataset')

		samples = []
		counter=0
		for file in os.listdir(path):
			filename_full = os.path.basename(file)
			filename = os.path.splitext(filename_full)[0]
			splitted = re.split('x|_|y', filename)
			p = Position()
			p.x=splitted[1]
			p.y=splitted[3]
			p.z=-90
			p.speed=1400
			#print path+'/'+os.path.relpath(file)
			cv_img = cv2.imread(path+'/'+os.path.relpath(file))
			if channels ==1:
				cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
			cv_img = cv2.resize(cv_img, (64, 64))
			#image_msg= bridge.cv2_to_imgmsg(cv_img, "bgr8")
			#samples.append({'image': image_msg, 'position':p, 'command':p})
			samples.append({'image': cv_img, 'position': p, 'command': p})
			#print int(p.x), ' ', int(p.y)
			counter=counter+1
			print (counter)
		with gzip.open(compressed_dataset, 'wb') as file:
			pickle.dump(samples, file, protocol=2)
		print ('saved')
		sys.exit(1)

