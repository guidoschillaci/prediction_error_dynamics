#!/usr/bin/env python
from __future__ import print_function  

#from minisom import MiniSom

#from copy import deepcopy
import h5py
import cv2
from models import Models
from intrinsic_motivation import IntrinsicMotivation

import plots # plot_exploration, plot_learning_progress #, plot_log_goal_inv, plot_log_goal_fwd,  plot_learning_comparisons

# from cv_bridge import CvBridge, CvBridgeError
import random
import os
import time
#import shutil
import pickle
import gzip
import datetime
import numpy as np
import signal
import sys, getopt
#from utils import RomiDataLoader
import utils
import threading
import random
from cam_sim import Cam_sim
from parameters import Parameters
from copy import deepcopy
#from doepy import build, read_write # pip install doepy - it may require also diversipy

#import tensorflow.compat.v1 as tf
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf


GPU_FRACTION = 1

print ('Tensorflow version ', str(tf.__version__))
#if tf.__version__ < "1.14.0":#
#	config = tf.ConfigProto()
#	config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
	#config.gpu_options.allow_growth = True
#	session = tf.Session(config=config)
#else:
#	config = tf.compat.v1.ConfigProto()
#	config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
	#config.gpu_options.allow_growth = True
#	session = tf.compat.v1.Session(config=config)


class GoalBabbling():

	def __init__(self, param):

		# reset
		#print('Clearing TF session')
		#if tf.__version__ < "1.8.0":
		#	tf.reset_default_graph()
		#else:
		#	tf.compat.v1.reset_default_graph()

		self.parameters = param
		# this simulates cameras and positions
		self.cam_sim = Cam_sim("./romi_data/", self.parameters)

		self.lock = threading.Lock()
		signal.signal(signal.SIGINT, self.Exit_call)

		print('Loading test dataset ', self.parameters.get('romi_dataset_pkl'))
		rdl = RomiDataLoader(self.parameters)
		self.train_images, self.test_images, self.train_cmds, self.test_cmds, self.train_pos, self.test_pos = rdl.load_data()


	def initialise(self, param):
		self.parameters = param

		if not param.get('goal_selection_mode') == 'som':
			print('wrong goal selection mode, exit!')
			sys.exit(1)

		self.intrinsic_motivation = IntrinsicMotivation(param)
		self.models = Models(param)

		self.exp_iteration = param.get('exp_iteration')
		self.iteration = 0

		self.pos = []
		self.cmd = []
		self.img = []
		
		self.goal_code = []

		self.current_goal_x = -1
		self.current_goal_y = -1
		self.current_goal_idx = -1
		self.prev_goal_idx = -1

		self.goal_image = np.zeros((1, param.get('image_size'), param.get('image_size'), param.get('image_channels')), np.float32)	


		np.random.seed() # change the seed

		self.prev_pos=self.get_starting_pos()

	def log_MSE(self):
		self.log_current_fwd_mse()
		self.log_current_inv_mse()

	def log_current_inv_mse(self):
		img_codes = self.models.encoder.predict(self.test_images)
		motor_pred = self.models.inv_model.predict(img_codes)
		mse = (np.linalg.norm(motor_pred-self.test_pos) ** 2) / self.parameters.get('romi_test_size')
		print ('Current mse inverse code model: ', mse)
		self.models.logger_inv.store_log(mse)

	def log_current_fwd_mse(self):
		img_obs_code = self.models.encoder.predict(self.test_images)
		img_pred_code = self.models.fwd_model.predict(self.test_pos)
		mse = (np.linalg.norm(img_pred_code-img_obs_code) ** 2) /  self.parameters.get('romi_test_size')
		print ('Current mse fwd model: ', mse)
		self.models.logger_fwd.store_log(mse)

	def create_motor_cmd(self):
		# choose random motor commands from time to time
		cmd_x = 0
		cmd_y = 0
		ran = random.random()
		if ran < self.parameters.get('random_cmd_rate'):
			self.random_cmd_flag = True
			print('generating random motor command')
			cmd_x = random.uniform(utils.x_lims[0], utils.x_lims[1])
			cmd_y = random.uniform(utils.y_lims[0], utils.y_lims[1])
			self.prev_goal_idx = -1 # previous goal was randomly selected

		else:
			self.random_cmd_flag = False # goals was not randmoly selected

			motor_pred = self.models.inv_model.predict(self.goal_code)
			noise_x = np.random.normal(0, self.intrinsic_motivation.get_std_dev_exploration_noise())
			noise_y = np.random.normal(0, self.intrinsic_motivation.get_std_dev_exploration_noise())
			print('prediction ', motor_pred)
			cmd_x = utils.clamp_x(utils.unnormalise_x(motor_pred[0][0] + noise_x, self.parameters))
			cmd_y = utils.clamp_y(utils.unnormalise_y(motor_pred[0][1] + noise_y, self.parameters))
		return cmd_x, cmd_y

	def run_babbling(self):
			
		for _ in range(self.parameters.get('max_iterations')):
			print ('Iteration ', self.iteration)
			# log current mean squared error for FWD and INV models
			self.log_MSE()

			# get the goal index in the SOM using the intrinsic motivation strategy
			self.current_goal_idx, self.current_goal_x, self.current_goal_y = self.intrinsic_motivation.select_goal()
			# get the goal coordinates from the selected neuron coordinates in the SOM feature space
			self.goal_code  = self.models.goal_som._weights[self.current_goal_x, self.current_goal_y].reshape(1, self.parameters.get('code_size'))

			# generate a motor command
			cmd = utils.Position()
			cmd.x, cmd.y = self.create_motor_cmd()

			# execute motor command and generate sensorimotor data
			self.generate_simulated_sensorimotor_data(self.prev_pos, cmd)
			# plot the explored points and the goal positions
			if self.iteration % self.parameters.get('plot_exploration_iter') == 0:
				goals_pos = self.models.inv_model.predict(self.models.goal_som._weights.reshape(len(self.models.goal_som._weights)*len(self.models.goal_som._weights[0]), len(self.models.goal_som._weights[0][0]) ))
				# plot observations and goals
				plots.plot_exploration(positions=self.pos,goals=goals_pos,iteration=self.iteration,param=self.parameters, title = self.parameters.get('goal_selection_mode')+'_'+str(self.iteration))
				# plot memory positions and goals
				plots.plot_exploration(positions=self.models.memory_fwd.input_variables,goals=goals_pos,iteration=self.iteration,param=self.parameters, title = 'memory_inputs_'+str(self.iteration))

			# log the last movement
			if not self.random_cmd_flag and (self.current_goal_idx == self.prev_goal_idx):
				self.intrinsic_motivation.update_movement_dynamics(current_pos=cmd, previous_pos=self.prev_pos)

			# update mse dynamics
			if self.iteration % self.parameters.get('im_frequency_of_update_mse_dynamics') == 0:
				self.intrinsic_motivation.update_mse_dynamics(self.models.logger_fwd.get_last_mse())

			# update error dynamics of the current goal (it is supposed that at this moment the action is finished
			if len(self.img)>0 and not (self.prev_goal_idx == -1) and not self.random_cmd_flag:
				cmd_vector = [ utils.normalise_x(cmd.x, self.parameters), utils.normalise_y(cmd.y, self.parameters)]
				predicted_code = self.models.fwd_model.predict(np.asarray(cmd_vector).reshape((1,2)))
				prediction_error = np.linalg.norm(np.asarray(self.goal_code[:])-np.asarray(predicted_code[:]))
				self.intrinsic_motivation.update_error_dynamics(self.current_goal_x, self.current_goal_y, prediction_error, _append=(self.current_goal_idx == self.prev_goal_idx))

			# fit models	
			if len(self.img) > self.parameters.get('batch_size'):# and (len(self.img) == len(self.pos)):
				# get image codes and position readings from the generated sensorimotor data
				observed_codes_batch = self.models.encoder.predict(np.asarray(self.img[-(self.parameters.get('batch_size')):]).reshape(self.parameters.get('batch_size'), self.parameters.get('image_size'), self.parameters.get('image_size'), self.parameters.get('image_channels'))  )
				observed_pos_batch = self.pos[-(self.parameters.get('batch_size')):]

				# fit the model with the current batch of observations and the memory!
				# create then temporary input and output tensors containing batch and memory
				obs_and_mem_pos = []
				obs_and_mem_img_codes =[]
				if not self.models.memory_fwd.is_memory_empty():
					obs_and_mem_pos = np.vstack((np.asarray(observed_pos_batch), np.asarray(self.models.memory_fwd.input_variables)))
					obs_and_mem_img_codes = np.vstack((np.asarray(observed_codes_batch), np.asarray(self.models.memory_fwd.output_variables)))
				else:
					obs_and_mem_pos = np.asarray(observed_pos_batch)
					obs_and_mem_img_codes =np.asarray(observed_codes_batch)
				# update forward and inverse models
				self.models.train_forward_code_model_on_batch( obs_and_mem_pos, obs_and_mem_img_codes)
				self.models.train_inverse_code_model_on_batch( obs_and_mem_img_codes, obs_and_mem_pos)
				# update autoencoder
				#train_autoencoder_on_batch(self.autoencoder, self.encoder, self.decoder, np.asarray(self.img[-32:]).reshape(32, self.image_size, self.image_size, self.channels), batch_size=self.batch_size, cae_epochs=5)
				# update goals' self organising map
				if self.parameters.get('update_goal_som'):
					self.models.update_som(np.asarray(observed_codes_batch).reshape( (self.parameters.get('batch_size'), self.parameters.get('code_size'))))

			##### post-process steps
			# update the previous goal index variable
			if not self.random_cmd_flag:
				self.prev_goal_idx = self.current_goal_idx
			# update the previous position variable for next iteration
			self.prev_pos.x=cmd.x
			self.prev_pos.y=cmd.y
			self.prev_pos.z=cmd.z
			self.prev_pos.speed=cmd.speed
			# increase iteration count
			self.iteration = self.iteration+1

		### experiment is finished, save models and plots
		print ('Saving models')
		self.save_models()

	def generate_simulated_sensorimotor_data(self, pos, cmd):
		#self.lock.acquire()
		a = [int(pos.x), int(pos.y)]
		b = [int(cmd.x),int(cmd.y)]

		tr = self.cam_sim.get_trajectory(a,b)
		#trn = self.cam_sim.get_trajectory_names(a,b)
		tr_img = self.cam_sim.get_trajectory_images(a, b)
		#print ('image size ', str(param.get('image_size')))
		rounded  = self.cam_sim.round2mul(tr,5) # only images every 5mm
		for i in range(len(tr)):
			self.pos.append([utils.normalise_x(float(rounded[i][0]), self.parameters), utils.normalise_y(float(rounded[i][1]), self.parameters) ])
			self.cmd.append([utils.normalise_x(float(int(cmd.x)), self.parameters), utils.normalise_y(float(int(cmd.y)), self.parameters) ])
			self.img.append(tr_img[i])
			'''
			cv2_img = cv2.imread(trn[i])#,1 )
			if param.get('image_channels') ==1 and (cv2_img.ndim == 3):
				cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
			if param.get('image_resize'):
				cv2_img = cv2.resize(cv2_img,(param.get('image_size'), param.get('image_size')), interpolation = cv2.INTER_LINEAR)
			cv2_img = cv2_img.astype('float32') / 255
			cv2_img.reshape(1, param.get('image_size'), param.get('image_size'), param.get('image_channels'))
			self.img.append(cv2_img)
			'''

			# update memory 
			# first update the memory, then update the models
			observed_pos = self.pos[-1]
			#observed_img = cv2_img
			observed_img = tr_img[i]
			observed_img_code = np.asarray(self.models.encoder.predict(observed_img.reshape(1, self.parameters.get('image_size'), self.parameters.get('image_size'), self.parameters.get('image_channels')))).reshape(self.parameters.get('code_size'))
			self.models.memory_fwd.update(observed_pos, observed_img_code)
			self.models.memory_inv.update(observed_img_code, observed_pos)

		def get_starting_pos(self):
			p = utils.Position()
			p.x = 0.0
			p.y = 0.0
			p.z = -50.0
			p.speed = 1400.0
			return utils.normalise(p, self.parameters)

		def goto_starting_pos(self):
			p = self.get_starting_pos()
			self.generate_simulated_sensorimotor_data(self.prev_pos, p, self.parameters)
			self.prev_pos.x = p.x
			self.prev_pos.y = p.y
			self.prev_pos.z = p.z
			self.prev_pos.speed = p.speed

	def save_models(self):
		#self.lock.acquire()
		self.parameters.save()
		self.models.save_models(self.parameters)
		self.models.save_logs(self.parameters)

		self.intrinsic_motivation.get_linear_correlation_btw_amplitude_and_mse_dynamics()
		self.intrinsic_motivation.get_linear_correlation_btw_amplitude_and_pe_dynamics()

		self.intrinsic_motivation.plot_slopes_of_goals()
		self.intrinsic_motivation.plot_buffer_size()
		self.intrinsic_motivation.save_im()
		self.intrinsic_motivation.plot_slopes(self.parameters)
		self.intrinsic_motivation.plot_correlations()
		#self.lock.release()
		print ('Models saved')
		
	def clear_session(self):
		# reset
		print('Clearing TF session')
		if tf.__version__ < "1.8.0":
			tf.reset_default_graph()
		else:
			tf.compat.v1.reset_default_graph()

	def Exit_call(self, signal, frame):
		print ('Terminating...')
		self.save_models(self.parameters)
		self.goto_starting_pos()
		#sys.exit(1)


class RomiDataLoader:

	def __init__(self, param):
		self.param = param

	#### utility functions for reading visuo-motor data from the ROMI dataset
	# https://zenodo.org/record/3552827#.Xk5f6hNKjjC
	def parse_data(self):
		reshape = self.param.get('load_data_reshape')
		file_name= self.param.get('romi_dataset_pkl')
		pixels = self.param.get('image_size')
		channels = self.param.get('image_channels')
		images = []
		positions = []
		commands = []
		with gzip.open(file_name, 'rb') as memory_file:
			memories = pickle.load(memory_file)#, encoding='bytes')
			print ('converting data...')
			count = 0
			for memory in memories:
				image = memory['image']
				if (channels == 1) and (image.ndim == 3):
					image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				if self.param.get('image_resize'):
					image = cv2.resize(image, (pixels, pixels))

				images.append(np.asarray(image).astype('float32') / 255)

				cmd = memory['command']
				commands.append([utils.normalise_x(float(cmd.x), self.param), utils.normalise_y(float(cmd.y), self.param)] )
				#cmd_p = Position()
				#cmd_p.x = float(cmd.x)
				#cmd_p.y = float(cmd.y)
				#cmd_p.z = -90
				#cmd_p.speed = 1400

				#commands.append(normalise(cmd_p))
				pos = memory['position']
				positions.append([utils.normalise_x(float(pos.x), self.param), utils.normalise_y(float(pos.y), self.param)] )
				#pos_p = Position()
				#pos_p.x = float(pos.x)
				#pos_p.y = float(pos.y)
				#pos_p.z = -90
				#pos_p.speed = 1400

				#positions.append(normalise(pos_p))

				count += 1

		positions = np.asarray(positions)
		commands = np.asarray(commands)
		images = np.asarray(images)
		if reshape:
			images = images.reshape((len(images), pixels, pixels, channels))
		#print ('images shape ', str(images.shape))
		return images, commands, positions

	#### utility functions for reading visuo-motor data from the ROMI dataset
	# https://zenodo.org/record/3552827#.Xk5f6hNKjjC
	def load_data(self):

		images, commands, positions = self.parse_data()
		# split train and test data
		# set always the same random seed, so that always the same test data are picked up (in case of multiple experiments in the same run)
		np.random.seed(self.param.get('romi_seed_test_data'))
		test_indexes = np.random.choice(range(len(positions)), self.param.get('romi_test_size'))
		print ('test_indexes head: ', test_indexes[0:10])
		#reset seed
		np.random.seed(int(time.time()))

		# print ('test idx' + str(test_indexes))
		train_indexes = np.ones(len(positions), np.bool)
		train_indexes[test_indexes] = 0

		# split images
		test_images = images[test_indexes]
		train_images = images[train_indexes]
		test_cmds = commands[test_indexes]
		train_cmds = commands[train_indexes]
		test_pos = positions[test_indexes]
		train_pos = positions[train_indexes]
		print ("number of train images: ", len(train_images))
		print ("number of test images: ", len(test_images))

		return train_images, test_images, train_cmds, test_cmds, train_pos, test_pos


if __name__ == '__main__':

	

	parameters = Parameters()
	parameters.set('goal_selection_mode', 'som')
	parameters.set('exp_iteration', 0)
	
	goal_babbling = GoalBabbling(parameters)
	parameters.set('results_directory', './results/')
	
	if not os.path.exists(parameters.get('results_directory')):
		print ('creating folders')	
		os.makedirs(parameters.get('results_directory'))
		os.makedirs(parameters.get('results_directory')+'plots')
		
	print ('Starting experiment')
		
	# if running different experiments, you can re-set parameters with initialise
	goal_babbling.initialise(parameters)
	goal_babbling.run_babbling(parameters)
	print ('Experiment done')

	'''
	os.chdir('experiments')
	exp_iteration_size = 5
	exp_type = ['db', 'som', 'random']#, 'kmeans']
	history_size = [0, 10, 20]
	prob = [0.1, 0.01]

	for e in range(len(exp_type)):
		print ('exp ', exp_type[e])

		for h in range( len (history_size)):
			print('history size ', history_size[h])

			for p in range(len(prob)):
				print('prob update ', prob[p])

				for i in range(exp_iteration_size):
					print( 'exp ', exp_type[e], ' history size ', str(history_size[h]), ' prob ', str(prob[p]), ' iteration ', str(i) )
					directory = './'+exp_type[e]+'_'+str(history_size[h])+'_'+str(prob[p])+'_'+str(i)+'/'
					if not os.path.exists(directory):
						os.makedirs(directory)

						if not os.path.exists(directory+'models'):
							os.makedirs(directory+'models')

						shutil.copy('../pretrained_models/autoencoder.h5', directory+'models/autoencoder.h5')
						shutil.copy('../pretrained_models/encoder.h5', directory+'models/encoder.h5')
						shutil.copy('../pretrained_models/decoder.h5', directory+'models/decoder.h5')
						shutil.copy('../pretrained_models/goal_som.h5', directory+'models/goal_som.h5')
						shutil.copy('../pretrained_models/kmeans.sav', directory+'models/kmeans.sav')

						os.chdir(directory)
						if not os.path.exists('./models/plots'):
							os.makedirs('./models/plots')
						if not os.path.exists('./data'):
							os.makedirs('./data')
						print ('current directory: ', os.getcwd())

						goal_babbling.initialise( goal_selection_mode= exp_type[e], exp_iteration = i, hist_size= history_size[h], prob_update=prob[p])
						goal_babbling.run_babbling()
						os.chdir('../')
						#GoalBabbling().
						print ('finished experiment ', exp_type[e], ' history size ', str(history_size[h]),' prob ', str(prob[p]), ' iter ', str(i))

						goal_babbling.clear_session()
					print ('experiment ', directory, ' already carried out')
	os.chdir('../')
	plot_learning_comparisons(model_type = 'fwd', exp_size = exp_iteration_size, save = True, show = True)
	plot_learning_comparisons(model_type = 'inv', exp_size = exp_iteration_size, save = True, show = True)
	'''
