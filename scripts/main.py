#!/usr/bin/env python
from __future__ import print_function  

#from minisom import MiniSom

#from copy import deepcopy
import h5py
import cv2
from models import Models
from intrinsic_motivation import IntrinsicMotivation

from plots import plot_exploration, plot_learning_progress #, plot_log_goal_inv, plot_log_goal_fwd,  plot_learning_comparisons

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
import tensorflow as tf


GPU_FRACTION = 0.5

print ('Tensorflow version ', str(tf.__version__))
#if tf.__version__ < "1.14.0":
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
session = tf.Session(config=config)
#else:
#    config = tf.compat.v1.ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
#    session = tf.compat.v1.Session(config=config)


class GoalBabbling():

	def __init__(self, param):

		# reset
		print('Clearing TF session')
		if tf.__version__ < "1.8.0":
			tf.reset_default_graph()
		else:
			tf.compat.v1.reset_default_graph()

		# this simulates cameras and positions
		self.cam_sim = Cam_sim("./romi_data/")
		self.parameters = param

		self.lock = threading.Lock()
		signal.signal(signal.SIGINT, self.Exit_call)

		print('Loading test dataset ', self.parameters.get('romi_dataset_pkl'))
		rdl = RomiDataLoader(self.parameters)
		self.train_images, self.test_images, self.train_cmds, self.test_cmds, self.train_pos, self.test_pos = rdl.load_data()


	def initialise(self, param):
		self.parameters = param
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


	def log_current_inv_mse(self, param):
		img_codes = self.models.encoder.predict(self.test_images)
		motor_pred = self.models.inv_model.predict(img_codes)
		#print ('motor pred ', np.asarray(motor_pred), ' test ', self.test_pos) 

		#mse = 0
		#for i in range(len(self.test_pos)):
		#	mse= mse + (np.power(motor_pred[i][0]- self.test_pos[i].x, 2) + np.power(motor_pred[i][1]- self.test_pos[i].y, 2))
		#mse = mse/param.get('romi_test_size')

		mse = (np.linalg.norm(motor_pred-self.test_pos) ** 2) / param.get('romi_test_size')
		print ('Current mse inverse code model: ', mse)
		self.models.logger_inv.store_log(mse)

	def log_current_fwd_mse(self, param):
		img_obs_code = self.models.encoder.predict(self.test_images)
		img_pred_code = self.models.fwd_model.predict(self.test_pos)
		mse = (np.linalg.norm(img_pred_code-img_obs_code) ** 2) /  param.get('romi_test_size')
		print ('Current mse fwd model: ', mse)
		self.models.logger_fwd.store_log(mse)

	def run_babbling(self, param):
		p = utils.Position()
			
		for _ in range(param.get('max_iterations')):

			# record logs and data
			self.log_current_inv_mse(param)
			self.log_current_fwd_mse(param)

			#print ('Mode ', self.goal_selection_mode, ' hist_size ', str(self.history_size), ' prob ', str(self.history_buffer_update_prob), ' iteration : ', self.iteration)
			print ('Iteration ', self.iteration)
			

			# select a goal using the intrinsic motivation strategy
			self.current_goal_idx, self.current_goal_x, self.current_goal_y = self.intrinsic_motivation.select_goal()

			if param.get('goal_selection_mode') =='db' or param.get('goal_selection_mode') =='random' :
				self.goal_image = self.test_images[self.current_goal_idx].reshape(1, param.get('image_size'), param.get('image_size'), param.get('image_channels'))
				self.goal_code  = self.models.encoder.predict(self.goal_image)
			elif param.get('goal_selection_mode') =='som':
				self.goal_code  = self.models.goal_som._weights[self.current_goal_x, self.current_goal_y].reshape(1, param.get('code_size'))
			else:
				print ('wrong goal selection mode, exit!')
				sys.exit(1)


			# choose random motor commands from time to time
			ran = random.random()
			if ran < param.get('random_cmd_rate') or param.get('goal_selection_mode') =='random': #or self.models.memory_fwd.is_memory_still_not_full()
				self.random_cmd_flag = True
				print ('generating random motor command')
				p.x = random.uniform(utils.x_lims[0], utils.x_lims[1])
				p.y = random.uniform(utils.y_lims[0], utils.y_lims[1])
				self.prev_goal_idx=-1
				
			else:
				self.random_cmd_flag = False

				motor_pred = []
				if param.get('goal_selection_mode') == 'db':
					motor_pred = self.models.inv_model.predict(self.goal_code)
					print ('pred ', motor_pred, ' real ', self.test_pos[self.current_goal_idx])
				else:
					goal_decoded = self.models.decoder.predict(self.goal_code)
					motor_pred = self.models.inv_model.predict(self.goal_code)
				image_pred = self.models.decoder.predict(self.models.fwd_model.predict(np.asarray(motor_pred)))

				noise_x = np.random.normal(0,0.02)
				noise_y = np.random.normal(0,0.02)
				print ('prediction ', motor_pred)
				p.x = utils.clamp_x(utils.unnormalise_x(motor_pred[0][0]+noise_x, param))
				p.y = utils.clamp_y(utils.unnormalise_y(motor_pred[0][1]+noise_y, param))
				#p = utils.clamp(utils.unnormalise(motor_pred[0])) # make it possible to add noise

			#print ('predicted utils.Position ', motor_pred[0], 'p+noise ', motor_pred[0][0]+noise_x, ' ' , motor_pred[0][1]+noise_y, ' utils.clamped ', p.x, ' ' , p.y, ' noise.x ', noise_x, ' n.y ', noise_y)

			p.z = int(-90)
			p.speed = int(1400)
			# generate movement
			self.create_simulated_data(self.prev_pos, p, param)
			# store the amplitude of this movement
			if not self.random_cmd_flag and (self.current_goal_idx == self.prev_goal_idx):
				self.intrinsic_motivation.log_last_movement(p, self.prev_pos)
			print ('current_p', p.x, ' ' , p.y)
			print ('prev_p', self.prev_pos.x, ' ', self.prev_pos.y)	
			# update the variables
			self.prev_pos.x=p.x
			self.prev_pos.y=p.y
			self.prev_pos.z=p.z
			self.prev_pos.speed=p.speed


			# plot the explored points and the utils.Position of the goals
			if self.iteration % param.get('plot_exploration_iter') == 0:
				if param.get('goal_selection_mode') == 'db' or param.get('goal_selection_mode') == 'random':
					goals_pos = self.test_pos[0:(param.get('goal_size')*param.get('goal_size'))]
				elif param.get('goal_selection_mode') == 'som':
					goals_pos = self.models.inv_model.predict(self.models.goal_som._weights.reshape(len(self.models.goal_som._weights)*len(self.models.goal_som._weights[0]), len(self.models.goal_som._weights[0][0]) ))

				plot_exploration(positions=self.pos,goals=goals_pos,iteration=self.iteration,param=param)

			if self.iteration % param.get('im_pe_buffer_size_update_frequency') == 0:
				self.intrinsic_motivation.update_mse_dynamics(self.models.logger_fwd.get_last_mse())


			# update error dynamics of the current goal (it is supposed that at this moment the action is finished
			if len(self.img)>0 and not (param.get('goal_selection_mode') == 'random') and not (self.random_cmd_flag and len(self.intrinsic_motivation.slopes_pe_buffer)>0) :

				#cmd = utils.normalise(p) # [p.x/float(utils.x_lims[1]), p.y/float(utils.y_lims[1])]
				cmd = [ utils.normalise_x(p.x, param), utils.normalise_y(p.y, param)]
				prediction_code = self.models.fwd_model.predict(np.asarray(cmd).reshape((1,2)))

				prediction_error = np.linalg.norm(np.asarray(self.goal_code[:])-np.asarray(prediction_code[:]))
				if not (self.prev_goal_idx == -1):
					self.intrinsic_motivation.update_error_dynamics(self.current_goal_x, self.current_goal_y, prediction_error, _append=(self.current_goal_idx == self.prev_goal_idx))
				

			# fit models	
			if (len(self.img) > param.get('batch_size')) and (len(self.img) == len(self.pos)):

				observed_codes_batch = self.models.encoder.predict(np.asarray(self.img[-(param.get('batch_size')):]).reshape(param.get('batch_size'), param.get('image_size'), param.get('image_size'), param.get('image_channels'))  )
				observed_pos_batch = self.pos[-(param.get('batch_size')):]

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

				self.models.train_forward_code_model_on_batch( obs_and_mem_pos, obs_and_mem_img_codes, param)
				self.models.train_inverse_code_model_on_batch( obs_and_mem_img_codes, obs_and_mem_pos, param)

				#train_autoencoder_on_batch(self.autoencoder, self.encoder, self.decoder, np.asarray(self.img[-32:]).reshape(32, self.image_size, self.image_size, self.channels), batch_size=self.batch_size, cae_epochs=5)
			if not self.random_cmd_flag:
				self.prev_goal_idx = self.current_goal_idx	
			self.iteration = self.iteration+1
		print ('Saving models')
		self.save_models(param)

	def create_simulated_data(self, pos, cmd, param):
		self.lock.acquire()
		a = [int(pos.x), int(pos.y)]
		b = [int(cmd.x),int(cmd.y)]

		tr = self.cam_sim.get_trajectory(a,b)
		trn = self.cam_sim.get_trajectory_names(a,b)
		#print ('image size ', str(param.get('image_size')))
		rounded  = self.cam_sim.round2mul(tr,5) # only images every 5mm
		for i in range(len(tr)):
			#pp = utils.Position()
			#pp.x = float(rounded[i][0])
			#pp.y = float(rounded[i][1])
			#pp.z = -90
			#pp.speed = 1400

			#print ('pp ',pp)
			#self.pos.append(utils.normalise(pp))
			self.pos.append([utils.normalise_x(float(rounded[i][0]), param), utils.normalise_y(float(rounded[i][1]), param) ])
			#self.cmd.append([float(int(cmd.x)) / utils.x_lims[1], float(int(cmd.y)) / utils.y_lims[1]] )
			self.cmd.append([utils.normalise_x(float(int(cmd.x)), param), utils.normalise_y(float(int(cmd.y)), param) ])
			#self.cmd.append( utils.normalise(cmd) )
			cv2_img = cv2.imread(trn[i])#,1 )
			cv2.imshow('image',cv2_img)
			if param.get('image_channels') ==1:
				cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
			cv2_img = cv2.resize(cv2_img,(param.get('image_size'), param.get('image_size')), interpolation = cv2.INTER_LINEAR)
			cv2_img = cv2_img.astype('float32') / 255
			cv2_img.reshape(1, param.get('image_size'), param.get('image_size'), param.get('image_channels'))
			self.img.append(cv2_img)


			# update memory 
			# first update the memory, then update the models
			observed_pos = self.pos[-1]
			observed_img = cv2_img
			observed_img_code = np.asarray(self.models.encoder.predict(observed_img.reshape(1, param.get('image_size'), param.get('image_size'), param.get('image_channels')))).reshape(param.get('code_size'))
			self.models.memory_fwd.update(observed_pos, observed_img_code)
			self.models.memory_inv.update(observed_img_code, observed_pos)

		self.lock.release()


	def save_models(self, param):
		#self.lock.acquire()
		self.models.save_models(param)
		self.models.save_logs(self.parameters)
		
		self.intrinsic_motivation.get_linear_correlation_btw_amplitude_and_pe_dynamics()
		self.intrinsic_motivation.save_im()
		self.intrinsic_motivation.plot_slopes()
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
		sys.exit(1)

	def get_starting_pos(self):
		p = utils.Position()
		p.x = 0.0
		p.y = 0.0
		p.z = -50.0
		p.speed = 1400.0
		return utils.normalise(p, self.parameters)

	def goto_starting_pos(self):
		p = self.get_starting_pos()
		self.create_simulated_data(self.prev_pos, p, self.parameters)
		self.prev_pos=p



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
			memories = pickle.load(memory_file)
			print ('converting data...')
			count = 0
			for memory in memories:
				image = memory['image']
				if (channels == 1) and (image.ndim == 3):
					image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
