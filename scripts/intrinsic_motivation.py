# author Guido Schillaci, Dr.rer.nat. - Scuola Superiore Sant'Anna
# Guido Schillaci <guido.schillaci@santannapisa.it>
import numpy as np
import random
import sys
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
from scipy.stats import linregress
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
import utils


class IntrinsicMotivation():

	def __init__(self, param):
		self.param = param

		# keep track of the goals that have been selected over time
		self.goal_id_history = [] 

		if self.param.get('goal_size') <=0:
			print ("Error: goal_size <=0")
			sys.exit(1)

		# For each goal, store the experienced prediction error into a buffer. The buffer size can or cannot be dynamic (param.im_fixed_pe_buffer_size)
		self.pe_buffer = []
		for i in range(self.param.get('goal_size')*self.param.get('goal_size')):
			self.pe_buffer.append([])
		# for each goal, keep track of the trend of the PE_dynamics (slope of the regression over the pe_buffer)
		self.slopes_pe_buffer = [] 

		# keep track of the MAX PE buffer size over time (all goals' buffers have the same size)
		self.pe_max_buffer_size_history = []

		# keep track of the current PE buffer size over time (all goals' buffers have the same size)
		self.pe_buffer_size_history = []

		# buffer of MSE fwd model. Its slope determines whether to increase or decrease the size of each pe_buffer
		self.mse_buffer = [] 
		# keep track of the trend of the MSE (slope of the regression over the mse_buffer, which is calculated on the test dataset - make it on the SOM?)
		self.slopes_mse_buffer = [] 

		# keep track of the displacement of the last movement 
		self.movements_amplitude = []

		# keep track of the displacement of the last movement in a buffer for calculating the slopes
		self.movements_buffer = []

		# slopes of the movement amplitudes over time
		self.slopes_movements = []

		self.linregr_pe_vs_raw_mov = []
		self.linregr_pe_vs_slopes_mov = []
		self.linregr_mse_vs_raw_mov = []
		self.linregr_mse_vs_slopes_mov = []



		#self.learning_progress= np.resize(self.learning_progress, (self.param.get('goal_size')*self.param.get('goal_size') ,))
		#for i in range (0, self.param.get('goal_size')*self.param.get('goal_size')):
	#		self.pe_history.append([0.0])
			#self.pe_derivatives.append([0.0])
		print ("Intrinsic motivation initialised. PE buffer size: ", len(self.pe_buffer))

	def select_goal(self):
		ran = random.random()
		goal_idx = 0
		if ran < self.param.get('im_random_goal_prob') or len(self.goal_id_history)==0:
			# select random goal
			goal_idx = random.randint(0, self.param.get('goal_size') * self.param.get('goal_size') - 1)
			print ('Intrinsic motivation: Selected random goal. Id: ', goal_idx)
		else:
		#if ran < 0.85 and np.sum(self.learning_progress)>0: # 70% of the times
			# select best goal
			goal_idx = self.get_best_goal_index()
			print ('Intrinsic motivation: Selected goal id: ', goal_idx)

		goal_x =int( goal_idx / self.param.get('goal_size')  )
		goal_y = goal_idx % self.param.get('goal_size')
		print ('Goal_SOM coords: ', goal_idx, ' x: ', goal_x, ' y: ', goal_y)
		return goal_idx, goal_x, goal_y

	#def update_max_pe_buffer_size(self):
		
	def get_goal_id(self, id_x, id_y):
		goal_id = int(id_x * self.param.get('goal_size') + id_y)
		if (goal_id <0 or goal_id>(self.param.get('goal_size')*self.param.get('goal_size'))):
			print ("Intrinsic motivation error, wrong goal id: ", goal_id)
			sys.exit(1)
		return goal_id

	def update_pe_buffer_size(self):
		if len(self.pe_max_buffer_size_history)==0:
			self.pe_max_buffer_size_history.append(self.param.get('im_initial_pe_buffer_size'))

		else:
			if self.param.get('im_fixed_pe_buffer_size'):
				self.pe_max_buffer_size_history.append(self.param.get('im_initial_pe_buffer_size'))
			else:
				new_buffer_size = self.pe_max_buffer_size_history[-1]
				if self.slopes_mse_buffer[-1] > 0:
					if new_buffer_size < self.param.get('im_max_pe_buffer_size'):
						new_buffer_size = new_buffer_size + 1 # or decrease?
				else:
					if new_buffer_size > self.param.get('im_min_pe_buffer_size'):
						new_buffer_size = new_buffer_size - 1 # or increase?
				self.pe_max_buffer_size_history.append(new_buffer_size)

	# update the dynamics of the overall mean squared error (forward model) calculated on the test dataset
	def update_mse_dynamics(self, mse, _append=True):
		print ('updating MSE dynamics')
		self.mse_buffer.append(mse)

		if _append:
			if len(self.mse_buffer) < 2 : # not enough prediction error to calculate the regression
				self.slopes_mse_buffer.append(0)
			else:
				while len(self.mse_buffer) > self.param.get('im_mse_buffer_size'):
					self.mse_buffer.pop(0) #remove first element
			
				# get the slopes of the prediction error dynamics
				regr_x = np.asarray(range(len(self.mse_buffer))).reshape((-1,1))
				print ('calculating regression on mse')
				model = LinearRegression().fit(regr_x, np.asarray(self.mse_buffer))
				self.slopes_mse_buffer.append(model.coef_[0]) # add the slope of the regression
			self.update_pe_buffer_size()

	# update the error dynamics for each gaol
	def update_error_dynamics(self, goal_id_x, goal_id_y, prediction_error, _append=True):

		if len(self.pe_max_buffer_size_history)==0:
			print ('Error in im.update_error_dynamics(). You need to call before im.update_pe_buffer_size()!')
			sys.exit(1)
		print ('updating error dynamics')
		goal_id = self.get_goal_id(goal_id_x, goal_id_y)
		
		# append the current predction error to the current goal
		self.pe_buffer[goal_id].append(prediction_error)
		for i in range(self.param.get('goal_size')*self.param.get('goal_size')):
			if i != goal_id:
				if len(self.pe_buffer[i])==0:
					self.pe_buffer[i].append(0)
				else:
					self.pe_buffer[i].append(self.pe_buffer[i][-1])

		# for each goal
		# - check that all the buffers are within the max buffer size
		# - compute regression on prediction error and save the slope (trend of error dynamics)
		current_slopes_err_dynamics = []
		pe_buffer_size_h = []
		for i in range(self.param.get('goal_size')*self.param.get('goal_size')):
			while len(self.pe_buffer[i]) > self.pe_max_buffer_size_history[-1] :
				if len(self.pe_buffer[i])>0:
					self.pe_buffer[i].pop(0) #remove first element

			if len(self.pe_buffer[i]) < 2 : # not enough prediction error to calculate the regression
				current_slopes_err_dynamics.append(0)
			else:
				# get the slopes of the prediction error dynamics
				regr_x = np.asarray(range(len(self.pe_buffer[i]))).reshape((-1,1))
				print ('calculating regression on goal ', str(i))
				model = LinearRegression().fit(regr_x, np.asarray(self.pe_buffer[i]))
				current_slopes_err_dynamics.append(model.coef_[0]) # add the slope of the regression

			pe_buffer_size_h.append(len(self.pe_buffer[i]))
		self.pe_buffer_size_history.append(pe_buffer_size_h)

		if _append:
			# keep track of the goal that have been selected
			self.goal_id_history.append(goal_id)
			self.slopes_pe_buffer.append(current_slopes_err_dynamics)
			print ('slopes', self.slopes_pe_buffer[-1])
		


	# get the index of the goal associated with the lowest slope in the prediction error dynamics
	def get_best_goal_index(self):
		if len(self.goal_id_history)>0:
			if (len(self.pe_buffer[self.goal_id_history[-1]]) < (self.param.get('im_min_pe_buffer_size')  )) and not self.param.get('im_fixed_pe_buffer_size'):
			#	print('here')
				return self.goal_id_history[-1]
			#if (len(self.pe_buffer[self.goal_id_history[-1]]) < (self.param.get('im_max_pe_buffer_size') )) and not self.param.get('im_fixed_pe_buffer_size'):
			#	print('here here')
			#	return self.goal_id_history[-1]
			curr_slopes = self.slopes_pe_buffer[-1]
			#print ('curr slope ', curr_slopes[self.goal_id_history[-1]] )
			#print ('curr goal ', self.goal_id_history[-1])
			if curr_slopes[self.goal_id_history[-1]] < 0:
				if np.abs(curr_slopes[self.goal_id_history[-1]]) > self.param.get('im_epsilon_error_dynamics'):
					#print('here here here')
					return self.goal_id_history[-1]
				else:
					return np.argmin(self.slopes_pe_buffer[-1])
					#idx = np.argmin(self.slopes_pe_buffer[-1])
					#if idx == self.goal_id_history[-1]:
					#	return random.randint(0, self.param.get('goal_size') * self.param.get('goal_size') - 1)
					#indexes = np.argsort(self.slopes_pe_buffer[-1])
					#if indexes[0] == self.goal_id_history[-1]:
					#	return indexes[1]
					#else:
					#	return indexes[0]

		return random.randint(0, self.param.get('goal_size') * self.param.get('goal_size') - 1)
		#return np.argmin(self.slopes_pe_buffer[-1])
		#return np.argmax(self.slopes_pe_buffer[-1])

	def log_last_movement(self, pos_a, pos_b):
		movement = utils.distance(pos_a,pos_b)
		self.movements_amplitude.append(movement) # this will log all the movements
		self.movements_buffer.append(movement) # this is a moving buffer
		if len(self.movements_buffer) < 2:  # not enough prediction error to calculate the regression
			self.slopes_movements.append(0)
		else:
			while len(self.movements_buffer) > self.param.get('im_movements_buffer_size'):
				self.movements_buffer.pop(0)  # remove first element

			# get the slopes of the prediction error dynamics
			regr_x = np.asarray(range(len(self.movements_buffer))).reshape((-1, 1))
			print ('calculating regression on movement buffer')
			model = LinearRegression().fit(regr_x, np.asarray(self.movements_buffer))
			self.slopes_movements.append(model.coef_[0])  # add the slope of the regression

	def get_linear_correlation_btw_amplitude_and_mse_dynamics(self):
		# mse_buffer is updated at a slower rate than movements recording. Interpolate to match the sizes
		x = np.arange(0, len(self.slopes_mse_buffer))
		y = self.slopes_mse_buffer
		f = interpolate.interp1d(x, y,fill_value="extrapolate")
		x_correct= np.linspace(0, len(self.slopes_mse_buffer)-1, num=len(self.movements_amplitude) )

		self.interpolated_slopes_mse_buffer= f(x_correct)

		#self.pearson_corr_mse_raw = pearsonr(np.asarray(self.slopes_mse_buffer), np.asarray(self.movements_amplitude))
		self.linregr_mse_vs_raw_mov = linregress(np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.movements_amplitude))
		print ('Pearson correlation btw MSE and raw movements', self.linregr_mse_vs_raw_mov)

		#self.pearson_corr_mse_slopes = pearsonr(np.asarray(self.slopes_mse_buffer), np.asarray(self.slopes_movements))
		self.linregr_mse_vs_slopes_mov = linregress(np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.slopes_movements))
		#self.pearson_corr = pearsonr(slope_array[positive_indexes], movement_array[positive_indexes])
		print ('Pearson correlation btw MSE and slope of movements', self.linregr_mse_vs_slopes_mov)

		#self.plot_slopes_of_goals(self.param)
		return self.linregr_mse_vs_raw_mov, self.linregr_mse_vs_slopes_mov

	def get_linear_correlation_btw_amplitude_and_pe_dynamics(self):
		# first make a vector storing the pe_dynamics of the current goals over time
		#self.slopes_of_goals = np.asarray([[ self.slopes_pe_buffer[elem][self.goal_id_history[elem]] ] for elem in self.goal_id_history]).flatten()
		self.slopes_of_goals = []
		for i in range(len(self.goal_id_history)):
		#	print ('i ', i)
		#	print ('self.goal_id_history[i] ',self.goal_id_history[i], ' shape ', np.asarray(self.goal_id_history).shape)
		#	print ('self.slopes_pe_buffer[self.goal_id_history[i]] ', self.slopes_pe_buffer[self.goal_id_history[i]], ' shpae ' , np.asarray(self.slopes_pe_buffer[self.goal_id_history[i]]).shape)
			self.slopes_of_goals.append(self.slopes_pe_buffer[i][self.goal_id_history[i]] )

		#slope_array = np.asarray(self.slopes_of_goals)
		#movement_array= np.asarray(self.movements_amplitude)
		#self.positive_indexes = np.argwhere(slope_array>0)
		#print ('corre shape np.asarray(self.slopes_of_goals)', np.asarray(self.slopes_of_goals).shape, ' mov ',np.asarray(self.movements_amplitude).shape)
		self.linregr_pe_vs_raw_mov = linregress(np.asarray(self.slopes_of_goals), np.asarray(self.movements_amplitude))
		#self.pearson_corr = pearsonr(slope_array[positive_indexes], movement_array[positive_indexes])
		print ('Pearson correlation btw current goals slope and raw movements', self.linregr_pe_vs_raw_mov)

		self.linregr_pe_vs_slopes_mov = linregress(np.asarray(self.slopes_of_goals), np.asarray(self.slopes_movements))
		#self.pearson_corr = pearsonr(slope_array[positive_indexes], movement_array[positive_indexes])
		print ('Pearson correlation btw current goals slope and slope of movements', self.linregr_pe_vs_slopes_mov)


		return self.linregr_pe_vs_raw_mov, self.linregr_pe_vs_slopes_mov


	def save_im(self):
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_mse_dynamics'), self.slopes_mse_buffer)
		np.save(os.path.join(self.param.get('results_directory'), 'im_interpolated_slopes_of_mse_dynamics'), self.interpolated_slopes_mse_buffer)
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_pe_dynamics'), self.slopes_pe_buffer)
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_goals'), self.slopes_of_goals)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pe_max_buffer_size_history'), self.pe_max_buffer_size_history)
		np.save(os.path.join(self.param.get('results_directory'), 'im_goal_id_history'), self.goal_id_history)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_pe_raw'), self.linregr_pe_vs_raw_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_pe_slopes'), self.linregr_pe_vs_slopes_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_mse_raw'), self.linregr_mse_vs_raw_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_mse_slopes'), self.linregr_mse_vs_slopes_mov)

	def plot_slopes(self, param, save=True):
		fig = plt.figure(figsize=(10, 10))
		num_goals = self.param.get('goal_size') * self.param.get('goal_size')
		ax1 = plt.subplot(num_goals + 1, 1, 1)
		plt.plot(self.goal_id_history)
		plt.ylim(1, num_goals)
		plt.ylabel('goal id')
		plt.xlabel('time')
		ax1.set_yticks(np.arange(0, num_goals))
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		# data = np.transpose(log_lp)
		data = np.transpose(self.slopes_pe_buffer)
		#data = self.slopes_pe_buffer
		for i in range(0, num_goals):
			ax = plt.subplot(num_goals + 1, 1, i + 2)
			plt.plot(data[i])
			plt.ylabel('g{}'.format(i))

		if save:
			plt.savefig(self.param.get('results_directory') + '/plots/im_slopes_pe_buffer.jpg')
		if param.get('show_plots'):
			plt.show()
		plt.close()

	def plot_correlations(self, save=True):
		fig = plt.figure(figsize=(10, 20))

		ax1 = plt.subplot(4, 1, 1)
		plt.scatter(  np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.movements_amplitude), s=1)
		plt.title('MSE dynamics VS movement distances')
		string = 'Pearson\'s r=' + str(self.linregr_mse_vs_raw_mov.rvalue) + '\np<' + str(self.linregr_mse_vs_raw_mov.pvalue)
		plt.text(0.65, 0.75, string, transform=ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_mse_vs_raw_mov.intercept + self.linregr_mse_vs_raw_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')


		ax1 = plt.subplot(4, 1, 2)
		plt.scatter(  np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.slopes_movements), s=1)
		plt.title('MSE dynamics VS movement distances dynamics')
		string = 'Pearson\'s r=' + str(self.linregr_mse_vs_slopes_mov.rvalue) + '\np<' + str(self.linregr_mse_vs_slopes_mov.pvalue)
		plt.text(0.65, 0.75, string, transform = ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_mse_vs_slopes_mov.intercept + self.linregr_mse_vs_slopes_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')


		ax1 = plt.subplot(4, 1, 3)
		plt.scatter(  np.asarray(self.slopes_of_goals), np.asarray(self.movements_amplitude), s=1)
		plt.title('Current Goal PE dynamics VS movement distances')
		string = 'Pearson\'s r=' + str(self.linregr_pe_vs_raw_mov.rvalue) + '\np<' + str(self.linregr_pe_vs_raw_mov.pvalue)
		plt.text(0.65, 0.75, string, transform = ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_pe_vs_raw_mov.intercept + self.linregr_pe_vs_raw_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')

		ax1 = plt.subplot(4, 1, 4)
		plt.scatter(  np.asarray(self.slopes_of_goals), np.asarray(self.slopes_movements), s=1)
		plt.title('Current Goal PE dynamics VS movement distances dynamics')
		string = 'Pearson\'s r=' + str(self.linregr_pe_vs_slopes_mov.rvalue) + '\np<' + str(self.linregr_pe_vs_slopes_mov.pvalue)
		plt.text(0.65, 0.75, string, transform = ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_pe_vs_slopes_mov.intercept + self.linregr_pe_vs_slopes_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')

		if save:
			plt.savefig(self.param.get('results_directory')+'/plots/im_correlations.jpg')
		if self.param.get('show_plots'):
			plt.show()
		plt.close()



	def plot_buffer_size(self, save=True):
		num_goals = self.param.get('goal_size') * self.param.get('goal_size')

		fig = plt.figure(figsize=(10, 10))

		ax1 = plt.subplot(num_goals + 1, 1, 1)
		plt.plot(self.pe_max_buffer_size_history)
		plt.ylabel('Max PE buffer size')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		data = np.transpose(self.pe_buffer_size_history)
		#data = self.slopes_pe_buffer
		for i in range(0, num_goals):
			ax = plt.subplot(num_goals + 1, 1, i + 2)
			plt.plot(data[i])

			plt.ylabel('buf_size{}'.format(i))
			plt.xlabel('time')
			ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		if save:
			plt.savefig(self.param.get('results_directory')+'/plots/im_buffer_size.jpg')
		if self.param.get('show_plots'):
			plt.show()
		plt.close()


	def plot_slopes_of_goals(self, save=True):
		fig = plt.figure(figsize=(10, 20))
		num_goals= self.param.get('goal_size')*self.param.get('goal_size')
		ax1 = plt.subplot(5, 1, 1)
		plt.plot(self.slopes_of_goals)
		plt.ylabel('Slope PE_dyn select. goal')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		#print ('movement amplitude ', self.movements_amplitude)
		ax1 = plt.subplot(5, 1, 2)
		plt.plot(self.movements_amplitude)
		plt.ylabel('Movement ampl.')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(5, 1, 3)
		plt.plot(self.slopes_movements)
		plt.ylabel('Slopes of mov.')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(5, 1, 4)
		plt.plot(self.interpolated_slopes_mse_buffer)
		plt.ylabel('Int.Slopes MSE buff')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(5, 1, 5)
		plt.plot(self.pe_max_buffer_size_history)
		plt.ylabel('Max PE buffer size')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		if save:
			plt.savefig(self.param.get('results_directory')+'/plots/im_slopes_of_goals.jpg')
		if self.param.get('show_plots'):
			plt.show()
		plt.close()
