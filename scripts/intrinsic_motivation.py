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
import copy


class IntrinsicMotivation():

	def __init__(self, param):
		self.param = param

		# keep track of the goals that have been selected over time
		self.history_selected_goals = []

		if self.param.get('goal_size') <=0:
			print ("Error: goal_size <=0")
			sys.exit(1)

		# For each goal, store the experienced prediction error into a buffer. The buffer size can or cannot be dynamic (param.im_fixed_pe_buffer_size)
		self.buffer_pe_on_goal = []
		for i in range(self.param.get('goal_size')*self.param.get('goal_size')):
			self.buffer_pe_on_goal.append([])
		# for each goal, keep track of the trend of the PE_dynamics (slope of the regression over the pe_buffer)
		self.dyn_pe_on_goal = []
		# keep track of the current PE buffer size over time (all goals' buffers have the same size)
		self.history_buffer_pe_size = []
		# keep track of the MAX PE buffer size over time (all goals' buffers have the same size)
		self.history_buffer_pe_max_size = []

		# buffer of MSE fwd model. Its slope determines whether to increase or decrease the size of each pe_buffer
		self.buffer_mse = []
		# keep track of the trend of the MSE (slope of the regression over the mse_buffer, which is calculated on the test dataset - make it on the SOM?)
		self.dyn_mse = []

		# keep track of the displacement of the last movement 
		self.movements = []
		# keep track of the displacement of the last movement in a buffer for calculating the slopes
		self.buffer_mov = []
		# slopes of the movement amplitudes over time
		self.dyn_mov = []

		# standard deviation of the exploration noise, which varies according to the MSE dynamics
		self.std_dev_exploration_noise = []
		# range for the stddev of the exploration  nois
		self.std_dev_exploration_range = float( self.param.get('im_std_exploration_noise_max') - self.param.get('im_std_exploration_noise_min') )
		# presumed range for the mse dynamics (clamped at plus or minus im_std_exploration_mse_dynamics_range)
		self.dyn_mse_range= float( self.param.get('im_std_exploration_mse_dynamics_max') - self.param.get('im_std_exploration_mse_dynamics_min'))

		# correlatiosn betwee PE or MSE AND movements
		self.linregr_pe_vs_raw_mov = []
		self.linregr_pe_vs_slopes_mov = []
		self.linregr_mse_vs_raw_mov = []
		self.linregr_mse_vs_slopes_mov = []

		self.iterations_on_same_goal = 0

		print ("Intrinsic motivation initialised. PE buffer size: ", len(self.buffer_pe_on_goal))

	# log the last movement, calculate the regression
	def update_movement_dynamics(self, current_pos, previous_pos):
		movement = utils.distance(current_pos, previous_pos)
		self.movements.append(movement) # this will log all the movements
		self.buffer_mov.append(movement) # this is a moving buffer
		# if buffer size is greater than threshold, pop first elements
		while len(self.buffer_mov) > self.param.get('im_movements_buffer_size'):
			self.buffer_mov.pop(0)  # remove first element
		# calculate dynamics
		if len(self.buffer_mov) < 2:  # not enough prediction error to calculate the regression
			self.dyn_mov.append(0)
		else:
			# get the slopes of the prediction error dynamics
			self.dyn_mov.append(utils.get_slope_of_regression(self.buffer_mov))  # add the slope of the regression

	# update the dynamics of the overall mean squared error (forward model) calculated on the test dataset
	def update_mse_dynamics(self, mse):
		#print ('updating MSE dynamics')
		self.buffer_mse.append(mse)
		# if buffer size is greater than threshold, pop first elements
		while len(self.buffer_mse) > self.param.get('im_mse_buffer_size'):
			self.buffer_mse.pop(0)  # remove first element

		if len(self.buffer_mse) < 2:  # not enough prediction error to calculate the regression
			self.dyn_mse.append(0)
			# update the std_dev of the exploration noise
			if len(self.std_dev_exploration_noise) == 0:
				self.std_dev_exploration_noise.append(self.param.get('im_std_exploration_noise_initial'))
			else:
				self.std_dev_exploration_noise.append(self.std_dev_exploration_noise[-1])
		else:
			# get the slopes of the dynamics of the mean squared error over the test dataset
			self.dyn_mse.append(utils.get_slope_of_regression(self.buffer_mse))

			# increase/decrease by step
			if self.param.get('im_std_exploration_use_step'):
				# update stddev of the exploration noise
				if self.dyn_mse[-1] > 0:
					self.std_dev_exploration_noise.append(self.std_dev_exploration_noise[-1] + self.param.get('im_std_exploration_noise_step'))
					if self.std_dev_exploration_noise[-1] > self.param.get('im_std_exploration_noise_max'):
						self.std_dev_exploration_noise[-1] = self.param.get('im_std_exploration_noise_max')
				else:
					self.std_dev_exploration_noise.append(self.std_dev_exploration_noise[-1] - self.param.get('im_std_exploration_noise_step'))
					if self.std_dev_exploration_noise[-1] < self.param.get('im_std_exploration_noise_min'):
						self.std_dev_exploration_noise[-1] = self.param.get('im_std_exploration_noise_min')
			else:
				if self.dyn_mse[-1]< self.param.get('im_std_exploration_mse_dynamics_min'):
					self.std_dev_exploration_noise.append(self.param.get('im_std_exploration_noise_min'))
				elif self.dyn_mse[-1]> self.param.get('im_std_exploration_mse_dynamics_max'):
					self.std_dev_exploration_noise.append(self.param.get('im_std_exploration_noise_max'))
				else:
					#d :d_range = m: m_range
					#d = m*d_r/m_r
					std_dev_increase = (self.dyn_mse[-1] - self.param.get('im_std_exploration_mse_dynamics_min')) * self.std_dev_exploration_range / self.dyn_mse_range
					std_expl = self.param.get('im_std_exploration_noise_min') + std_dev_increase
					self.std_dev_exploration_noise.append(std_expl)

		# update the size of the buffer of the prediction error for each goal. This is done here to have a lower pace
		# (the same frequency of the MSE update).
		self.update_size_of_buffer_pe()

	def update_size_of_buffer_pe(self):
		if len(self.history_buffer_pe_max_size)==0 or self.param.get('im_size_buffer_pe_fixed'):
			self.history_buffer_pe_max_size.append(self.param.get('im_size_buffer_pe_initial'))
		else:
			new_buffer_size = copy.deepcopy(self.history_buffer_pe_max_size[-1])
			if self.dyn_mse[-1] > 0: # if mse is increasing, increase size of the pe buffer
				if new_buffer_size < self.param.get('im_size_buffer_pe_max'):
					new_buffer_size = new_buffer_size + 1 # or decrease?
			else:
				if new_buffer_size > self.param.get('im_size_buffer_pe_min'):
					new_buffer_size = new_buffer_size - 1 # or increase?
			self.history_buffer_pe_max_size.append(new_buffer_size)


	# update the error dynamics for each gaol
	def update_error_dynamics(self, goal_id_x, goal_id_y, prediction_error, _append=True):

		if len(self.history_buffer_pe_max_size) == 0:
			print ('Error in im.update_error_dynamics(). You need to call before im.update_pe_buffer_size()!')
			sys.exit(1)
		print ('updating error dynamics')
		goal_id = utils.get_goal_id(goal_id_x, goal_id_y, self.param)

		# append the current predction error to the current goal
		self.buffer_pe_on_goal[goal_id].append(prediction_error)

		# for each goal
		# - check that all the buffers are within the max buffer size
		# - compute regression on prediction error and save the slope (trend of error dynamics)
		slope_pe_dynamics = []
		pe_buffer_size_h = []
		for i in range(self.param.get('goal_size') * self.param.get('goal_size')):

			while len(self.buffer_pe_on_goal[i]) > self.history_buffer_pe_max_size[-1] and len(self.buffer_pe_on_goal[i]) > 0:
				self.buffer_pe_on_goal[i].pop(0)  # remove first element

			# not enough prediction error to calculate the regression?
			if len(self.buffer_pe_on_goal[i]) < self.param.get('im_size_buffer_pe_minimum_nr_of_sample_for_regression'):
				slope_pe_dynamics.append(0)
			else:
				# get the slopes of the prediction error dynamics
				print ('calculating regression on goal ', str(i))
				slope_pe_dynamics.append(utils.get_slope_of_regression(self.buffer_pe_on_goal[i]))

			pe_buffer_size_h.append(len(self.buffer_pe_on_goal[i]))

		if _append:
			# keep track of the goal that have been selected
			self.history_selected_goals.append(goal_id)
			# keep track of the buffer size for each goal
			self.history_buffer_pe_size.append(pe_buffer_size_h)
			# store the slopes of the prediction error dynamics for each goal
			self.dyn_pe_on_goal.append(slope_pe_dynamics)
			#print ('slopes', self.dyn_pe_on_goal[-1])


	def select_goal(self):
		goal_idx = self.get_best_goal_index()
		goal_x =int( goal_idx / self.param.get('goal_size')  )
		goal_y = goal_idx % self.param.get('goal_size')
		print ('Goal ID: ', goal_idx, ' som_x: ', goal_x, ' som_y: ', goal_y)
		return goal_idx, goal_x, goal_y

	def get_random_goal(self):
		goal_idx = random.randint(0, self.param.get('goal_size') * self.param.get('goal_size') - 1)
		print ('IM. Selecting random goal idx ', goal_idx)
		return goal_idx

	# goal selection strategy
	# get the index of the goal associated with the lowest slope in the prediction error dynamics
	def get_best_goal_index(self):

		#ran = random.random()
		if random.random() < self.param.get('im_random_goal_prob') or len(self.history_selected_goals) == 0:
			self.iterations_on_same_goal = 0
			return self.get_random_goal()

		# what is the index of the last selected goal?
		last_goal_idx = copy.deepcopy(self.history_selected_goals[-1])

		#if (len(self.buffer_pe_on_goal[last_goal_idx]) < self.param.get('im_size_buffer_pe_minimum_nr_of_sample_for_regression')) or ((len(self.buffer_pe_on_goal[last_goal_idx]) < (self.param.get('im_size_buffer_pe_min'))) and not self.param.get('im_size_buffer_pe_fixed')):
		#	self.iterations_on_same_goal = self.iterations_on_same_goal+1
		#	return last_goal_idx
		pe_slopes = copy.deepcopy(self.dyn_pe_on_goal[-1])
		print ('pe slopes', pe_slopes)
		print('self.iterations_on_same_goal ', self.iterations_on_same_goal)
		if pe_slopes[last_goal_idx] < 0 and np.fabs(pe_slopes[last_goal_idx]) > float(self.param.get('im_epsilon_error_dynamics')):
			return last_goal_idx
		else:
			if self.iterations_on_same_goal < self.param.get('im_min_iterations_on_same_goal'):
				self.iterations_on_same_goal = self.iterations_on_same_goal + 1
				return last_goal_idx
			else:
				self.iterations_on_same_goal = 0
				indexes = np.argsort(pe_slopes)
				for i in range(len(indexes)):
					if indexes[i] == last_goal_idx or pe_slopes[i] > 0:
						pass
					else:
						return copy.deepcopy( indexes[i] )

		self.iterations_on_same_goal = 0
		return random.randint(0, self.param.get('goal_size') * self.param.get('goal_size') - 1)
		#return np.argmin(pe_slopes)



	# get the standard deviation of the exploration noise, which varies according to the PE dynamics
	def get_std_dev_exploration_noise(self):
		if len(self.std_dev_exploration_noise) == 0:
			return self.param.get('im_std_exploration_noise_initial')
		return self.std_dev_exploration_noise[-1]


	def get_linear_correlation_btw_amplitude_and_mse_dynamics(self):
		# mse_buffer is updated at a slower rate than movements recording. Interpolate to match the sizes
		x = np.arange(0, len(self.dyn_mse))
		y = self.dyn_mse
		f = interpolate.interp1d(x, y,fill_value="extrapolate")
		x_correct= np.linspace(0, len(self.dyn_mse) - 1, num=len(self.movements))

		self.interpolated_slopes_mse_buffer= f(x_correct)

		#self.pearson_corr_mse_raw = pearsonr(np.asarray(self.slopes_mse_buffer), np.asarray(self.movements_amplitude))
		self.linregr_mse_vs_raw_mov = linregress(np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.movements))
		print ('Pearson correlation btw MSE and raw movements', self.linregr_mse_vs_raw_mov)

		#self.pearson_corr_mse_slopes = pearsonr(np.asarray(self.slopes_mse_buffer), np.asarray(self.slopes_movements))
		self.linregr_mse_vs_slopes_mov = linregress(np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.dyn_mov))
		#self.pearson_corr = pearsonr(slope_array[positive_indexes], movement_array[positive_indexes])
		print ('Pearson correlation btw MSE and slope of movements', self.linregr_mse_vs_slopes_mov)

		#self.plot_slopes_of_goals(self.param)
		return self.linregr_mse_vs_raw_mov, self.linregr_mse_vs_slopes_mov

	def get_linear_correlation_btw_amplitude_and_pe_dynamics(self):
		# first make a vector storing the pe_dynamics of the current goals over time
		#self.slopes_of_goals = np.asarray([[ self.slopes_pe_buffer[elem][self.goal_id_history[elem]] ] for elem in self.goal_id_history]).flatten()
		self.slopes_of_goals = []
		for i in range(len(self.history_selected_goals)):
		#	print ('i ', i)
		#	print ('self.goal_id_history[i] ',self.goal_id_history[i], ' shape ', np.asarray(self.goal_id_history).shape)
		#	print ('self.slopes_pe_buffer[self.goal_id_history[i]] ', self.slopes_pe_buffer[self.goal_id_history[i]], ' shpae ' , np.asarray(self.slopes_pe_buffer[self.goal_id_history[i]]).shape)
			self.slopes_of_goals.append(self.dyn_pe_on_goal[i][self.history_selected_goals[i]])

		#slope_array = np.asarray(self.slopes_of_goals)
		#movement_array= np.asarray(self.movements_amplitude)
		#self.positive_indexes = np.argwhere(slope_array>0)
		#print ('corre shape np.asarray(self.slopes_of_goals)', np.asarray(self.slopes_of_goals).shape, ' mov ',np.asarray(self.movements_amplitude).shape)
		self.linregr_pe_vs_raw_mov = linregress(np.asarray(self.slopes_of_goals), np.asarray(self.movements))
		#self.pearson_corr = pearsonr(slope_array[positive_indexes], movement_array[positive_indexes])
		print ('Pearson correlation btw current goals slope and raw movements', self.linregr_pe_vs_raw_mov)

		self.linregr_pe_vs_slopes_mov = linregress(np.asarray(self.slopes_of_goals), np.asarray(self.dyn_mov))
		#self.pearson_corr = pearsonr(slope_array[positive_indexes], movement_array[positive_indexes])
		print ('Pearson correlation btw current goals slope and slope of movements', self.linregr_pe_vs_slopes_mov)


		return self.linregr_pe_vs_raw_mov, self.linregr_pe_vs_slopes_mov


	def save_im(self):
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_mse_dynamics'), self.dyn_mse)
		np.save(os.path.join(self.param.get('results_directory'), 'im_interpolated_slopes_of_mse_dynamics'), self.interpolated_slopes_mse_buffer)
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_pe_dynamics'), self.dyn_pe_on_goal)
		np.save(os.path.join(self.param.get('results_directory'), 'im_slopes_of_goals'), self.slopes_of_goals)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pe_max_buffer_size_history'), self.history_buffer_pe_max_size)
		np.save(os.path.join(self.param.get('results_directory'), 'im_goal_id_history'), self.history_selected_goals)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_pe_raw'), self.linregr_pe_vs_raw_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_pe_slopes'), self.linregr_pe_vs_slopes_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_mse_raw'), self.linregr_mse_vs_raw_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_pearson_corr_mse_slopes'), self.linregr_mse_vs_slopes_mov)
		np.save(os.path.join(self.param.get('results_directory'), 'im_std_dev_exploration_noise'), self.std_dev_exploration_noise)

	def plot_slopes(self, save=True):
		fig = plt.figure(figsize=(10, 10))
		num_goals = self.param.get('goal_size') * self.param.get('goal_size')
		ax1 = plt.subplot(num_goals + 1, 1, 1)
		plt.plot(self.history_selected_goals)
		plt.ylim(1, num_goals)
		plt.ylabel('goal id')
		plt.xlabel('time')
		ax1.set_yticks(np.arange(0, num_goals))
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		# data = np.transpose(log_lp)
		data = np.transpose(self.dyn_pe_on_goal)
		#data = self.slopes_pe_buffer
		for i in range(0, num_goals):
			ax = plt.subplot(num_goals + 1, 1, i + 2)
			plt.plot(data[i])
			plt.ylabel('g{}'.format(i))

		if save:
			plt.savefig(self.param.get('results_directory') + '/plots/im_slopes_pe_buffer.jpg')
		if self.param.get('show_plots'):
			plt.show()
		plt.close()

	def plot_correlations(self, save=True):
		fig = plt.figure(figsize=(10, 20))

		ax1 = plt.subplot(4, 1, 1)
		plt.scatter(np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.movements), s=1)
		plt.title('MSE dynamics VS movement distances')
		string = 'Pearson\'s r=' + str(self.linregr_mse_vs_raw_mov.rvalue) + '\np<' + str(self.linregr_mse_vs_raw_mov.pvalue)
		plt.text(0.65, 0.75, string, transform=ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_mse_vs_raw_mov.intercept + self.linregr_mse_vs_raw_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')


		ax1 = plt.subplot(4, 1, 2)
		plt.scatter(np.asarray(self.interpolated_slopes_mse_buffer), np.asarray(self.dyn_mov), s=1)
		plt.title('MSE dynamics VS movement distances dynamics')
		string = 'Pearson\'s r=' + str(self.linregr_mse_vs_slopes_mov.rvalue) + '\np<' + str(self.linregr_mse_vs_slopes_mov.pvalue)
		plt.text(0.65, 0.75, string, transform = ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_mse_vs_slopes_mov.intercept + self.linregr_mse_vs_slopes_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')


		ax1 = plt.subplot(4, 1, 3)
		plt.scatter(np.asarray(self.slopes_of_goals), np.asarray(self.movements), s=1)
		plt.title('Current Goal PE dynamics VS movement distances')
		string = 'Pearson\'s r=' + str(self.linregr_pe_vs_raw_mov.rvalue) + '\np<' + str(self.linregr_pe_vs_raw_mov.pvalue)
		plt.text(0.65, 0.75, string, transform = ax1.transAxes)
		x_vals = np.array(ax1.get_xlim())
		y_vals = self.linregr_pe_vs_raw_mov.intercept + self.linregr_pe_vs_raw_mov.slope * x_vals
		plt.plot(x_vals, y_vals, '--', color='r')

		ax1 = plt.subplot(4, 1, 4)
		plt.scatter(np.asarray(self.slopes_of_goals), np.asarray(self.dyn_mov), s=1)
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
		plt.plot(self.history_buffer_pe_max_size)
		plt.ylabel('Max PE buffer size')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		data = np.transpose(self.history_buffer_pe_size)
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
		ax1 = plt.subplot(6, 1, 1)
		plt.plot(self.slopes_of_goals)
		plt.ylabel('Slope PE selected goal')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		#print ('movement amplitude ', self.movements_amplitude)
		ax1 = plt.subplot(6, 1, 2)
		plt.plot(self.movements)
		plt.ylabel('Movement ampl')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(6, 1, 3)
		plt.plot(self.dyn_mov)
		plt.ylabel('Slopes of mov')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(6, 1, 4)
		plt.plot(self.std_dev_exploration_noise)
		plt.ylabel('stddev expl noise')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(6, 1, 5)
		plt.plot(self.interpolated_slopes_mse_buffer)
		plt.ylabel('Int.Slopes MSE buff')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		ax1 = plt.subplot(6, 1, 6)
		plt.plot(self.history_buffer_pe_max_size)
		plt.ylabel('Max PE buffer size')
		plt.xlabel('time')
		ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

		if save:
			plt.savefig(self.param.get('results_directory')+'/plots/im_slopes_of_goals.jpg')
		if self.param.get('show_plots'):
			plt.show()
		plt.close()
