from __future__ import print_function # added this to support usage of python2

import os
import numpy as np
from enum import Enum
import pickle

class MemUpdateStrategy(Enum):
    HIGH_LEARNING_PROGRESS = 0
    LOW_LEARNING_PROGRESS = 1
    RANDOM = 2


class Parameters:

    def __init__(self):
        self.dictionary = {
            'directory': '',
            'results_directory': '',
            'image_size': 64,
            'image_channels': 1,
            'code_size': 32, # dimensions of the latent space of convoltioanl autoencoder
            'goal_size': 3, # SOMs size. There will be goal_size*goal_size  goal. TODO: improve this

	   
            'normalise_with_zero_mean': False, 
	    'load_data_reshape': True, 
            'batch_size':16, # for the fwd/inv models update
            'epochs': 1, # online fwd/inv models
            'goal_selection_mode':'som',
            'exp_iteration': 0,

            'max_iterations':1000,

            'cae_filename': 'autoencoder.h5',
            'encoder_filename': 'encoder.h5',
            'decoder_filename':'decoder.h5',
            'cae_batch_size':32,
            'cae_epochs':2,
            'cae_max_pool_size': 2,
            'cae_conv_size': 3,

            'romi_input_dim': 2,
            'romi_dataset_pkl': 'romi_data/compressed_dataset.pkl',
            #'romi_test_data_step': 500,
            'romi_test_size': 100,
            'romi_seed_test_data': 10, # seed for the random number generator for picking up test data from training dataset

            'fwd_filename': 'forward_code_model.h5',
            'inv_filename': 'inverse_code_model.h5',
            'som_filename': 'goal_som.h5',

            'plot_exploration_iter': 50,

            'random_cmd_flag': False,
            'random_cmd_rate': 0.2,


            'im_competence_measure': 'euclidean',
            'im_decay_factor': 0.9,
            # there is a hierarchical dynamics monitoring: over the mean squared error of the fwd model (higher level) and over each goal (lower)
            # the slope of the MSE buffer controls the size of the goal PE buffer (in case im_fixed_pe_buffer_size is False)
            'im_mse_buffer_size': 50, # initial size of the mean squared error buffer (should be bigger than max PE_buffer_size
            'im_initial_pe_buffer_size': 10, # initial size of the prediction error buffer
            'im_min_pe_buffer_size': 5, # max size of the prediction error buffer
            'im_max_pe_buffer_size': 40, # max size of the prediction error buffer
            'im_fixed_pe_buffer_size': False, # make the size of the prediction error buffer fixed or dependent on the dinamics of the FWD MSE
            'im_pe_buffer_size_update_frequency': 20, # every how many iteration to wait for updating the pe_buffer_size according to the slope of the higher level?
            'im_random_goal_prob': 0.05, # probability of selecting a random goal instead of the best one

            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'memory_size': 500,
            'memory_update_probability': 0.01,
            'memory_update_strategy': MemUpdateStrategy.RANDOM.value,  # possible choices:  random, learning_progress
            #'batch_size': 32,
            'batchs_to_update_online': 3,
            'mse_test_dataset_fraction' : 20,  #   how many samples to use in the MSE calculations? dataset_size / this.
            'mse_calculation_step': 4, # calculate MSE every X model fits
            'experiment_repetition': -1,
            'verbosity_level': 1
        }

    def get(self, key_name):
        if key_name in self.dictionary.keys():
            return self.dictionary[key_name]
        else:
            print('Trying to access parameters key: '+ key_name+ ' which does not exist')

    def set(self, key_name, key_value):
        if key_name in self.dictionary.keys():
            print('Setting parameters key: ', key_name, ' to ', str(key_value))
            self.dictionary[key_name] = key_value
        else:
            print('Trying to modify parameters key: '+ key_name+ ' which does not exist')

    def save(self):
        pickle.dump(self.dictionary, open(os.path.join(self.get('directory'), 'parameters.pkl'), 'wb'),  protocol=2) # protcolo2 for compatibility with python2
        # save also as plain text file
        with open(os.path.join(self.get('directory'), 'parameters.txt'), 'w') as f:
            print(self.dictionary, file=f)
