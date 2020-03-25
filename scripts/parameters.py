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
            #
            'directory_main': '',
            'directory_results': '',
            'directory_models': '',
            'directory_pretrained_models': '',
            'directory_plots': '',
            'directory_romi_dataset': '',

            # design of experiments
            'fixed_goal_som': True,
            'fixed_expl_noise': True,
            'random_cmd_rate': 0.02,
            'single_run_duration': 5000,
            'doe_experiment_id': 0, # in case of multiple experiments with different configuration
            'run_id': 0, # in case of multiple runs of the same experiemtn, this identifies the current iteration
            'use_pretrained_cae': True,
            'show_plots': False,
            'save_data_every_x_iteration': 200,
            'verbosity_level': 1,
            'plot_exploration_iter': 200, # plot scatter plot every x iteartions

            # CAE and SOM paraeters
            'train_cae_offline': False,
            'train_som_offline': False,
            'image_size': 64,
            'image_resize':False, # resize images to specified image_size? only if you are not sure that input images are of the desired size
            'image_channels': 1,
            'code_size': 32, # dimensions of the latent space of convoltioanl autoencoder
            'goal_size': 3, # SOMs size. There will be goal_size*goal_size  goal. TODO: improve this
            'reduce_som_learning_rate_factor':2000,

            # other normalisaition and ANN parametrs
            'normalise_with_zero_mean': False,
            'load_data_reshape': True,
            'batch_size':16, # for the fwd/inv models update
            'epochs': 1, # online fwd/inv models
            'goal_selection_mode':'som',
            'loss': 'mean_squared_error',
            'optimizer': 'adam',
            'memory_size': 1000,
            'memory_update_probability': 0.01, #0.002,
            'memory_update_strategy': MemUpdateStrategy.RANDOM.value,  # possible choices:  random, learning_progress

            'batchs_to_update_online': 3,

            'cae_filename': 'autoencoder.h5',
            'encoder_filename': 'encoder.h5',
            'decoder_filename':'decoder.h5',
            'cae_batch_size':32,
            'cae_epochs':40,
            'cae_max_pool_size': 2,
            'cae_conv_size': 3,

            'romi_input_dim': 2, # xy motors
            'romi_dataset_pkl': 'compressed_dataset.pkl',
            #'romi_test_data_step': 500,
            'romi_test_size': 200,
            'romi_seed_test_data': 10, # seed for the random number generator for picking up test data from training dataset

            'fwd_filename': 'forward_code_model.h5',
            'inv_filename': 'inverse_code_model.h5',
            'som_filename': 'goal_som.h5',

            # utility flags
            'random_cmd_flag': False,

            # intrinsic motivation related constants
            'im_movements_buffer_size': 50, # buffer size of the movement amplitudeds
            'im_mse_buffer_size': 10, # initial size of the mean squared error buffer (should be bigger than max PE_buffer_size
            'im_frequency_of_update_mse_dynamics': 40, # every how many iteration to wait for updating the pe_buffer_size according to the slope of the higher level?
            'im_size_buffer_pe_initial': 10, # initial size of the prediction error buffer
            'im_size_buffer_pe_min': 10, # max size of the prediction error buffer
            'im_size_buffer_pe_max': 50, # max size of the prediction error buffer
            'im_size_buffer_pe_fixed': False, # make the size of the prediction error buffer fixed or dependent on the dinamics of the FWD MSE
            'im_random_goal_prob': 0.01, # probability of selecting a random goal instead of the best one
            'im_epsilon_error_dynamics': 0.001, # switch goal when error slope is smaller than this value
            'im_size_buffer_pe_minimum_nr_of_sample_for_regression': 4,
            'im_min_iterations_on_same_goal': 50,

            'im_std_exploration_use_step': False, # increase/decrease stddev by step, or use predefined mappings
            'im_std_exploration_noise_if_fixed': 0.05,
            'im_std_exploration_noise_initial': 0.15,
            'im_std_exploration_noise_min': 0.01,
            'im_std_exploration_noise_max': 0.15,
            'im_std_exploration_noise_step': 0.01,
            'im_std_exploration_mse_dynamics_min': -0.001,
            'im_std_exploration_mse_dynamics_max': 0.0001

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
        pickle.dump(self.dictionary, open(os.path.join(self.get('directory_main'), 'parameters.pkl'), 'wb'),  protocol=2) # protcolo2 for compatibility with python2
        # save also as plain text file
        with open(os.path.join(self.get('directory_main'), 'parameters.txt'), 'w') as f:
            print(self.dictionary, file=f)
