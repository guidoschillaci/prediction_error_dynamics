import numpy as np
import datetime
import os
import utils
#from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.python.keras.models import Model, Sequential, load_model, Input
from tensorflow.python.keras.layers import Input, Dense, Dropout, Reshape, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
#from tensorflow.keras import optimizers

#import tensorflow as tf
import sys
import h5py

from minisom import MiniSom
import model_logger
import memory
from tensorflow.python.keras import backend as K

#import random
#from copy import deepcopy

#from tensorflow.python.keras.models import Sequential

#from tensorflow.python.keras.layers import LSTM
#from tensorflow.python.keras.layers import BatchNormalization
#from tensorflow.python.keras.activations import sigmoid
#from tensorflow.python.keras.activations import hard_sigmoid
#from tensorflow.python.keras.utils import plot_model


#from parameters import Parameters, MemUpdateStrategy

class Models:

    def __init__(self, param):

        self.time_start = datetime.datetime.now()

        self.parameters = param

        # initialise autoencoder, fwd and inv models
        self.autoencoder, self.encoder, self.decoder = self.load_autoencoder(self.parameters)
        self.fwd_model = self.load_forward_code_model(self.parameters)
        self.inv_model = self.load_inverse_code_model(self.parameters)
        self.goal_som = self.load_som(self.parameters)
        self.reduce_som_learning_rate(self.parameters.get('reduce_som_learning_rate_factor')) # by a factor of 1/10, if not otherwise specified
        # initialise memory (one per model - autoencoder is kept fixed for the moment)
        # how many elements to keep in memory?
        self.memory_size = self.parameters.get('memory_size')
        # probability of substituting an element of the memory with the current observation
        self.prob_update = self.parameters.get('memory_update_probability')
        self.memory_fwd = memory.Memory(param = self.parameters)
        self.memory_inv = memory.Memory(param = self.parameters)

        # initialise loggers
        self.logger_fwd = model_logger.Logger(param = self.parameters, name='fwd')
        self.logger_inv = model_logger.Logger(param = self.parameters, name='inv')


    def activation_positive_tanh(self, x, target_min = 0, target_max = 1):
        x02 = K.tanh(x) + 1  # x in range(0,2)
        scale = (target_max - target_min) / 2.
        return x02 * scale + target_min


    def load_autoencoder(self, param, train_images=None, train_offline=True):
        cae_file = param.get('directory_models') + param.get('cae_filename')
        e_file = param.get('directory_models') + param.get('encoder_filename')
        d_file = param.get('directory_models') + param.get('decoder_filename')
        #cae_file = './pretrained_models/' + param.get('cae_filename')
        #e_file = './pretrained_models/' + param.get('encoder_filename')
        #d_file = './pretrained_models/' + param.get('decoder_filename')

        autoencoder = []
        encoder = []
        decoder = []
        # if cae file already exists (i.e. cae has been already trained):
        if os.path.isfile(cae_file) and os.path.isfile(e_file) and os.path.isfile(
                d_file):
            # load convolutional autoencoder
            print ('Loading existing pre-trained autoencoder: ', cae_file)
            # clear tensorflow graph
            #utils.clear_tensorflow_graph()
            autoencoder = load_model(cae_file) # keras.load_model function

            # Create a separate encoder model
            encoder_inp = Input(shape=(param.get('image_size'), param.get('image_size'), param.get('image_channels')))
            encoder_layer = autoencoder.layers[1](encoder_inp)
            enc_layer_idx = utils.getLayerIndexByName(autoencoder, 'encoded')
            for i in range(2, enc_layer_idx + 1):
                encoder_layer = autoencoder.layers[i](encoder_layer)
            encoder = Model(encoder_inp, encoder_layer)
            if (param.get('verbosity_level') > 2):
                print (encoder.summary())
            # Create a separate decoder model
            decoder_inp = Input(shape=(param.get('code_size'),))
            decoder_layer = autoencoder.layers[enc_layer_idx + 1](decoder_inp)
            for i in range(enc_layer_idx + 2, len(autoencoder.layers)):
                decoder_layer = autoencoder.layers[i](decoder_layer)

            decoder = Model(decoder_inp, decoder_layer)
            if (param.get('verbosity_level') > 2):
                print (decoder.summary())
            print ('Autoencoder loaded')
        else: # otherwise train a new one
            print ('Could not find autoencoder files. Building and training a new one.')
            autoencoder, encoder, decoder = self.build_autoencoder(param)
            if train_offline:
                if train_images is None:
                    print ('I need some images to train the autoencoder')
                    sys.exit(1)
                self.train_autoencoder_offline(autoencoder, encoder, decoder, train_images, param)
        return autoencoder, encoder, decoder

    # build and compile the convolutional autoencoder
    def build_autoencoder(self, param):
        autoencoder = None
        input_img = Input(shape=(param.get('image_size'), param.get('image_size'), param.get('image_channels')), name='input')
        x = Conv2D(256, (param.get('cae_conv_size'), param.get('cae_conv_size')), activation='relu', padding='same')(input_img)  # tanh?
        x = MaxPooling2D((param.get('cae_max_pool_size'), param.get('cae_max_pool_size')), padding='same')(x)
        x = Conv2D(128, (param.get('cae_conv_size'), param.get('cae_conv_size')), activation='relu', padding='same')(x)
        x = MaxPooling2D((param.get('cae_max_pool_size'), param.get('cae_max_pool_size')), padding='same')(x)
        x = Conv2D(128, (param.get('cae_conv_size'), param.get('cae_conv_size')), activation='relu', padding='same')(x)
        x = MaxPooling2D((param.get('cae_max_pool_size'), param.get('cae_max_pool_size')), padding='same')(x)
        x = Flatten()(x)
        encoded = Dense(param.get('code_size'), name='encoded')(x)

        print  ('encoded shape ', encoded.shape)
        ims = 8
        first = True
        x = Dense(int(ims * ims), activation='relu')(encoded)
        x = Reshape(target_shape=(ims, ims, 1))(x)  # -12
        while ims != param.get('image_size'):
            x = Conv2D(int(ims * ims / 2), (param.get('cae_conv_size'), param.get('cae_conv_size')), activation='relu', padding='same')(x)
            x = UpSampling2D((param.get('cae_max_pool_size'), param.get('cae_max_pool_size')))(x)
            ims = ims * param.get('cae_max_pool_size')
        decoded = Conv2D(param.get('image_channels'), (param.get('cae_conv_size'), param.get('cae_conv_size')), activation='sigmoid', padding='same', name='decoded')(x)

        print ('decoded shape ', decoded.shape)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Create a separate encoder model
        encoder = Model(input_img, encoded)
        encoder.compile(optimizer='adam', loss='mean_squared_error')
        encoder.summary()

        # Create a separate decoder model
        decoder_inp = Input(shape=(param.get('code_size'),))
        #		decoder_inp = Input(shape=encoded.output_shape)
        enc_layer_idx = utils.getLayerIndexByName(autoencoder, 'encoded')
        print ('encoder layer idx ', enc_layer_idx)
        decoder_layer = autoencoder.layers[enc_layer_idx + 1](decoder_inp)
        for i in range(enc_layer_idx + 2, len(autoencoder.layers)):
            decoder_layer = autoencoder.layers[i](decoder_layer)
        decoder = Model(decoder_inp, decoder_layer)
        decoder.compile(optimizer='adam', loss='mean_squared_error')
        if (param.get('verbosity_level') > 2):
            decoder.summary()

        return autoencoder, encoder, decoder

    def train_autoencoder(self, autoencoder, encoder, decoder, train_data, param):
        #tensorboard_callback = TensorBoard(log_dir='./logs/cae', histogram_freq=0, write_graph=True,
        #                                   write_images=True)

        autoencoder.fit(train_data, train_data, epochs=param.get('cae_epochs'), batch_size=param.get('cae_batch_size'), shuffle=True,verbose=1)
                        #callbacks=[tensorboard_callback], verbose=1)

        autoencoder.save(param.get('directory_models')+ 'autoencoder.h5')
        encoder.save(param.get('directory_models') + 'encoder.h5')
        decoder.save(param.get('directory_models') + 'decoder.h5')
        print ('autoencoder trained and saved ')

    def load_forward_code_model(self, param):
        filename = param.get('directory_models') + param.get('fwd_filename')

        forward_model = []
        if os.path.isfile(filename):
            print ('Loading existing pre-trained forward code model: ', filename)
            forward_model = load_model(filename)
            print ('Forward code model loaded')
        else:
            print (' image_size load ', param.get('image_size'))
            forward_model = self.build_forward_code_model(param)
            print ('Forward model does not exist, yet. Built and compiled a new one')
        return forward_model

    def build_forward_code_model(self, param):
        print ('building forward code model...')

        # create fwd model layers
        cmd_fwd_inp = Input(shape=(param.get('romi_input_dim'),), name='fwd_input')
        #x = Dense(param.get('code_size'), activation=self.activation_positive_tanh)(cmd_fwd_inp)
        x = Dense(param.get('code_size'), activation='relu')(cmd_fwd_inp)
        #x = Dense(param.get('code_size') * 10, activation=self.activation_positive_tanh)(x)
        #x = Dense(param.get('code_size') * 10, activation=self.activation_positive_tanh)(x)
        x = Dense(param.get('code_size') * 10, activation='relu')(x)
        code = Dense(param.get('code_size'), name='output')(x)
        fwd_model = Model(cmd_fwd_inp, code)
        #sgd = optimizers.SGD(lr=0.0014, decay=0.0, momentum=0.8, nesterov=True)
        fwd_model.compile(optimizer='adadelta', loss='mean_squared_error')
        #fwd_model.compile(optimizer=sgd, loss='mean_squared_error')
        if (param.get('verbosity_level') > 2):
            print ('forward model')
            fwd_model.summary()
        return fwd_model


    def train_forward_code_model_on_batch(self, positions, codes):
        # tensorboard_callback = log(TensorBoard_dir='./logs/fwd_code', histogram_freq=0, write_graph=True, write_images=True)
        self.fwd_model.fit(positions, codes, epochs=self.parameters.get('epochs'), batch_size=self.parameters.get('batch_size'), verbose=1,
                          shuffle=True)  # , callbacks=[tensorboard_callback])
        print ('Forward code model updated')

    def load_inverse_code_model(self, param):
        filename = param.get('directory_models') + param.get('inv_filename')
        # build inverse model
        if os.path.isfile(filename):
            print ('Loading existing pre-trained inverse code model: ', filename)
            inverse_model = load_model(filename)
            print ('Inverse model loaded')
        else:
            inverse_model = self.build_inverse_code_model(param)
            print ('Inverse model does not exist, yet. Built and compiled a new one')
        return inverse_model

    def build_inverse_code_model(self, param):
        print ('building inverse code model...')

        input_code = Input(shape=(param.get('code_size'),), name='inv_input')
        x = Dense(param.get('code_size'), activation='relu')(input_code)
        x = Dense(param.get('code_size') * 10, activation='relu')(x)
        #x = Dropout(0.2)(x)
        x = Dense(param.get('code_size') * 10, activation='relu')(x)
        #x = Dropout(0.2)(x)
        #command = Dense(param.get('romi_input_dim'), activation=self.activation_positive_tanh, name='command')(x)
        command = Dense(param.get('romi_input_dim'), activation='sigmoid', name='command')(x)
        #command = Dense(param.get('romi_input_dim'), name='command')(x)

        inv_model = Model(input_code, command)
        #sgd = optimizers.SGD(lr=0.0014, decay=0.0, momentum=0.8, nesterov=True)
        #inv_model.compile(optimizer=sgd, loss='mean_squared_error')
        inv_model.compile(optimizer='adadelta', loss='mean_squared_error')
        if (param.get('verbosity_level') > 2):
            print ('inverse code model')
            inv_model.summary()
        return inv_model


    def train_inverse_code_model_on_batch(self, codes, motor_cmd):
        # tensorboard_callback = TensorBoard(log_dir='./logs/inv_code', histogram_freq=0, write_graph=True, write_images=True)
        self.inv_model.fit(codes, motor_cmd, epochs=self.parameters.get('epochs'), batch_size=self.parameters.get('batch_size'), verbose=1,
                          shuffle=True)  # , callbacks=[tensorboard_callback])#, callbacks=[showLR()])
        print ('Inverse code model trained on batch')


    def load_som(self, param, encoder=None, train_images=None):
        if not param.get('fixed_goal_som'):
            goal_som = MiniSom(param.get('goal_size'), param.get('goal_size'), param.get('code_size'), sigma=0.5, learning_rate=0.5)
            print ('Initialising goal SOM...')
            # goal_som.random_weights_init(train_images_codes)
            return goal_som


        filename = param.get('directory_models') + param.get('som_filename')
        #filename = './pretrained_models/' + param.get('som_filename')
        print ('Looking for som file: ', filename)
        goal_som = None
        if os.path.isfile(filename):
            print ('Loading existing trained SOM...')
            h5f = h5py.File(filename, 'r')
            weights = h5f['goal_som'][:]
            code_size = len(weights[0][0])
            h5f.close()
            print ('code_size read ', code_size)
            goal_som = MiniSom(param.get('goal_size'), param.get('goal_size'), param.get('code_size'))
            goal_som._weights = weights
            print (len(weights))
            print ('Goal SOM loaded! Number of goals: ', str(param.get('goal_size') * param.get('goal_size')))
        else:
            print ('Could not find Goal SOM files.')
            if encoder is None or train_images is None:
                print ('I need an encoder and some sample images to train a new SOM!')
                sys.exit(1)
            print ('Creating a new one')
            # creating self-organising maps for clustering the image codes <> the image goals

            # encoding test images
            print ('Encoding train images...')
            train_images_codes = encoder.predict(train_images)
            code_size = len(train_images_codes[0])

            goal_som = MiniSom(param.get('goal_size'), param.get('goal_size'), param.get('code_size'), sigma=0.5, learning_rate=0.5)
            print ('Initialising goal SOM...')
            goal_som.random_weights_init(train_images_codes)

            # plot_som_scatter( encoder, goal_som, train_images)

            print ('som quantization error: ', goal_som.quantization_error(train_images_codes))
            print("Training goal SOM...")
            goal_som.train_random(train_images_codes, 100)  # random training

            trained_som_weights = goal_som.get_weights().copy()
            som_file = h5py.File(filename, 'w')
            som_file.create_dataset('goal_som', data=trained_som_weights)
            som_file.close()
            print("SOM trained and saved!")
        return goal_som

    def reduce_som_learning_rate(self, factor = 10.0):
        self.goal_som._learning_rate = self.goal_som._learning_rate / factor

    def update_som(self, data, iterations=2):
        self.goal_som.train_batch(data, iterations, reinit_T=False)

    def save_logs(self, show=False):
        self.logger_fwd.save_log()
        self.logger_fwd.plot_mse(show=show)

        self.logger_inv.save_log()
        self.logger_inv.plot_mse(show=show)

    def save_models(self):

        self.autoencoder.save(self.parameters.get('directory_models')+'autoencoder.h5', overwrite=True)
        self.encoder.save(self.parameters.get('directory_models')+'encoder.h5', overwrite=True)
        self.decoder.save(self.parameters.get('directory_models')+'decoder.h5', overwrite=True)
        self.inv_model.save(self.parameters.get('directory_models')+'inv_model.h5', overwrite=True)
        self.fwd_model.save(self.parameters.get('directory_models')+'fwd_model.h5', overwrite=True)

        # save som
        som_weights = self.goal_som.get_weights().copy()
        som_file = h5py.File(self.parameters.get('directory_models')+'goal_som.h5', 'w')
        som_file.create_dataset('goal_som', data=som_weights)
        som_file.close()
