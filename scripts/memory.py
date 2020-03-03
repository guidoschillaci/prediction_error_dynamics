import numpy as np
import random
from parameters import Parameters, MemUpdateStrategy
from copy import deepcopy
import matplotlib as plt

class Memory:

    def __init__(self, param):
        self.parameters = param

        # compute the MSE every these steps
        self.mse_calculation_step = self.parameters.get('batch_size') * self.parameters.get('batchs_to_update_online')

        # content of the memory
        self.input_variables = []
        self.output_variables = []
        #self.greenhouse_index = []
        #self.sample_confidence_interval = []
        self.prediction_errors = [] # for each sample in the memory, store the prediction errors calculated at the last fits at times t and t-1
        self.learning_progress = [] # derivative of the prediction errors (for the moment, just simply pe(t) - pe(t-1)

    def is_memory_empty(self):
        if len(self.input_variables) == 0:
            return True
        else:
            return False

    def is_memory_still_not_full(self):
        if len(self.input_variables) < self.parameters.get('memory_size'):
            return True
        else:
            return False

    def update(self, input, output):

        counter_of_changed_elements = 0
        #print ('memory ', np.asarray(input).shape)
        if self.parameters.get('memory_size') != 0:
            # if the size of the stored samples has not reached the full size of the memory, then just append the samples
            if len(self.input_variables) < self.parameters.get('memory_size'):
                ran = random.random()
                if ran < self.parameters.get('memory_update_probability'):
                    self.input_variables.append(input)
                    self.output_variables.append(output)
                    self.prediction_errors.append([])
                    self.learning_progress.append(np.nan)
            else:
                #print ('self.parameters.getmemory_update_strategy) ' , self.parameters.get('memory_update_strategy'))
                if self.parameters.get('memory_update_strategy') == MemUpdateStrategy.RANDOM.value:
                    # iterate the memory and decide whether to assign the current sample to an element or not, with probability p
                    #for i in range(len(self.input_variables)):
                    i = random.randrange(0, len(self.input_variables))
                    ran = random.random()
                    if ran < self.parameters.get('memory_update_probability'):
                        self.input_variables[i] = input
                        self.output_variables[i] = output
                        self.prediction_errors[i] = []
                        self.learning_progress[i] = np.nan
                        #counter_of_changed_elements = counter_of_changed_elements + 1
                elif self.parameters.get('memory_update_strategy') == MemUpdateStrategy.LOW_LEARNING_PROGRESS.value:
                    # select the element with the highest or lowest learning progress (which of the two is the best?
                    # and substitute it with the new sample - with probability p
                    #for i in range(len(self.input_variables)):
                    ran = random.random()
                    if ran < self.parameters.get('memory_update_probability'):
                        index = self.learning_progress.index( np.nanmin (self.learning_progress)) # gives high plasticity?
                        self.input_variables[index] = input
                        self.output_variables[index] = output
                        self.prediction_errors[index] = []
                        self.learning_progress[index] = np.nan
                        #counter_of_changed_elements = counter_of_changed_elements +1
                elif self.parameters.get('memory_update_strategy') == MemUpdateStrategy.HIGH_LEARNING_PROGRESS.value:
                    # select the element with the highest or lowest learning progress (which of the two is the best?
                    # and substitute it with the new sample - with probability p
                    #for i in range(len(self.input_variables)):
                    ran = random.random()
                    if ran < self.parameters.get('memory_update_probability'):
                        index = self.learning_progress.index( np.nanmax (self.learning_progress)) # gives low plasticity?
                        self.input_variables[index] = input
                        self.output_variables[index] = output
                        self.prediction_errors[index] = []
                        self.learning_progress[index] = np.nan
                        #counter_of_changed_elements = counter_of_changed_elements +1
                else:
                    print ('Wrong parameter memory_update_strategy')
                counter_of_changed_elements = counter_of_changed_elements + 1
        return counter_of_changed_elements

    def get_variance(self):
        input_var = 0
        output_var = 0
        if len(self.input_variables) >0:
            input_var = np.var(self.input_variables)
            print ('input var  ' + str(input_var))
            output_var = np.var(self.output_variables)
            print ('output var  ' + str(output_var))
        return input_var, output_var

    def get_learning_progress(self):
        return self.learning_progress

    def update_learning_progress(self, model):
        if len(self.input_variables) >0:
            predictions = model.predict( np.asarray(self.input_variables) )
            for i in range (len  (self.output_variables) ):
                prediction_error = (np.linalg.norm(predictions[i] - self.output_variables[i]) ** 2)

                if len(self.prediction_errors[i]) >= 2:
                    self.learning_progress[i] = np.fabs(self.prediction_errors[i][-1] - self.prediction_errors[i][-2])
                    self.prediction_errors[i][-2] = deepcopy(self.prediction_errors[i][-1])
                    self.prediction_errors[i][-1] = prediction_error
                elif len(self.prediction_errors[i]) == 1:
                    self.prediction_errors[i].append(deepcopy(prediction_error))
                    self.learning_progress[i] = np.fabs(self.prediction_errors[i][-1] - self.prediction_errors[i][-2])
                elif len ( self.prediction_errors[i]) == 0:
                    self.prediction_errors[i].append(deepcopy(prediction_error))

    def plot_input_variables(self, iteration, goals=[], save=True):
        
        title = "mem_input_var"
        fig2 = plt.figure(figsize=(10, 10))
        # print (log_goal)
        if self.parameters.get('romi_input_dim') == 2:
            plt.scatter(np.transpose(self.input_variables)[0], np.transpose(self.input_variables)[1], s=1, color='g')
            if len(goals) > 0:
                plt.plot(np.asarray(goals[:, 0]).astype('float32'), np.asarray(goals[:, 1]).astype('float32'), 'ro')

            plt.xlabel('Pos x')
            plt.ylabel('Pos y')
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
        elif self.parameters.get('romi_input_dim') == 4:
            plt.subplot(1, 2, 1)
            plt.scatter(self.input_variables[:, 0], self.input_variables[:, 1], s=1, color='g')
            if len(goals) > 0:
                plt.plot(np.asarray(goals[:, 0]).astype('float32'), np.asarray(goals[:, 1]).astype('float32'), 'ro')
            plt.xlim(-0.4, 0.4)
            plt.ylim(-0.4, 0.4)
            plt.xlabel('Dim 0')
            plt.ylabel('Dim 1')
            plt.subplot(1, 2, 2)
            plt.scatter(self.input_variables[:, 0], self.input_variables[:, 1], s=1, color='g')
            if len(goals) > 0:
                plt.plot(np.asarray(goals[:, 2]).astype('float32'), np.asarray(goals[:, 3]).astype('float32'), 'ro')
            plt.xlim(-0.4, 0.4)
            plt.ylim(-0.4, 0.4)
            plt.xlabel('Dim 2')
            plt.ylabel('Dim 3')

        if save:
            filename = self.parameters.get('results_directory') + 'plots/plot_' + title + '_' + str(iteration) + '.jpg'
            plt.savefig(filename)
        if self.parameters.get('show_plots'):
            plt.show()
        plt.close()