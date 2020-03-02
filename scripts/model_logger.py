import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import os

class Logger:

    def __init__(self, param, name):
        self.name = name
        self.parameters = param
        # contains the list of MSE calculation over time
        self.mse = []

        #self.count_of_changed_memory_elements = []
        
        self.input_variances= []
        self.output_variances= []
        #self.learning_progress = []

    def store_log(self, mse = [], input_var = [], output_var = []): #, count_of_changed_memory_elements = [], learning_progress = []):
        self.mse.append(mse)
        self.input_variances.append(input_var)
        self.output_variances.append(output_var)
        #self.count_of_changed_memory_elements.append(count_of_changed_memory_elements)
        #self.learning_progress.append(deepcopy(learning_progress))
        #print (str(self.learning_progress))

    def store_log(self, mse = []):
        self.mse.append(mse)

    def get_last_mse(self):
        if len(self.mse)>0:
            return self.mse[-1]
        return -1


    def get_iteration_count(self):
        return len(self.mse)

    def save_log(self):
        exp_name = 'mse_' + self.name
        np.save(os.path.join(self.parameters.get('results_directory'), exp_name), self.mse)

    def plot_mse(self, save = True, show = False):
        fig2 = plt.figure(figsize=(10, 10))
        plt.plot(self.mse)
        plt_name = 'MSE ' + self.name
        plt.title(plt_name)
        plt.ylabel('MSE')
        plt.xlabel('Time')

        if save:
            filename = self.parameters.get('results_directory')+'/plots/' + plt_name + '.jpg'
        plt.savefig(filename)
        if show:
            plt.show()
        plt.close()
