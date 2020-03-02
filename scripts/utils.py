
import gzip
import pickle
import cv2
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import time

x_lims=[0.0,750.0]
x_mean = (x_lims[1] - x_lims[0]) /2.0
y_lims=[0.0,750.0]
y_mean = (y_lims[1]-y_lims[0]) /2.0
z_lims=[-10.0,-90.0]
z_mean = (z_lims[1]-z_lims[0]) /2.0
speed_lim = 3400.0



def normalise_x(x, param):
	if param.get('normalise_with_zero_mean'):
		return  (x - x_mean) / x_lims[1]
	else:
		return x / x_lims[1]

def normalise_y(y, param):
	if param.get('normalise_with_zero_mean'):
		return  (y - y_mean) / y_lims[1]
	else:
		return y / y_lims[1]

def normalise_z(z, param):
	if param.get('normalise_with_zero_mean'):
		return  (z - z_mean) / z_lims[1]
	else:
		return z / z_lims[1]

def normalise(p, param):
	p_n = Position()
	p_n.x = normalise_x(p.x, param)
	p_n.y = normalise_y(p.y, param)
	p_n.z = normalise_z(p.z, param)
	p_n.speed = p.speed
	return p_n

def unnormalise_x(x, param):
	if param.get('normalise_with_zero_mean'):
		return  (x * x_lims[1]) + x_mean
	else:
		return x * x_lims[1]

def unnormalise_y(y, param):
	if param.get('normalise_with_zero_mean'):
		return  (y * y_lims[1]) + y_mean
	else:
		return y * y_lims[1]

def unnormalise_z(z, param):
	if param.get('normalise_with_zero_mean'):
		return  (z * z_lims[1]) + z_mean
	else:
		return z * z_lims[1]

def unnormalise(p, param):
	p_n = Position()
	p_n.x = unnormalise_x(p.x, param)
	p_n.y = unnormalise_y(p.y, param)
	p_n.z = unnormalise_z(p.z, param)
	p_n.speed = p.speed
	return p_n

def clamp(p):
	p_n=Position()
	p_n.x = clamp_x(p.x)
	p_n.y = clamp_y(p.y)
	p_n.z = clamp_z(p.z)
	p_n.speed = p.speed
	return speed

def clamp_x(x):
	if x <= x_lims[0]:
		return x_lims[0]
	if x > x_lims[1]:
		return x_lims[1]
	return x

def clamp_y(y):
	if y <= y_lims[0]:
		return y_lims[0]
	if y > y_lims[1]:
		return y_lims[1]
	return y

def clamp_z(z):
	if z <= z_lims[0]:
		return z_lims[0]
	if z > z_lims[1]:
		return z_lims[1]
	return z


class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.speed = 0

def distance (pos_a, pos_b):
	return np.sqrt( np.power(pos_a.x - pos_b.x, 2) + np.power(pos_a.y - pos_b.y, 2) + np.power(pos_a.z - pos_b.z,2) )


def clear_tensorflow_graph():
	print('Clearing TF session')
	if tf.__version__ < "1.8.0":
		tf.reset_default_graph()
	else:
		tf.compat.v1.reset_default_graph()

# get NN layer index by name
def getLayerIndexByName(model, layername):
	for idx, layer in enumerate(model.layers):
		if layer.name == layername:
			return idx


