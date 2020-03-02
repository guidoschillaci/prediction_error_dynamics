#! /usr/bin/env python

#from enum import Enum
import numpy as np
import yarp
import data_keys



# this does not work yet. check the returned values (only first elements look ok)
def get_joint_pos_limits_dumb():
	lim = [] # only head joitns so far
	lim.append([-30.0,22.0])
	lim.append([-20.0,20.0])
	lim.append([-45.0,45.0])
	lim.append([-0.0,0.0])
	lim.append([-0.0,0.0]) 
	return lim

# this does not work yet. check the returned values (only first elements look ok)
def get_joint_pos_limits():
	# Initialise YARP
	yarp.Network.init()
	joint_limits = []
	for j in range(len(data_keys.JointNames)):	
		props= yarp.Property()
		props.put("device", "remote_controlboard")
		props.put("local", "/client_lim/"+data_keys.JointNames[j])
		props.put("remote", "/icubSim/"+data_keys.JointNames[j])

		joint_driver=yarp.PolyDriver(props)
		ctrl_lim=joint_driver.viewIControlLimits()

		limits_low = yarp.Vector(joint_driver.viewIPositionControl().getAxes())
		limits_up = yarp.Vector(joint_driver.viewIPositionControl().getAxes())
		ctrl_lim.getLimits(j, limits_low.data(), limits_up.data())
		
		joint_limits.append( [np.fromstring(limits_low.toString(-1,1), dtype=float, sep=' ') , np.fromstring(limits_up.toString(-1,1), dtype=float, sep=' ') ] )
	return joint_limits


