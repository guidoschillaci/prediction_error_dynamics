# author Guido Schillaci, Dr.rer.nat. - Humboldt-Universitaet zu Berlin
# Guido Schillaci <guido.schillaci@informatik.hu-berlin.de>

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# taken from https://github.com/despoisj/LatentSpaceVisualization/blob/master/visuals.py
# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom, imageSize):
	images = []
	for i in range(len(x)):
		x0, y0 = x[i], y[i]
		# Convert to image
		img = imageData[i]*255.
		img = img.astype(np.uint8).reshape([imageSize,imageSize])
		img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
		# Note: OpenCV uses BGR and plt uses RGB
		image = OffsetImage(img, zoom=zoom)
		ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
		images.append(ax.add_artist(ab))
	
	ax.update_datalim(np.column_stack([x, y]))
	ax.autoscale()

def plot_exploration(positions, goals, iteration, param, save = True, show = False):
	print ('position shape ', np.asarray(positions).shape)
	title = param.get('goal_selection_mode')+'_'+str(iteration)
	fig2 = plt.figure(figsize=(10, 10))
	#print (log_goal)
	if param.get('romi_input_dim') ==2:
		plt.scatter(np.transpose(positions)[0],np.transpose(positions)[1], s=2, color='g')
		plt.plot( np.asarray(goals[:,0]).astype('float32'), np.asarray(goals[:,1]).astype('float32'), 'ro')

		plt.xlabel('Pos x')
		plt.ylabel('Pos y')
		plt.xlim(-1.2,1.2)
		plt.ylim(-1.2,1.2)
	elif param.get('romi_input_dim')==4:
		plt.subplot(1,2,1)
		plt.scatter(positions[:,0],positions[:,1], s=2, color='g')
		plt.plot( np.asarray(goals[:,0]).astype('float32'), np.asarray(goals[:,1]).astype('float32'), 'ro')
		plt.xlim(-0.4,0.4)
		plt.ylim(-0.4,0.4)
		plt.xlabel('Dim 0')
		plt.ylabel('Dim 1')
		plt.subplot(1,2,2)
		plt.scatter(positions[:,0],positions[:,1], s=2, color='g')
		plt.plot(np.asarray( goals[:,2]).astype('float32'), np.asarray(goals[:,3]).astype('float32'), 'ro')
		plt.xlim(-0.4,0.4)
		plt.ylim(-0.4,0.4)
		plt.xlabel('Dim 2')
		plt.ylabel('Dim 3')

	if save:
		filename =param.get('results_directory')+ 'plots/plot_'+title+'_'+str(iteration)+'.jpg'
		plt.savefig(filename)
	if show:
		plt.show()
	plt.close()

def plot_som(decoder, som, data):
	print ("Plotting decoded SOM nodes")
	goal_codes = None
	goal_codes =  som._weights.reshape(len(som._weights)*len(som._weights[0]), len(som._weights[0][0]) )
#	print som._weights.shape, " ", goal_codes.shape
	decoded = decoder.predict(goal_codes[:])
	print (decoded.shape)
	s = decoded.reshape(len(som._weights), len(som._weights[0]), len(decoded[0]), len(decoded[0]), 3)
	print (s.shape)

	fig1 = plt.figure(figsize=(len(som._weights), len(som._weights)))
	#fig, axes = plt.subplots( len(som._weights), len(som._weights))
	for i in range(len(som._weights)):
		for j in range(len(som._weights)):
			#axes[i,j] = plt.imshow(s[i][j])
			ax1 = plt.subplot(len(som._weights), len (som._weights), j + i * len(som._weights) +1 )
			plt.imshow(s[i][j])
	plt.show()
	plt.close()

def plot_som_scatter(encoder, som, data):
	encoded_imgs = encoder.predict(data)
	fig2 = plt.figure(figsize=(10, 10))
	print (som._weights)
	goal_codes =  som._weights.reshape(len(som._weights)*len(som._weights[0]), len(som._weights[0][0]) )
	print (goal_codes)
	plt.subplot(1, 2, 1)
	plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1], s=2, color='g') 
	plt.scatter(goal_codes[:,0],goal_codes[:,1], s=6, color='r') 
	
	plt.subplot(1, 2, 2)
	plt.scatter(encoded_imgs[:,2],encoded_imgs[:,3], s=2, color='g') 
	plt.scatter(goal_codes[:,2],goal_codes[:,3], s=6, color='r') 

	plt.show()
	plt.close()

def plots_fwd_model(forward_model, commands, original_images, image_size=128, channels =1):
	predicted_images = forward_model.predict(commands[:])
	n = 10
	fig1 = plt.figure(figsize=(n*2, 4))
	for i in range(n):
		# display original
		ax1 = plt.subplot(2, n, i+1)
		if channels ==1:
			plt.gray()
			plt.imshow(original_images[i].reshape(image_size, image_size))
		else:
			plt.imshow(original_images[i].reshape(image_size, image_size,3))
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)

		# display reconstruction
		ax2 = plt.subplot(2, n, i + n +1)
		if channels ==1:
			#print predicted_images[i]
			#print 'command ', commands[i]
			plt.gray()
			plt.imshow(predicted_images[i].reshape(image_size, image_size))
		else:
			plt.imshow(predicted_images[i].reshape(image_size, image_size,3))
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)
	plt.show()
	#print predicted_images[i]
	plt.close()

def plots_full_model(full_model, full_inv_model, full_fwd_model, commands, original_images, image_size=128, channels =1):
	predicted_images = full_fwd_model.predict(commands[:])
	n = 10
	fig1 = plt.figure(figsize=(n*2, 4))
	for i in range(n):
		# display original
		ax1 = plt.subplot(2, n, i+1)
		if channels ==1:
			plt.gray()
			plt.imshow(original_images[i].reshape(image_size, image_size))
		else:
			plt.imshow(original_images[i].reshape(image_size, image_size,3))
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)

		# display reconstruction
		ax2 = plt.subplot(2, n, i + n +1)
		if channels ==1:
			#print predicted_images[i]
			#print 'command ', commands[i]
			plt.gray()
			plt.imshow(predicted_images[i].reshape(image_size, image_size))
		else:
			plt.imshow(predicted_images[i].reshape(image_size, image_size,3))
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)
	plt.show()
	#print predicted_images[i]

	predicted_cmd = full_inv_model.predict(original_images[:])

	print (np.hstack(( commands, predicted_cmd)))
	print ('total error pred-cmd ', np.linalg.norm(predicted_cmd-commands))
	print ('average error pred-cmd', np.mean( np.linalg.norm(predicted_cmd-commands, axis=1)))

	pre_predicted_cmd = full_inv_model.predict(predicted_images[:])

	print (np.hstack(( predicted_cmd, pre_predicted_cmd)) )
	print ('total err pred_img- code ', np.linalg.norm(pre_predicted_cmd-predicted_cmd) )
	print ('average error pred_img- code ', np.mean( np.linalg.norm(pre_predicted_cmd-predicted_cmd, axis=1)) )



def plots_cae_decoded(decoder, test_images_codes, original_images, image_size=64, show = False, save=True, directory='./'):
	# plotting decoded predictions
	decoded = decoder.predict(test_images_codes[:])
	n = 5
	fig1 = plt.figure(figsize=(n*2, 4))
	for i in range(n):
		# display original
		ax1 = plt.subplot(2, n, i+1)
		plt.imshow(original_images[i].reshape(image_size, image_size))
		plt.gray()
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)

		# display reconstruction
		ax2 = plt.subplot(2, n, i + n +1)
		plt.imshow(decoded[i].reshape(image_size, image_size))
		plt.gray()
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)
	if show:
		plt.show()
	if save:
		plt.savefig(directory + 'cae_decoded.png')

def plots_cae(autoencoder, encoder, decoder, train_images, test_images, image_size,channels=1):

	print ('Plotting CAE test images and decoded')
	n_figures = 5
	'''
	decoded_imgs = autoencoder.predict(test_images)
	# plotting autoencoder predictions
#	decoded_imgs = autoencoder.predict(train_images)
	n = n_figures
	fig1 = plt.figure(figsize=(n*2, 4))
	for i in range(n):
		# display original
		ax1 = plt.subplot(2, n, i+1)
		plt.imshow(test_images[i].reshape(image_size, image_size,3))
		plt.gray()
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)

		# display reconstruction
		ax2 = plt.subplot(2, n, i + n +1)
		plt.imshow(decoded_imgs[i].reshape(image_size, image_size,3))
		plt.gray()
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)
	'''

	# plotting encoder predictions
#	encoded_imgs = encoder.predict(test_images)
	encoded_imgs = encoder.predict(train_images)
#	fig2 = plt.figure(figsize=(10, 20))

#	plt.subplot(1, 2, 1)
#	plt.scatter(encoded_imgs[:,0],encoded_imgs[:,1], s=2) 
#	print encoded_imgs[0][:].shape, ' ',encoded_imgs[:].shape
#	print 'max ',np.max(encoded_imgs[:][0]), ' ',np.max(encoded_imgs[:][1]), ' ',np.max(encoded_imgs[:][2]), ' ',np.max(encoded_imgs[:][3]), ' '
#	print 'min ',np.min(encoded_imgs[:][0]), ' ',np.min(encoded_imgs[:][1]), ' ',np.min(encoded_imgs[:][2]), ' ',np.min(encoded_imgs[:][3]), ' '

#	plt.subplot(1, 2, 2)
#	plt.scatter(encoded_imgs[:,2],encoded_imgs[:,3], s=2) 

	# plotting decoded predictions
	decoded_2 = decoder.predict(encoded_imgs[:])
	#print decoded_2.shape  
	#plt.show()

	n = n_figures
	fig1 = plt.figure(figsize=(n*2, 4))
	for i in range(n):
		# display original
		ax1 = plt.subplot(2, n, i+1)
		if channels ==1:
			plt.gray()
			plt.imshow(test_images[i].reshape(image_size, image_size))
		else:
			plt.imshow(test_images[i].reshape(image_size, image_size, channels))
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)

		# display reconstruction
		ax2 = plt.subplot(2, n, i + n +1)
		if channels ==1:
			plt.gray()
			plt.imshow(decoded_2[i].reshape(image_size, image_size))
		else:
			plt.imshow(decoded_2[i].reshape(image_size, image_size, channels))
#		plt.gray()
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)

	plt.show()

def plot_learning_progress( log_lp, log_goal_id, num_goals = 9, save = True, show= False):
	fig = plt.figure(figsize=(10, 10))

	ax1=plt.subplot( num_goals+1, 1, 1)
	plt.plot(log_goal_id)
	plt.ylim(1,num_goals)
	plt.ylabel('goal id')
	plt.xlabel('time')
	ax1.set_yticks(np.arange(0,num_goals))
	ax1.yaxis.grid(which="major", linestyle='-', linewidth=2)

	data = np.transpose(log_lp)
	for i in range(0, num_goals):
		ax=plt.subplot(num_goals+1, 1, i+2)
		plt.plot(data[i])
		plt.ylabel('g{}'.format(i))
		#plt.ylim(0,2)		
		#idx = np.argwhere(log_goal_id==i)
		#print findall(log_goal_id, lambda x:x==i)#
		#print np.where(log_goal_id==i, 1,0)
		#plt.fill_between(np.arange(0, len(data[i])), 0,1, where=log_goal_id==i, transform=ax.get_xaxis_transform(), alpha=0.3)
	
	if save:
		plt.savefig('./models/plots/plot_learning_progress.jpg')
	if show:
		plt.show()  	
	plt.close()

def plot_log_goal_inv( log_goal, log_pos, num_goals = 9, save = True,show= False):
	fig = plt.figure(figsize=(15, 15))

	for i in range(0, num_goals):
		ax=plt.subplot(num_goals, 1, i+1)
		g = log_goal[i]
		p = log_pos[i]
		if len(g)>0 and len(p)>0:
			#print i, 'goal x', np.transpose(g)[0]
			#print i,'goal y', np.transpose(g)[1]
			#print i,'pos x', np.transpose(p)[0]
			#print i,'pos y', np.transpose(p)[1]
			plt.plot(np.transpose(g)[0], 'r-', label='goal_x')
			plt.plot(np.transpose(g)[1], 'b-', label='goal_y')
			plt.plot(np.transpose(p)[0], 'r--', label='pos_x')
			plt.plot(np.transpose(p)[1], 'b--', label='pos_y')
		plt.ylabel('g{}'.format(i))
		plt.ylim(-0.1,1.2)		
		#idx = np.argwhere(log_goal_id==i)
		#print findall(log_goal_id, lambda x:x==i)#
		#print np.where(log_goal_id==i, 1,0)
		#plt.fill_between(np.arange(0, len(data[i])), 0,1, where=log_goal_id==i, transform=ax.get_xaxis_transform(), alpha=0.3)
	
	if save:
		plt.savefig('./models/plots/plot_log_goal_inv.jpg')
	if show:
		plt.show()
	plt.close()

def plot_log_goal_fwd( log_goal, log_curr, num_goals = 9, save = True,show= True):
	fig = plt.figure(figsize=(15, 15))

	for i in range(0, num_goals):
		ax=plt.subplot(num_goals, 1, i+1)
		g = log_goal[i]
		p = log_curr[i]
		#print 'g ', g
		#print 'p ', p
		if len(g)>0 and len(p)>0:
			#print i, 'goal x', np.transpose(g)[0]
			#print i,'goal y', np.transpose(g)[1]
			#print i,'pos x', np.transpose(p)[0]
			#print i,'pos y', np.transpose(p)[1]
			#print g[0]
			#print np.asarray(g).shape
			plt.plot( np.transpose(g)[0], 'r-', label='goal code[0]')
			plt.plot( np.transpose(g)[1], 'b-', label='goal code[1]')
#			plt.plot( np.transpose(g)[0][2], 'g-', label='goal code[2]')
#			plt.plot( np.transpose(g)[0][3], 'y-', label='goal code[3]')
			plt.plot( np.transpose(p)[0], 'r--', label='curr code[0]')
			plt.plot( np.transpose(p)[1], 'b--', label='curr code[1]')
#			plt.plot( np.transpose(p)[0][2], 'g--', label='curr code[2]')
#			plt.plot( np.transpose(p)[0][3], 'y--', label='curr code[3]')
		plt.ylabel('g{}'.format(i))
		#plt.ylim(-0.1,1.2)		
		#idx = np.argwhere(log_goal_id==i)
		#print findall(log_goal_id, lambda x:x==i)#
		#print np.where(log_goal_id==i, 1,0)
		#plt.fill_between(np.arange(0, len(data[i])), 0,1, where=log_goal_id==i, transform=ax.get_xaxis_transform(), alpha=0.3)
	
	if save:
		plt.savefig('./models/plots/plot_log_goal_fwd.jpg')
	if show:
		plt.show()
	plt.close()

'''
def plot_learning_comparisons(model_type = 'fwd', exp_size = 2, save = True, show = True):

	data_db = []
	data_kmeans = []
	data_som = []
	data_random = []
	for i in range(exp_size):
		filename_db = './experiments/db_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_db.append(np.asarray(np.loadtxt(filename_db)))

		filename_kmeans = './experiments/kmeans_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_kmeans.append(np.asarray(np.loadtxt(filename_kmeans)))

		filename_som = './experiments/som_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_som.append(np.asarray(np.loadtxt(filename_som)))

		filename_random = './experiments/random_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_random.append(np.asarray(np.loadtxt(filename_random)))

	mean_db = np.mean(data_db, axis=0)
	stddev_db = np.std(data_db, axis=0)

	mean_kmeans = np.mean(data_kmeans, axis=0)
	stddev_kmeans = np.std(data_kmeans, axis=0)

	mean_som = np.mean(data_som, axis=0)
	stddev_som = np.std(data_som, axis=0)
	
	mean_random = np.mean(data_random, axis=0)
	stddev_random = np.std(data_random, axis=0)
	

	fig2 = plt.figure(figsize=(10, 10))
	plot1, = plt.plot(mean_db, color='#CC4F1B', label='db')
	plt.fill_between(np.arange(len(mean_db)), mean_db-stddev_db, mean_db+stddev_db, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	plot2, = plt.plot(mean_kmeans, color='#1B2ACC', label='kmeans')
	plt.fill_between(np.arange(len(mean_kmeans)), mean_kmeans-stddev_kmeans, mean_kmeans+stddev_kmeans, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

	plot3, = plt.plot(mean_som, color='#3F7F4C', label='som')
	plt.fill_between(np.arange(len(mean_som)), mean_som-stddev_som, mean_som+stddev_som, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')

	plot4, = plt.plot(mean_som, color='#FF7F4F', label='random')
	plt.fill_between(np.arange(len(mean_random)), mean_random-stddev_random, mean_random+stddev_random, alpha=0.5, edgecolor='#FF7F4F', facecolor='#FEFF9F')

	plt.legend(handles=[plot1, plot2, plot3, plot4], loc=1)
	plt.title(model_type+' model MSE')
#	plt.show()

	if save:
		filename ='./experiments/learning_comparison_'+model_type+'.jpg'
		plt.savefig(filename)
	if show:
		plt.show()
	plt.close()

def plot_learning_comparisons2(model_type = 'fwd', exp_size = 2, save = True, show = True):

	data_db1 = []
	data_db2 = []
	data_som = []
	data_random = []
	for i in range(exp_size):
		filename_db1 = './experiments_9_goals_0.1_prob/db_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_db1.append(np.asarray(np.loadtxt(filename_db1)))

		filename_db2 = './experiments_9_goals_0.01_prob/db_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_db2.append(np.asarray(np.loadtxt(filename_db2)))

	mean_db1 = np.mean(data_db1, axis=0)
	stddev_db1 = np.std(data_db1, axis=0)

	mean_db2 = np.mean(data_db2, axis=0)
	stddev_db2 = np.std(data_db2, axis=0)

	fig2 = plt.figure(figsize=(8, 5))
	plot1, = plt.plot(mean_db1, color='#CC4F1B', label='Prob = 0.1')
	plt.fill_between(np.arange(len(mean_db1)), mean_db1-stddev_db1, mean_db1+stddev_db1, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	plot2, = plt.plot(mean_db2, color='#1B2ACC', label='Prob = 0.01')
	plt.fill_between(np.arange(len(mean_db2)), mean_db2-stddev_db2, mean_db2+stddev_db2, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

	plt.legend(handles=[plot1, plot2], loc=1)
	plt.title(model_type+' model MSE')
#	plt.show()

	if save:
		filename ='./experiments/learning_comparison_prob_'+model_type+'.jpg'
		plt.savefig(filename)
	if show:
		plt.show()
	plt.close()


def plot_learning_comparisons3(model_type = 'fwd', exp_size = 2, save = True, show = True):

	data_db1 = []
	data_db2 = []
	data_som = []
	data_random = []
	for i in range(exp_size):
		filename_db1 = './db_9_goals_0.1_prob/db_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_db1.append(np.asarray(np.loadtxt(filename_db1)))

		filename_db2 = './experiments_9_goals_0.01_prob/db_'+str(i)+'/models/log_mse_'+model_type+'.txt'
		data_db2.append(np.asarray(np.loadtxt(filename_db2)))

	mean_db1 = np.mean(data_db1, axis=0)
	stddev_db1 = np.std(data_db1, axis=0)

	mean_db2 = np.mean(data_db2, axis=0)
	stddev_db2 = np.std(data_db2, axis=0)

	fig2 = plt.figure(figsize=(8, 5))
	plot1, = plt.plot(mean_db1, color='#CC4F1B', label='Prob = 0.1')
	plt.fill_between(np.arange(len(mean_db1)), mean_db1-stddev_db1, mean_db1+stddev_db1, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	plot2, = plt.plot(mean_db2, color='#1B2ACC', label='Prob = 0.01')
	plt.fill_between(np.arange(len(mean_db2)), mean_db2-stddev_db2, mean_db2+stddev_db2, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

	plt.legend(handles=[plot1, plot2], loc=1)
	plt.title(model_type+' model MSE')
#	plt.show()

	if save:
		filename ='./experiments/learning_comparison_prob_'+model_type+'.jpg'
		plt.savefig(filename)
	if show:
		plt.show()
	plt.close()


def do_plots(directory = './experiments/', exp_type =['db', 'som'], history_size = [0, 10, 20], prob = [0.1, 0.01], exp_size = 5, model_type ='fwd', save = True, show = True):



	# for each experiment
	for exp_t in range(len(exp_type)):


		# PLOT OVER PROBABILITY
		for p in range(len(prob)):
			colors = ['r', 'g', 'b']
			data = []
			#mean = []
			#stddev=[]
			exp_string = directory + str(exp_type[exp_t])

			fig = plt.figure(figsize=(8, 5))
			# varying history size...
			for h in range(len(history_size)):
				plt.title('MSE '+model_type + ' model. Prob: '+ str(prob[p]))
				data.append([])
				#mean.append([])
				#stddev.append([])
				for it in range(exp_size):
					filename =  exp_string + '_' + str(history_size[h]) + '_' + str(prob[p]) + '_' + str(it) + '/models/log_mse_' + str(model_type) + '.txt'
					data[-1].append(np.asarray(np.loadtxt(filename)))
				# mean[-1].append( np.mean(data[-1], axis = 0) )
				# stddev[-1].append(np.std(data[-1], axis = 0))
				mean = np.mean(data[-1], axis = 0)
				stddev = np.std(data[-1], axis = 0)

				print(filename)
				print ('mean ' , np.asarray(mean).shape)

				print ('std ' , np.asarray(stddev).shape)

				plt.plot(mean, color= colors[h], label='Memory size: '+str(history_size[h]) + ' * batch_size'  )
				if (model_type=='fwd'):
					plt.ylim(5,30)
				else:
					plt.ylim(0,1.5)
				plt.fill_between(np.arange(len(mean)), mean-stddev, mean+stddev, alpha=0.5, color=colors[h])

			plt.legend(loc='upper right', prop={'size': 6})
			if save:
				fig_file =exp_string + '_prob_' + str(prob[p]) + '_' + str(model_type) +'.jpg'
				plt.savefig(fig_file)
			if show:
				plt.show()
			plt.close()

		# PLOT OVER HISTORY SIZE
		for h in range(len(history_size)):
			colors = ['r', 'g', 'b']
			data = []
			#mean = []
			#stddev=[]
			exp_string = directory + str(exp_type[exp_t])

			fig = plt.figure(figsize=(8, 5))
			# varying history size...
			for p in range(len(prob)):
				plt.title('MSE '+model_type + ' model. Memory size: '+ str(history_size[h])+ '*batch_size')
				data.append([])
				#mean.append([])
				#stddev.append([])
				for it in range(exp_size):
					filename =  exp_string + '_' + str(history_size[h]) + '_' + str(prob[p]) + '_' + str(it) + '/models/log_mse_' + str(model_type) + '.txt'
					data[-1].append(np.asarray(np.loadtxt(filename)))
				# mean[-1].append( np.mean(data[-1], axis = 0) )
				# stddev[-1].append(np.std(data[-1], axis = 0))
				mean = np.mean(data[-1], axis = 0)
				stddev = np.std(data[-1], axis = 0)

				print(filename)
				print ('mean ' , np.asarray(mean).shape)

				print ('std ' , np.asarray(stddev).shape)

				plt.plot(mean, color= colors[p], label='Prob: '+str(prob[p]) )
				if (model_type=='fwd'):
					plt.ylim(5,30)
				else:
					plt.ylim(0,1.5)
				plt.fill_between(np.arange(len(mean)), mean-stddev, mean+stddev, alpha=0.5, color=colors[p])

			plt.legend(loc='upper right', prop={'size': 6})
			if save:
				fig_file =exp_string + '_historysize_' + str(history_size[h]) + '_' + str(model_type) +'.jpg'
				plt.savefig(fig_file)
			if show:
				plt.show()
			plt.close()

'''
if __name__ == '__main__':
	pass
	#do_plots(model_type='fwd')
	#do_plots(model_type='inv')
	#plot_learning_comparisons2(model_type='inv')
	#plot_learning_comparisons2(model_type='fwd')
	#log_goal_pos = np.loadtxt('./models/log_goal_pos.txt')
	#log_curr_pos = np.loadtxt('./models/log_curr_pos.txt')
	#log_lp = np.loadtxt('./models/log_learning_progress.txt')	
	#log_goal_id = np.loadtxt('./models/log_goal_id.txt')
	#plot_learning_progress(log_lp, log_goal_id)
