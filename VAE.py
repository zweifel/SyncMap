from keras.utils import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from tensorflow.keras.models import Model

class VAE:
	
	def __init__(self, input_size, latent_dim, timesteps):
		
		self.organized= False
		self.latent_dim= latent_dim
		self.input_size= input_size
		self.timesteps = timesteps
		self.counter = 0
		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []

		self.map = np.zeros((self.input_size,latent_dim))
		self.createModel()
		
		self.createMap()
	
	def createModel(self):
		
		self.createConvModel()
		
	def createConvModel(self):

		input_shape= (self.input_size)
		input_layer = Input(shape=input_shape)
		layer = Dense(self.latent_dim)(input_layer)
		#layer = Droupout(0.4)(layer)
		#layer = Dense(latent_dim)(layer)
		output = Dense(self.input_size,activation='sigmoid')(layer)
		
		self.model = Model(input_layer, output)
		self.encoder = Model(input_layer, layer)


	def createLSTMModel(self):
		
		inputs = Input(shape=(self.timesteps, self.input_size))
		encoded = LSTM(self.latent_dim)(inputs)

		decoded = RepeatVector(self.timesteps)(encoded)
		decoded = LSTM(self.input_size, return_sequences=True)(decoded)

		
		self.model = Model(inputs, decoded)
		self.encoder = Model(inputs, encoded)

	def input(self, x):
		
		#convert to n-gram or skip-gram
		for i,sample in enumerate(x):
			if i - self.timesteps >= 0:
				position = int(self.timesteps/2)
				y = x[i-position]
				a = i-self.timesteps
				b = i-position
				c = i-position+1
				d = i+1
				#rint(a,b,c,d)
				#xit()
				#sample = x[i-self.timesteps:(i-position)] + x[i-position+1:i+1]
				sample = x[np.r_[a:b,c:d]]
				
				#print(sample)
				#sample= [a.argmax() for a in sample]
				sample = np.sum(sample, axis=0)
				#print(sample)
				sample/= sample.sum()
				#print(sample)
				#exit()
				#skip-gram
				#self.dataset.append((y, sample))
				self.x_train.append(y)
				self.y_train.append(sample)
				#print(sample)
				#sample= np.argmax(sample)
				#print(sample)
				#exit()
				#n-gram
				#self.dataset.append((sample, y))
		

		learning_rate= 1e-3
		epochs=10
		batch_size=64
		loss= "mean_squared_error"
		optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
		self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

		np_x_train= np.array(self.x_train)
		np_y_train= np.array(self.y_train)

		#self.x_train = self.x_train[:,None]
		
		#print(self.x_train.shape, self.y_train.shape)

		self.model.fit(
			np_x_train,
			np_y_train,
			epochs=epochs,
			#validation_data=(x_val, val_labels),
			verbose=2,  # Logs once per epoch.
			batch_size=batch_size)


		#print(len(self.dataset))
		#print(self.dataset[0])
		#k= self.dataset[0]
		#print(k[0].shape)

		#exit()	
			
	def createMap(self):
		

		all_possible_inputs = [np_utils.to_categorical(i, self.input_size) for i in range(self.input_size)]
		for i,a in enumerate(all_possible_inputs):
			sample = a[None,:]
			#print("a shape", a.shape)
			predicted= self.encoder.predict(sample)
			#print(predicted)
			self.map[i] = predicted
		

	def organize(self):
	
		self.organized= True
		#self.labels= DBSCAN(eps=3, min_samples=2).fit_predict(self.syncmap)

		self.createMap()

		self.labels= DBSCAN(eps=1, min_samples=2).fit_predict(self.map)

		return self.labels

	def activate(self, x):
		'''
		Return the label of the index with maximum input value
		'''

		if self.organized == False:
			print("Activating a non-organized SyncMap")
			return
		
		#maximum output
		max_index= np.argmax(x)

		return self.labels[max_index]

	def plotSequence(self, input_sequence, input_class,filename="plot.png"):

		input_sequence= input_sequence[1:500]
		input_class= input_class[1:500]

		a= np.asarray(input_class)
		t = [i for i,value in enumerate(a)]
		c= [self.activate(x) for x in input_sequence] 
		

		plt.plot(t, a, '-g')
		plt.plot(t, c, '-.k')
		#plt.ylim([-0.01,1.2])


		plt.savefig(filename,quality=1, dpi=300)
		plt.show()
		plt.close()
	

	def plot(self, color=None, save = False, filename= "plot_map.png"):

		if color is None:
			color= self.labels
		
		#print(self.syncmap)
		#print(self.syncmap[:,0])
		#print(self.syncmap[:,1])
		if self.latent_dim == 2:
			#print(type(color))
			#print(color.shape)
			ax= plt.scatter(self.map[:,0],self.map[:,1], c=color)
			
		if self.latent_dim == 3:
			fig = plt.figure()
			ax = plt.axes(projection='3d')

			ax.scatter3D(self.map[:,0],self.map[:,1], self.map[:,2], c=color);
			#ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
		
		if save == True:
			plt.savefig(filename)
		
		plt.show()
		plt.close()

	def save(self, filename):
		"""save class as self.name.txt"""
		file = open(filename+'.txt','w')
		file.write(cPickle.dumps(self.__dict__))
		file.close()

	def load(self, filename):
		"""try load self.name.txt"""
		file = open(filename+'.txt','r')
		dataPickle = file.read()
		file.close()

		self.__dict__ = cPickle.loads(dataPickle)
		

