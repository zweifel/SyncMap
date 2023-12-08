from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt


class OverlapChunkTest2:
	
	def __init__(self, time_delay):
		self.chunk= 0
		self.output_size = 10
		self.counter = -1
		self.time_delay = time_delay
		self.time_counter = time_delay
		self.output_class= 0
		self.previous_output_class= None

		self.sequenceA_length = 4
		self.sequenceB_length = 4 #np.random.randint(2)+5
	
	def getOutputSize(self):
		return self.output_size
	
	def trueLabel(self):
		truelabel= np.array((0,0,0,0,0,1,1,1,1,1))
		return truelabel

	def updateTimeDelay(self):
		self.time_counter+= 1
		if self.time_counter > self.time_delay:
			self.time_counter = 0 
			return True
		else:
			return False

	#create an input pattern for the system
	def getInput(self, reset = False):
		
		if reset == True:
			self.chunk=0
			self.counter=-1

		update = self.updateTimeDelay()

		if update == True:
			if self.chunk == 0:
				if self.counter > self.sequenceA_length:
					self.chunk = 1
					self.counter= 0
				else:
					self.counter+= 1
			else:
				if self.counter > self.sequenceB_length:
					#self.sequenceB_length = np.random.randint(20)+5
					self.chunk = 0
					self.counter= 0
				else:
					self.counter+= 1

			if self.chunk == 0:
				#input_value = np.random.randint(10)
				#input_value= self.counter
				self.previous_output_class= self.output_class
				
				#possible outputs are 0,1,2,3,4
				self.output_class = np.random.randint(5)  
			else:
				self.previous_output_class= self.output_class
				#possible outputs are 0,1,2,3,4,5,6,7,8,9
				self.output_class = np.random.randint(10)

		noise_intensity= 0
#		input_value = to_categorical(self.output_class, self.output_size) + np.random.randn(self.output_size)*noise_intensity
		if self.previous_output_class is None or self.previous_output_class == self.output_class:
			input_value = to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
		else:
			input_value = to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + to_categorical(self.previous_output_class, self.output_size)*np.exp(-0.1*(self.time_counter+self.time_delay))

		return input_value
	
	def getOutputSize(self):
		return self.output_size

	def getSequence(self, iterations):
	
		input_class = np.empty(iterations)
		input_sequence = np.empty((iterations, self.output_size))

		for i in range(iterations):
			input_value = self.getInput()
			#input_class.append(self.chunk)
			#input_sequence.append(input_value)
			input_class[i] = self.chunk
			input_sequence[i] = input_value
		


		return input_sequence, input_class

	def plot(self, input_class, input_sequence = None, save = False):
		
		a = np.asarray(input_class)
		t = [i for i,value in enumerate(a)]

		plt.plot(t, a)
		
		if input_sequence != None:
			sequence = [np.argmax(x) for x in input_sequence]
			plt.plot(t, sequence)

		if save == True:
			plt.savefig("plot.png")
		
		plt.show()
		plt.close()
	
	def plotSuperposed(self, input_class, input_sequence = None, save = False):
	
		input_sequence= np.asarray(input_sequence)
		
		t = [i for i,value in enumerate(input_sequence)]

		print(input_sequence.shape)

		#exit()

		for i in range(input_sequence.shape[1]):
			a = input_sequence[:,i]
			plt.plot(t, a)
		
		a = np.asarray(input_class)
		plt.plot(t, a)

		if save == True:
			plt.savefig("plot.png")
		
		plt.show()
		plt.close()

