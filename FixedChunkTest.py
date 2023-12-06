from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt


class FixedChunkTest:
	
	def __init__(self, time_delay, filename="fixed_chunk2.txt"):
		'''
		Chunks are written in the filename in which every line is a sequence of outputs followed by the number of the respective chunk
		All chunk numbers must be in ascending order and must have the same number of outputs
		Chunks will be shuffled and presented repeatedly throughout
		'''
		dataset= np.loadtxt(filename, dtype="i", delimiter=",")
		self.time_delay = time_delay
		self.time_counter = 0
		self.current_index= 0

		self.output_size= dataset.shape[1]-1
		self.data = dataset[:,:self.output_size]
		self.data_class= dataset[:,self.output_size]

		acc = np.zeros(len(self.data_class),  dtype=int)
		for i,sample in enumerate(self.data):
			#print(sample)
			#print(self.data_class)
			tmp= sample*self.data_class
			acc[i]= int(tmp.sum())
		
		acc-= 1
		self.true_labels= acc

		self.chunk= []
		new_chunk= None
		new_chunk_index= None
		for i,sample in enumerate(self.data):
			if new_chunk is None:
				new_chunk_index= self.data_class[i]
				new_chunk= [sample]
			else:
				if new_chunk_index == self.data_class[i]:
					new_chunk.append(sample)
				else:
					self.chunk.append(np.asarray(new_chunk))
					new_chunk= [sample]
					new_chunk_index= self.data_class[i]

		self.chunk.append(np.asarray(new_chunk))

		self.chunk= np.asarray(self.chunk)
		self.number_of_chunks= self.chunk.shape[0]
		self.chunk_index= np.random.randint(self.number_of_chunks)
		
		#print(self.chunk)
#		print(self.chunk.shape)
#		for i in range(10):
#			rand= np.random.randint(self.number_of_chunks)
#			print(self.chunk[rand])

#		exit()



#		self.chunk= 0
#		self.output_size = output_size
#		self.counter = -1
#		self.output_class= data_class[current_index]
		self.previous_output_class= None
		self.previous_previous_output_class= None
			
		#print(self.data_class.shape[0])
		#exit()

#		self.sequenceA_length = 4
#		self.sequenceB_length = 4 #np.random.randint(2)+5
	
	def getOutputSize(self):
		return self.output_size
	
	def trueLabel(self):
		return self.true_labels

	def updateTimeDelay(self):
		self.time_counter+= 1
		if self.time_counter > self.time_delay:
			self.time_counter = 0 
			self.previous_previous_output_class= self.previous_output_class
			self.previous_output_class= self.output_class
			return True
		else:
			return False

	#create an input pattern for the system
	def getInput(self, reset = False):
		
		if reset == True:
			self.current_index=0
			self.time_counter=0

		update = self.updateTimeDelay()
		
		#print(self.chunk[self.chunk_index].shape)
		#exit()

		if update == True:
			
			self.current_index+= 1

			#check if a new chunk should start
			if self.current_index >= self.chunk[self.chunk_index].shape[0]:
				self.chunk_index= np.random.randint(self.number_of_chunks)
				self.current_index= 0
			
		
					
		#chunk is the cluster it pertains
		#output class is the current output
		#self.chunk_index= 
		#print("chunk",self.chunk)
		self.output_class = self.chunk[self.chunk_index][self.current_index]
		
		noise_intensity= 0
		if self.previous_output_class is None or np.array_equal(self.previous_output_class, self.output_class):
			input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
		else:
			input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + self.previous_output_class*np.exp(-0.1*(self.time_counter+self.time_delay))



		return input_value

	def getSequence(self, sequence_size):
	
		#print(self.data.shape[0])
		#print(input_sequence.shape)
		#exit()
		self.input_sequence = np.empty((sequence_size, self.data.shape[1]))
		self.input_class = np.empty(sequence_size)
		
		for i in range(sequence_size):
			
			input_value = self.getInput()
			
			#input_class.append(self.chunk)
			#input_sequence.append(input_value)
			self.input_class[i] = self.chunk_index
			self.input_sequence[i] = input_value

		return self.input_sequence, self.input_class

	
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

