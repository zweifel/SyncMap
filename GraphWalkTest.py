from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz

class GraphWalkTest:
	
	def __init__(self, time_delay, filename="graph.dot"):
		'''
		'''
		self.time_delay = time_delay
		self.time_counter = 0

		self.G= nx.DiGraph(nx.nx_agraph.read_dot(filename))
		#print(nx.get_node_attributes(self.G,'1'))
		#print(self.G.nodes)
		label= self.G.nodes(data="label")
		label = np.asarray(list(label))
		#print(label)
		#print(label[:,1])
		self.true_label = label[:,1]

		self.true_label= [int(x) for x in self.true_label]
		#print(self.true_label)
		self.true_label= np.asarray(self.true_label) -1
		#print(list(self.true_label))

		self.output_size= self.G.number_of_nodes()
			
		self.A = nx.adjacency_matrix(self.G)
		self.A = self.A.todense()
		self.A = np.array(self.A, dtype = np.float64)
	
		for i in range(self.output_size):
			accum = self.A[i].sum()
			if accum != 0:
				self.A[i]= self.A[i]/accum
			else:
				print("ERROR: Node ",i," without connections from found")
				exit()
			print(self.A[i])
		

		#random start	
		self.output_class= np.random.randint(self.output_size)
		self.previous_output_class= None
		self.previous_previous_output_class= None


		#self.plotGraph()
		#self.getInput()

	def trueLabel(self):
		return self.true_label
	
	def getOutputSize(self):
		return self.output_size

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
	
		update = self.updateTimeDelay()
		
		if update == True:
		
			#print(self.G)
			#print(A.shape)
			#transition= self.A[self.output_class]
			#print(len(transition))
			#print(transition.shape)
			#print(transition)
			
			self.previous_output_class= self.output_class
			self.output_class= np.random.choice(self.output_size ,1 ,p= self.A[self.output_class])[0]
		#print(choice)
		#print(A[0])
		#print(A[1])
		#print(A[2])
		
		noise_intensity= 0
		if self.previous_output_class is None or self.previous_output_class == self.output_class:
			input_value = to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
		else:
			input_value = to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + to_categorical(self.previous_output_class, self.output_size)*np.exp(-0.1*(self.time_counter+self.time_delay))
		
#		noise_intensity=0

#		if  self.previous_output_class is None or np.array_equal(self.previous_output_class, self.output_class):
#			input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
#		else:
#			if  self.previous_previous_output_class is None or np.array_equal(self.previous_previous_output_class, self.previous_output_class):
#				input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + self.previous_output_class*np.exp(-0.1*(self.time_counter+self.time_delay))
#			else:
#				input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + self.previous_output_class*np.exp(-0.1*(self.time_counter+self.time_delay)) + self.previous_previous_output_class*np.exp(-0.1*(self.time_counter+2.0*self.time_delay))
#	
		return input_value

		#for i in range():
			
		#print(A)
		#D = np.diag(np.sum(A, axis=0))
		#print(D)
		#T = np.dot(np.linalg.inv(D),A)
		#print(T)
		# let's evaluate the degree matrix D
		# ...and the transition matrix T
		#exit()
	
		
		#exit()
	
	def getSequence(self, sequence_size):
	
		#print(self.data.shape[0])
		#print(input_sequence.shape)
		#exit()
		self.input_sequence = np.empty((sequence_size, self.output_size))
		self.input_class = np.empty(sequence_size)
		
		for i in range(sequence_size):
			
			input_value = self.getInput()
			
			#input_class.append(self.chunk)
			#input_sequence.append(input_value)
			self.input_class[i] = self.output_class
			self.input_sequence[i] = input_value

		return self.input_sequence, self.input_class

	def plotGraph(self, save = True):

		options = {
			'node_size': 100,
			'arrowstyle': '-|>',
			'arrowsize': 12,
		}
		nx.draw_networkx(self.G, arrows=True,**options)
		
		if save == True:
			plt.savefig("graph_plot.png")
		
		plt.show()

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

