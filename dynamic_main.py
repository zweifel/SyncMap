from keras.utils import to_categorical
import numpy as np
import math
import matplotlib.pyplot as plt

#problems
from ChunkTest import *
from OverlapChunkTest1 import *
from OverlapChunkTest2 import *
from LongChunkTest import *
from FixedChunkTest import *
from GraphWalkTest import *
import sys

#neurons
from SyncMap import *
#from MRILNeuron import *
from VAE import *


########################## Parameters ##########################################
plot= False
#uncomment the following for setting up the problem type
#problem_type=1

################################################################################

time_delay = 10
list_env = []

save_dir= "output_files/"

arg_size = len(sys.argv)
if arg_size > 1:
	save_filename = save_dir + sys.argv[2]
	save_truth_filename = save_dir + sys.argv[2] + "_truth"
	problem_type= sys.argv[1]
	problem_type = int(problem_type)
else:
	save_filename= None
	problem_type = None


if problem_type == 1:
	env1= GraphWalkTest(time_delay)
	list_env.append(env1)
	env2= GraphWalkTest(time_delay,"graph2.dot")
	list_env.append(env2)

if problem_type == 2:
	env1 = FixedChunkTest(time_delay)
	list_env.append(env1)
	env2 = FixedChunkTest(time_delay,"fixed_chunk3.txt")
	list_env.append(env2)

output_size= list_env[0].getOutputSize()

#sanity check
for e in list_env:
	if e.getOutputSize() != output_size:
		print("Error! Provided env have different output size")
		exit()

print("Output Size",output_size)


####### SyncMap #####
sequence_length = 100000
number_of_nodes= output_size
adaptation_rate= 0.01
#adaptation_rate= 0.01*output_size
map_dimensions= 3
neuron_group= SyncMap(number_of_nodes, map_dimensions, adaptation_rate)
####### SyncMap #####

###### VAE #####
#input_size= output_size
#latent_dim= 3
#timesteps= 100
#neuron_group = VAE(input_size, latent_dim, timesteps)
###### VAE #####


for i,env in enumerate(list_env):
	input_sequence, input_class = env.getSequence(sequence_length)

	neuron_group.input(input_sequence)

	labels= neuron_group.organize()
	
	print("Learned Labels: ",labels)
	print("Correct Labels: ",env.trueLabel())
	
	if save_filename is not None:

		with open(save_filename+str(i),"a+") as f:
			tmp = np.array2string(labels, precision=2, separator=',' )
			f.write(tmp+"\n")
			f.closed
		
		if labels is not None:
			with open(save_truth_filename+str(i),"a+") as f:
				tmp = np.array2string(env.trueLabel(), precision=2, separator=',')
				f.write(tmp+"\n")
				f.closed
	

	if plot == True:	
		color=None
		save= True
		filename= "plot_env"+str(i)+".png"
		neuron_group.plot(color,save,filename)

		
		input_sequence, input_class = env.getSequence(1000)
		filename= "plot_sequence_env"+str(i)+".png"
		neuron_group.plotSequence(input_sequence, input_class, filename)

