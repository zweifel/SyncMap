from keras.utils import np_utils
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



save_dir= "output_files/"

arg_size = len(sys.argv)
if arg_size > 1:
	problem_type= sys.argv[1]
	save_filename = save_dir + sys.argv[2]
	save_truth_filename = save_dir + sys.argv[2] + "_truth"
else:
	save_filename= None
	problem_type = None


time_delay = 10

print("problem type:",problem_type)

problem_type = int(problem_type)

if problem_type == 1:
	env= GraphWalkTest(time_delay)
if problem_type == 2:
	env = FixedChunkTest(time_delay)
if problem_type == 3:
	env= GraphWalkTest(time_delay, "sequence2.dot")
if problem_type == 4:
	env= GraphWalkTest(time_delay, "sequence1.dot")
if problem_type == 5:
	env = LongChunkTest(time_delay)
if problem_type == 6:
	env = OverlapChunkTest1(time_delay)
if problem_type == 7:
	env = OverlapChunkTest2(time_delay)


output_size= env.getOutputSize()


print("Output Size",output_size)




sequence_length = 100000

####### SyncMap #####
number_of_nodes= output_size
adaptation_rate= 0.001*output_size
#adaptation_rate= 0.01*output_size
#adaptation_rate= 0.1/output_size
print("Adaptation rate:", adaptation_rate)
map_dimensions= 2
neuron_group= SyncMap(number_of_nodes, map_dimensions, adaptation_rate)
####### SyncMap #####

###### VAE #####
#input_size= output_size
#latent_dim= 3
#timesteps= 100
#neuron_group = VAE(input_size, latent_dim, timesteps)
###### VAE #####



input_sequence, input_class = env.getSequence(sequence_length)

neuron_group.input(input_sequence)
labels= neuron_group.organize()

print("Learned Labels: ",labels)
print("Correct Labels: ",env.trueLabel())

if save_filename is not None:

	with open(save_filename,"a+") as f:
		tmp = np.array2string(labels, precision=2, separator=',')
		f.write(tmp+"\n")
		f.closed
	
	if labels is not None:
		with open(save_truth_filename,"a+") as f:
			tmp = np.array2string(env.trueLabel(), precision=2, separator=',')
			f.write(tmp+"\n")
			f.closed

#exit()

#color=None
#save= True
#neuron_group.plot(color,save)


#input_sequence, input_class = env.getSequence(1000)
#neuron_group.plotSequence(input_sequence, input_class)




