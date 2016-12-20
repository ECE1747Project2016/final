from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer

import math
import multiprocessing
import numpy
import sys
import time

window = 200
filename = "./data/signal05/signal"

def partialTraining(index, length, result_queue):
	actualLength = 0
	t_set = []
	for i in range (index, index+length):
		try:
			t_set.append( numpy.loadtxt(filename+str(i+1)+".txt") )
			actualLength += 1
		except Exception as e:
			break

	d_set = SupervisedDataSet(window, window)
	for i in range (0, actualLength-1):
		d_set.addSample(t_set[i], t_set[i+1])

	network = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
	bpTrainer = BackpropTrainer(network, d_set)
	bpTrainer.trainEpochs(100)

	t_s = UnsupervisedDataSet(window,)
	#add the sample to be predicted
	t_s.addSample(t_set[actualLength-1])

	result = network.activateOnDataset(t_s)
	result_queue.put(result[0])


if __name__ == "__main__":
	for size in (100, 200, 400, 800, 1600):
		#size = 100

		result_list = []
		time_list = []

		print "number of samples tested: ", size

		for tm in range(10):
			t0 = time.time()

			finalTrainingSet = []
			#size = 10000
			ds = SupervisedDataSet(window, window)

			for i in range (size):
			    finalTrainingSet.append( numpy.loadtxt(filename + str(i+1) + ".txt") )

			for i in range (size-1):
			    ds.addSample(finalTrainingSet[i], finalTrainingSet[i+1])

			net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
			trainer = BackpropTrainer(net, ds)
			trainer.trainEpochs(100)

			ts = UnsupervisedDataSet(window,)
			ts.addSample(finalTrainingSet[size-1])

			finalResult = net.activateOnDataset(ts)


			t1 = time.time()

			time_list.append(t1 - t0)
			result_list.append(finalResult[0])

			#for elem in finalResult[0]:
			#    print elem
		print "time average: ", numpy.mean(time_list)
		print "time std deviation", numpy.std(time_list)

		arr = numpy.array(result_list)
		
		print "signal average:"
		arr_mean = numpy.mean(arr, axis=0)
		for elem in arr_mean:
			print elem

		print "signal std deviation:"
		arr_std = numpy.std(arr, axis=0)
		for elem in arr_std:
			print elem

		avg_std = numpy.mean(arr_std, axis=0)
		print "average signal standard deviation:", avg_std

