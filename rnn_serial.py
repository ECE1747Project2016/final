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
filename = "./data/signal5/signal"

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


'''
ds = SupervisedDataSet(window, window)

#filename = "./data/cassandra_data_new/cassandra_new"

#train1 = open(filename+"1",'rU').read().split('\n')
#train2 = open(filename+"2",'rU').read().split('\n')
#train1 = numpy.loadtxt(filename+"1")
#train2 = numpy.loadtxt(filename+"2")
#train3 = numpy.loadtxt(filename+"3")
#train4 = numpy.loadtxt(filename+"4")
#train5 = numpy.loadtxt(filename+"5")
#train6 = numpy.loadtxt(filename+"6")
#train7 = numpy.loadtxt(filename+"7")

trainingSet = []
for i in range (1, 1001):
	trainingSet.append( numpy.loadtxt(filename+str(i)+".txt") )

for i in range (0, 999):
	ds.addSample(trainingSet[i], trainingSet[i+1])

#ds.addSample(train1,train2)
#ds.addSample(train2,train3)
#ds.addSample(train3,train4)
#ds.addSample(train4,train5)
#ds.addSample(train5,train6)
#ds.addSample(train6,train7)
 
 
net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(100)

ts = UnsupervisedDataSet(window,)
#add the sample to be predicted
ts.addSample(trainingSet[999])

result = net.activateOnDataset(ts)
for elem in result[0]:
    print elem
'''

'''
window = 110
ds = SupervisedDataSet(window, window)

filename = "./data/mongo_data/mongo_"

#train1 = open(filename+"1",'rU').read().split('\n')
#train2 = open(filename+"2",'rU').read().split('\n')
train1 = numpy.loadtxt(filename+"1")
train2 = numpy.loadtxt(filename+"2")
train3 = numpy.loadtxt(filename+"3")
train4 = numpy.loadtxt(filename+"4")
train5 = numpy.loadtxt(filename+"5")
train6 = numpy.loadtxt(filename+"6")
train7 = numpy.loadtxt(filename+"7")
train8 = numpy.loadtxt(filename+"8")
train9 = numpy.loadtxt(filename+"9")
train10 = numpy.loadtxt(filename+"10")
train11 = numpy.loadtxt(filename+"11")
train12 = numpy.loadtxt(filename+"12")


ds.addSample(train1,train2)
ds.addSample(train2,train3)
ds.addSample(train3,train4)
ds.addSample(train4,train5)
ds.addSample(train5,train6)
ds.addSample(train6,train7)
ds.addSample(train7,train8)
ds.addSample(train8,train9)
ds.addSample(train9,train10)
 
 
net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(100)

ts = UnsupervisedDataSet(window,)
#add the sample to be predicted
ts.addSample(train10)

result = net.activateOnDataset(ts)
for elem in result[0]:
        print elem 
        
#cassandra

window = 64
ds = SupervisedDataSet(window, window)

filename = "./data/cassandra_data/cassandra_"

#train1 = open(filename+"1",'rU').read().split('\n')
#train2 = open(filename+"2",'rU').read().split('\n')
train1 = numpy.loadtxt(filename+"1")
train2 = numpy.loadtxt(filename+"2")
train3 = numpy.loadtxt(filename+"3")
train4 = numpy.loadtxt(filename+"4")
train5 = numpy.loadtxt(filename+"5")
train6 = numpy.loadtxt(filename+"6")
train7 = numpy.loadtxt(filename+"7")
train8 = numpy.loadtxt(filename+"8")
train9 = numpy.loadtxt(filename+"9")
train10 = numpy.loadtxt(filename+"10")
train11 = numpy.loadtxt(filename+"11")


ds.addSample(train1,train2)
ds.addSample(train2,train3)
ds.addSample(train3,train4)
ds.addSample(train4,train5)
ds.addSample(train5,train6)
ds.addSample(train6,train7)
ds.addSample(train7,train8)
ds.addSample(train8,train9)
ds.addSample(train9,train10)
 
 
net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(100)

ts = UnsupervisedDataSet(window,)
#add the sample to be predicted
ts.addSample(train10)

print("Cassandra...")
result = net.activateOnDataset(ts)
for elem in result[0]:
        print elem
'''
