from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer
import numpy


#cassandra_new

window = 200
ds = SupervisedDataSet(window, window)

filename = "./data/cassandra_data_new/cassandra_new"

#train1 = open(filename+"1",'rU').read().split('\n')
#train2 = open(filename+"2",'rU').read().split('\n')
train1 = numpy.loadtxt(filename+"1")
train2 = numpy.loadtxt(filename+"2")
train3 = numpy.loadtxt(filename+"3")
train4 = numpy.loadtxt(filename+"4")
train5 = numpy.loadtxt(filename+"5")
train6 = numpy.loadtxt(filename+"6")
train7 = numpy.loadtxt(filename+"7")


ds.addSample(train1,train2)
ds.addSample(train2,train3)
ds.addSample(train3,train4)
ds.addSample(train4,train5)
ds.addSample(train5,train6)
ds.addSample(train6,train7)
 
 
net = buildNetwork(window, window-1, window, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(100)

ts = UnsupervisedDataSet(window,)
#add the sample to be predicted
ts.addSample(train7)

result = net.activateOnDataset(ts)
for elem in result[0]:
        print elem
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
