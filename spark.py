#!/usr/bin/python
# -*- coding: utf-8 -*-
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.structure import LinearLayer
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

import math
import multiprocessing
import numpy
import sys
import time

window = 200
filename = './data/signal3/signal'

# for samples in (40):

samples = 40

# for num_cores in (4):

num_cores = 2


def partialTraining(index, length, result_queue):

    # change t_set to RDD

    actualLength = 0
    t_set = []
    for i in range(index, index + length):
        try:
            t_set.append(numpy.loadtxt(filename + str(i + 1) + '.txt'))
            actualLength += 1
        except Exception, e:
            break

    d_set = SupervisedDataSet(window, window)
    for i in range(0, actualLength - 1):
        d_set.addSample(t_set[i], t_set[i + 1])

    network = buildNetwork(
        window,
        window - 1,
        window,
        outclass=LinearLayer,
        bias=True,
        recurrent=True,
        )
    bpTrainer = BackpropTrainer(network, d_set)
    bpTrainer.trainEpochs(100)

    t_s = UnsupervisedDataSet(window)

    # add the sample to be predicted

    t_s.addSample(t_set[actualLength - 1])

    result = network.activateOnDataset(t_s)
    result_queue.put(result[0])


############MAP##############################################

def training(t_set):
    d_set = SupervisedDataSet(window, window)
    for i in len(t_set):
        d_set.addSample(t_set[i], t_set[i + 1])

    network = buildNetwork(
        window,
        window - 1,
        window,
        outclass=LinearLayer,
        bias=True,
        recurrent=True,
        )
    bpTrainer = BackpropTrainer(network, d_set)
    bpTrainer.trainEpochs(100)

    t_s = UnsupervisedDataSet(window)

    # add the sample to be predicted

    t_s.addSample(t_set[actualLength - 1])

    result = network.activateOnDataset(t_s)
    return result


###########REDUCE#############################################

def final_training(results):
    finalTrainingSet = []
    size = num_cores
    ds = SupervisedDataSet(window, window)

    while not results.empty():
        finalTrainingSet.append(results.get())

    for i in range(size - 1):
        ds.addSample(finalTrainingSet[i], finalTrainingSet[i + 1])

    net = buildNetwork(
        window,
        window - 1,
        window,
        outclass=LinearLayer,
        bias=True,
        recurrent=True,
        )
    trainer = BackpropTrainer(net, ds)
    trainer.trainEpochs(100)

    ts = UnsupervisedDataSet(window)
    ts.addSample(finalTrainingSet[size - 1])

    finalResult = net.activateOnDataset(ts)
    return finalResult


if __name__ == '__main__':

    conf = SparkConf().setAppName('rnn_spark')
    sc = SparkContext(conf=conf)

    for tm in range(1):
        t0 = time.time()

        # jobs = []
        # tSet_length = 1024 / num_cores
        # ################
        # read files

        t_set = []
        for i in range(1, samples + 1):
            try:
                t_set.append(numpy.loadtxt(filename + str(i + 1)
                             + '.txt'))
            except Exception, e:

                # actualLength += 1

                break

        rdd = sc.parallelize(t_set, num_cores)

        intermediate_result = rdd.map(training).collect()

        final_result = \
            intermediate_result.reduce(final_training).collect()

        # ############################

        t1 = time.time()

        print 'which loop: ', tm, 'number of samples tested: ', \
            samples,
        print 'number of cores: ', num_cores, 'running time: ', t1 - t0

        # for elem in finalResult[0]:
        #    print elem
                # IMPORTANT: MAP REDUCE HERE
                # WORK TBD

        spark.stop()


			
