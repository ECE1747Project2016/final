import numpy

filename = "./data/signal05/signal"

if __name__ == "__main__":
    for samples in (50, 100, 200, 400, 800, 1600):
        t_set = []
        for i in range(samples):
            t_set.append(numpy.loadtxt(filename + str(i + 1) + ".txt"))

        arr = numpy.array(t_set)
        print "number of samples:", samples
        arr_mean = numpy.mean(arr, axis=0)
        for elem in arr_mean:
            print elem
