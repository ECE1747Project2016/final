ECE1747 Project - Parallelizing Pattern Recognition 
Process Using Multiple Recurrent Neural Networks
===============================================

USAGE
-----

run any of the script below using
	python <script_name> >> <output_file>

rnn.py: parallelized training using multiprocess.Process()

rnn_pool.py: parallelized training using multiprocess.Pool()

rnn_serial.py: sequential training using a single RNN

To use different input signals, change the global 'filename'
variable at the beginning of the scripts, options include:
	"./data/signal05/signal" - low noise
	"./data/signal3/signal"  - medium noise
	"./data/signal5/signal"  - high noise

To test on different numbers of processes and input samples,
change the ranges of 'samples' and 'num_cores' of the nested
for loops at the beginning of "__main__" method

The script 'signal_average.py' can be used to calcuate the
average of different numbers of input signal samples

