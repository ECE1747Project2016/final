#!/bin/bash

python rnn.py >> result_regular_process
python rnn_pool.py >> result_pool
python rnn_serial.py >> result_serial
