#!/bin/bash

python rnn.py >> results/high_noise/result_regular_process
python rnn_serial.py >> results/high_noise/result_serial

python rnn_2.py >> results/low_noise/result_regular_process
python rnn_serial_2.py >> results/low_noise/result_serial
