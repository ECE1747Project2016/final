#!/bin/bash

python rnn.py >> results/low_noise/result_with_bug_correction
python rnn_2.py >> results/medium_noise/result_with_bug_correction
