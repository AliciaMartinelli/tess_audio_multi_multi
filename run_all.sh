#!/bin/bash

set -e

mkdir -p log

run_with_log() {
    script=$1
    feature=$2
    log_file="log/${script%.*}_${feature}.log"
    echo "Running: python $script $feature"
    python "$script" "$feature" &> "$log_file"
    echo "Finished: $script $feature → log: $log_file"
}

run_with_log train_RF.py rms
run_with_log train_RF.py zcr
run_with_log train_RF.py mfcc
run_with_log train_KNN.py rms
run_with_log train_KNN.py zcr
run_with_log train_KNN.py mfcc
run_with_log train_XGB.py rms
run_with_log train_XGB.py zcr
run_with_log train_XGB.py mfcc
run_with_log train_Ridge.py rms
run_with_log train_Ridge.py zcr
run_with_log train_Ridge.py mfcc
run_with_log train_XGB.py mel_spectrogram
run_with_log train_KNN.py mel_spectrogram
run_with_log train_RF.py mel_spectrogram
run_with_log train_Ridge.py mel_spectrogram
