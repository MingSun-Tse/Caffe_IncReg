#!/usr/bin/env sh
echo "Training"
export WORK_DIR="02_on_alexnet"
build/tools/caffe train  --solver   MyProject/pruning_filter/$WORK_DIR/solver.prototxt \
                         2>         MyProject/pruning_filter/$WORK_DIR/weights/log_$(date +%Y%m%d-%H%M)_acc.txt \
                         1>         MyProject/pruning_filter/$WORK_DIR/weights/log_$(date +%Y%m%d-%H%M)_prune.txt \
                         --gpu 2 \
                         --weights  MyProject/pruning_filter/$WORK_DIR/_iter_50000_converted.caffemodel

                         
# --snapshot MyProject/pruning_filter/$WORK_DIR/_iter_180000.solverstate \