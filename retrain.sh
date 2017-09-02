#!/usr/bin/env sh
echo "ReTraining"
export WORK_DIR="test"
build/tools/caffe train  --solver   MyProject/pruning_filter/$WORK_DIR/solver_retrain.prototxt \
                         2>>         MyProject/pruning_filter/$WORK_DIR/log_20170828-0928_acc_retrain.txt \
                         1>>         MyProject/pruning_filter/$WORK_DIR/log_20170828-0928_prune_retrain.txt \
                         --gpu 0 \
                         --snapshot  MyProject/pruning_filter/$WORK_DIR/retrain_iter_15000.solverstate

                         
# --snapshot MyProject/pruning_filter/$WORK_DIR/_iter_180000.solverstate \
