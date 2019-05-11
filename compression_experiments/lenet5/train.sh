#!/usr/bin/env sh

# Set bin
CAFFE_ROOT="."

# Set project
PROJECT="compression_experiments/lenet5"
TIME="$(date +%Y%m%d-%H%M)"

# Set pretrained model
PRETRAINED="compression_experiments/lenet5/weights/baseline_lenet5.caffemodel"

$CAFFE_ROOT/build/tools/caffe train \
--gpu $1 \
--weights $PRETRAINED \
--solver  $PROJECT/solver.prototxt \
2>        $PROJECT/weights/log_$TIME\_acc.txt \
1>        $PROJECT/weights/log_$TIME\_prune.txt