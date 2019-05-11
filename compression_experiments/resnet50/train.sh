#!/usr/bin/env sh

# Set bin
CAFFE_ROOT="."

# Set project
PROJECT="compression_experiments/resnet50"
TIME="$(date +%Y%m%d-%H%M)"

# Set pretrained model
PRETRAINED="/home/wanghuan/Caffe/caffe_models/resnet50/baseline_iter_57000_acc0.74928_0.92300.caffemodel"

$CAFFE_ROOT/build/tools/caffe train \
--gpu $1 \
--weights $PRETRAINED \
--solver  $PROJECT/solver.prototxt \
2>        $PROJECT/weights/log_$TIME\_acc.txt \
1>        $PROJECT/weights/log_$TIME\_prune.txt