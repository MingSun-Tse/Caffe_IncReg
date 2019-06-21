#!/usr/bin/env sh

# *************************************************************************************************
# What you need to set manually:
PROJECT="compression_experiments/vgg16_spa" # set project directory
PRETRAINED="/home/wanghuan/Caffe/caffe_models/vgg16/vgg16.caffemodel" # set pretrained model
# *************************************************************************************************

SOLVER=$PROJECT/solver.prototxt
sed -i "/snapshot_prefix/d" $SOLVER 
sed -i "/net.*prototxt/d" $SOLVER
echo "snapshot_prefix: \"$PROJECT/weights/\"" >> $SOLVER
echo "            net: \"$PROJECT/train_val.prototxt\"" >> $SOLVER

TIME="$(date +%Y%m%d-%H%M)"

build/tools/caffe train \
--gpu $1 \
--weights $PRETRAINED \
--solver  $PROJECT/solver.prototxt \
2>        $PROJECT/weights/log_$TIME\_acc.txt \
1>        $PROJECT/weights/log_$TIME\_prune.txt
