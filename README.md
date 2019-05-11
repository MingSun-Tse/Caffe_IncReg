

## Trained models
https://drive.google.com/open?id=112OyHziceXoww6rZ_aIjTUzpKElg0sFd

(More discriptions will be updated soon update.)

| model | original accuracy | speedup ratio | pruned accuracy |
| :-: | :-: | :-: | :-: |
| vgg16 |       [0.702/0.896](xx)     | 2x | [0.702/0.896](xx) |
| resnet50 |    [0.702/0.896](xx)     | 2x | [0.702/0.896](xx) |
| inceptionv3 | [0.702/0.896](xx)     | 2x | [0.702/0.896](xx) |

Note: speedup ratio is the theoretical value measured by FLOPs reduction in *only conv* layers.


## Environment
- Ubuntu 1404
- Python 2.7
- Use cudnn

## How to run the code
1. Download this repo and compile: `make -j24`, see Caffe's [official guide](http://caffe.berkeleyvision.org/installation.html). Make sure you get it through. 
2. Here we show how to run the code, taking lenet5 as an example:
    - Preparation: 
        - Data: Create your mnist training and testing lmdb (either you can download [ours](https://drive.google.com/open?id=1zMbKKfOFXH3chi9xdwCPi14YfqRzC_pe)), put them in `data/mnist/mnist_train_lmdb` and `data/mnist/mnist_test_lmdb`. 
        - Pretrained model: We provide a pretrained lenet5 model in `compression_experiments/mnist/weights/baseline_lenet5.caffemodel` (test accuracy = 0.991).
    - (We have set up an experiment folder in `compression_experiments/lenet5`, where there are three files: `train.sh, solver.prototxt, train_val.prototxt`. There are some path settings in them and pruning configs in `solver.prototxt`, where we have done that for you, but you are free to change them.)
    - In your caffe root path, run `nohup  sh  compression_experiments/lenet5/train.sh  <gpu_id>  >  /dev/null  &`, then you are all set! Check your log at `compression_experiments/lenet5/weights`.

For vgg16, resnet50, and inceptionv3, we also provided their experiment folders in `compression_experiments`, check them out and have a try!

## Check the log
There are two logs generated during pruning: `log_<TimeID>_acc.txt` and `log_<TimeID>_prune.txt`. The former saves the logs printed by the original Caffe; the latter saves the logs printed by our added codes.

Go to the project folder, e.g., `compression_experiments/lenet5` for lenet5, then run `cat weights/*prune.txt | grep app` you will see the pruning and retraining course.

## Detailed explanation of the options in solver.prototxt
- target_reg:
- IF_eswpf:

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite these in your publications if this code helps your research:

    @proceedins{huan2019increg,
      Author = {Wang, Huan and Zhang, Qiming and Wang, Yuehai and Yu, Lu and Hu, Haoji},
      Title = {Structured Pruning for Efficient ConvNets via Incremental Regularization},
      Booktitle = {IJCNN},
      Year = {2019}
    }
    @proceedins{huan2018spp,
      Author = {Wang, Huan and Zhang, Qiming and Wang, Yuehai and Hu, Haoji},
      Title = {Structured probabilistic pruning for convolutional neural network acceleration},
      Booktitle = {BMVC},
      Year = {2018}
    }
    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
