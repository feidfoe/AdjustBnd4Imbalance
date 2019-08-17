#!/bin/bash

GPUID=0
dataset_path=~/bj/dataset/
checkpoint_path=~/bj/checkpoint/pytorch-cls


dataset=cifar10
arch=resnet
depth=32



ExpName=${dataset}_${arch}_${depth}
CKPT=checkpoints/${ExpName}/checkpoint.pth.tar
nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  -e CUDA_VISIBLE_DEVICES=${GPUID} \
                  --name ${ExpName}'_gpu'${GPUID} \
                  feidfoe/pytorch:v.2 \
                  python cifar.py -a ${arch} \
                                  --depth $depth \
                                  --resume ${CKPT} \
                                  --evaluate

