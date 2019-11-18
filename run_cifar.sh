#!/bin/bash

GPUID=1
dataset_path=~/bj/dataset/
checkpoint_path=~/bj/checkpoint/pytorch-cls


dataset=CIFAR10
arch=resnet
depth=32

TRAIN=true
EVAL=false

if $TRAIN; then 
ExpName=${dataset}_${arch}_${depth}
NV_GPU=${GPUID} nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  --name ${ExpName}'_gpu'${GPUID} \
                  feidfoe/pytorch:v.2 \
                  python cifar.py -a ${arch} \
                                  --depth $depth \
                                  --epoch 300 \
                                  --schedule 150 250 \
                                  --gamma 0.1 \
                                  --wd 5e-4 \
                                  --checkpoint checkpoints/${ExpName} \
fi


if $EVAL; then
ExpName=${dataset}_${arch}_${depth}
CKPT=checkpoints/${ExpName}/checkpoint.pth.tar
NV_GPU=${GPUID} nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  --name ${ExpName}'_gpu'${GPUID} \
                  feidfoe/pytorch:v.2 \
                  python cifar.py -a ${arch} \
                                  --depth $depth \
                                  --resume ${CKPT} \
                                  --evaluate

fi
