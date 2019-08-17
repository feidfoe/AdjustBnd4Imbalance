#!/bin/bash

GPUID=0
dataset_path=/media/SSD/dataset/ILSVRC2015/Data/CLS-LOC
checkpoint_path=~/bj/checkpoint/pytorch-cls


dataset=imagenet
arch=resnet18
depth=32



ExpName=${dataset}_${arch}
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
                  python imagenet.py -a ${arch} \
                                     --depth $depth \
                                     --pretrained \
                                     --evaluate

                                  #--resume ${CKPT} \
