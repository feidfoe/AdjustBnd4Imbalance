#!/bin/bash

GPUID=0
dataset_path=~/bj/dataset/
checkpoint_path=~/bj/checkpoint/AdjustBnd

dataset=cifar10
arch=resnet
depth=32
imb=100

ExpName=${dataset}_${arch}${depth}_imb${imb}


TRAIN=true
EVAL=false



if $TRAIN; then 
NV_GPU=${GPUID} nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  --name ${ExpName} \
                  feidfoe/pytorch:v.2 \
                  python cifar.py -a ${arch} \
                                  --depth $depth \
                                  --imbalance $imb \
                                  --WVN \
                                  --checkpoint checkpoints/${ExpName}
fi


if $EVAL; then
CKPT=checkpoints/${ExpName}/checkpoint.pth.tar
#CKPT=checkpoints/${ExpName}/model_best.pth.tar
NV_GPU=${GPUID} nvidia-docker run -v `pwd`:`pwd` \
                  -v ${dataset_path}:`pwd`/data/ \
                  -v ${checkpoint_path}:`pwd`/checkpoints/ \
                  -w `pwd` \
                  --rm -it \
                  --ipc=host \
                  --name ${ExpName} \
                  feidfoe/pytorch:v.2 \
                  python cifar.py -a ${arch} \
                                  --depth $depth \
                                  --imbalance $imb \
                                  --RS 0.5 \
                                  --checkpoint checkpoints/${ExpName} \
                                  --resume ${CKPT} \
                                  --evaluate

fi
