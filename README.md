# Adjusting Decision Boundary for Class Imbalanced Learning
This repository contains PyTorch implementation of WVN-RS.


## Requirements
1. NVIDIA docker : Docker image will be pulled from cloud.
2. CIFAR dataset : The "dataset_path" in run_cifar.sh should be
```
cifar10/
    data_batch_N
    test_batch
cifar100/
    train
    test
```
They are available [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## How to use
Run the shell script.



## Notes
This codes use docker image "feidfoe/pytorch:v.2" with pytorch version, '0.4.0a0+0640816'.
The image only provides basic libraries such as NumPy or PIL.

WVN is implemented on ResNet architecture only.

## Contact
[Byungju Kim](https://feidfoe.github.io/) (byungju.kim@kaist.ac.kr)

### Baseline repository
This repository is forked and modified from [original repo](https://github.com/bearpaw/pytorch-classification).

