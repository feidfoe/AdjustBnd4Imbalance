# Adjusting Decision Boundary for Class Imbalanced Learning
This repository is the official PyTorch implementation of WVN-RS.


### Requirements
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

### How to use
Run the shell script.
```
bash run_cifar.sh
```
To use Weight Vector Normalization (WVN), use --WVN flag. (It is already in the script.)

### Results
1. *Validation error* on Long-Tailed CIFAR10

Imbalance|200|100|50|20|10|1
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Baseline   | 35.67 | 29.71 | 22.91 | 16.04 | 13.26 | 6.83
Over-sample| 32.19 | 28.27 | 21.40 | 15.23 | 12.24 | 6.61
Focal      | 34.71 | 29.62 | 23.28 | 16.77 | 13.19 | 6.60 
CB         | 31.11 | 25.43 | 20.73 | 15.64 | 12.51 | 6.36 
LDAM-DRW   | 28.09 | 22.97 | 17.83 | 14.53 | 11.84 | 6.32 
Baseline+RS| 27.02 | 21.36 | 17.16 | 13.46 | 11.86 | 6.32 
WVN+RS     | 27.23 | 20.17 | 16.80 | 12.76 | 10.71 | 6.29 


2. Validation error on Long-Tailed CIFAR100
Imbalance|200|100|50|20|10|1
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Baseline   | 64.21 | 60.38 | 55.09 | 48.93 | 43.52 | 29.69
Over-sample| 66.39 | 61.53 | 56.65 | 49.03 | 43.38 | 29.41
Focal      | 64.38 | 61.31 | 55.68 | 48.05 | 44.22 | 28.52
CB         | 63.77 | 60.40 | 54.68 | 47.41 | 42.01 | 28.39
LDAM-DRW   | 61.73 | 57.96 | 52.54 | 47.14 | 41.29 | 28.85
Baseline+RS| 59.59 | 55.65 | 51.91 | 45.09 | 41.45 | 29.80
WVN+RS     | 59.48 | 55.50 | 51.80 | 46.12 | 41.02 | 29.22




### Notes
This codes use docker image "feidfoe/pytorch:v.2" with pytorch version, '0.4.0a0+0640816'.
The image only provides basic libraries such as NumPy or PIL.

WVN is implemented on ResNet architecture only.

### Contact
[Byungju Kim](https://feidfoe.github.io/) (byungju.kim@kaist.ac.kr)

#### Baseline repository
This repository is forked and modified from [original repo](https://github.com/bearpaw/pytorch-classification).

