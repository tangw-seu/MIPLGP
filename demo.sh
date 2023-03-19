#!/bin/bash


CUDA_VISIBLE_DEVICES='0' python main.py --ds MNIST_MIPL --ds_suffix r1 --lr 0.1 --epochs 500
