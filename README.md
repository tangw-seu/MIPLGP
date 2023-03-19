# A PyTorch Implementation of MIPLGP

This is a PyTorch implementation of our paper "[Multi-Instance Partial-Label Learning: Towards Exploiting Dual Inexact Supervision](http://palm.seu.edu.cn/zhangml/files/SCIS'23.pdf). Science China Information Sciences, *in press*." 

Authors: Wei Tang, [Weijia Zhang](https://www.weijiazhangxh.com/), and [Min-Ling Zhang](http://palm.seu.edu.cn/zhangml/)

```
@article{tang2023mipl,
  title={Multi-Instance Partial-Label Learning: Towards Exploiting Dual Inexact Supervision},
  author={Wei Tang and Weijia Zhang and Min-Ling Zhang},
  journal={Science China Information Sciences},
  year={2023}
}
```



## Requirements

```sh
gpytorch==1.8.0
numpy==1.21.5
scipy==1.7.3
torch==1.12.0
```

To install the requirement packages, please run the following command:

```sh
pip install -r requirements.txt
```



## Datasets

The datasets used in this paper can be found on this [link](http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data).



## Demo

To reproduce the results of MNIST_MIPL dataset in the paper, please run the following command:

```sh
bash demo.sh
```



This package is only free for academic usage. Have fun!

