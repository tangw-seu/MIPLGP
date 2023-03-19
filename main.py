#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import gpytorch
import os
import time
import gc
import torch.optim as optim
from utils_data import *
from model import ExactGPModel_multiGPUs, DisDirichletClassificationLikelihood

parser = argparse.ArgumentParser(description="MIPLGP")
parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--epsilon', type=float, default=0.0001, help='the value of alpha_epsilon')
parser.add_argument('--sample_size', type=int, default=512, help='number of sampling bags')
parser.add_argument('--ds', type=str, default='MNIST_MIPL', help='dataset')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--index', type=str, default='index', help='')
parser.add_argument('--ds_suffix', type=str, default='r1', help='the specific type of the data set')
parser.add_argument('--data_path', type=str, default='./data', help='directory of data')

args = parser.parse_args()
seed_everything(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is available.')

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
             'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']

# MNIST_MIPL
nr_class = 5 + 1
nr_fea = 784
nr_all_te = 2500


def find_best_gpu_setting(x_train, y_train, preconditioner_size):
    N = x_train.size(0)
    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N))))]
    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train(x_train, y_train, checkpoint_size_para=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)
            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()


def train(x_train, y_train, checkpoint_size_para, preconditioner_size, n_training_iter):
    nr_x_train = x_train.shape[0]
    likelihood = DisDirichletClassificationLikelihood(num_x_tr=nr_x_train, targets=y_train, num_classes=nr_class,
                            alpha_epsilon=args.epsilon, learn_additional_noise=True).to(device)
    model = ExactGPModel_multiGPUs(x_train, likelihood.transformed_targets,
                                   likelihood, nr_class, n_devices).to(device)
    transformed_targets = likelihood.transformed_targets
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Includes GaussianLikelihood parameters
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size_para), \
            gpytorch.settings.max_preconditioner_size(preconditioner_size):
        for ep in range(n_training_iter):
            optimizer.zero_grad()       # Zero gradients from previous iteration
            output = model(x_train)     # Output from model
            loss = -mll(output, transformed_targets).mean()
            _, transformed_targets, alpha = likelihood._prepare_targets(num_x_tr=nr_x_train, targets=y_train,
                                                                        num_classes=nr_class, out=output.loc,
                                                                        alpha_epsilon=args.epsilon,
                                                                        dtype=torch.float)
            loss.backward()
            if ep == 0 or (ep + 1) % 10 == 0:
                print('Iter %d/%d - Loss: %.3f' % (ep + 1, n_training_iter, loss.item()))
            optimizer.step()
            scheduler.step()
        print(f"Finished training on {x_train.size(0)} data points using {n_devices} GPUs.")
    return likelihood, model


def prediction(likelihood, model, x_test, idx_test, bag_id_test):
    likelihood.eval().to(device)
    model.eval().to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
        # Make predictions on a small number of test points to get the test time caches computed
        test_dist = model(x_test[:10, :])
        del test_dist  # We don't care about these predictions, we really just want the caches.

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
        test_dist = model(x_test)

    pred_samples = test_dist.sample(torch.Size((args.sample_size,))).exp()
    pred_samples = pred_samples[:, :nr_class - 1, :]
    prob = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    y_pred = []
    prob = prob.cpu().numpy()
    for i in range(len(idx_test)):
        idx = np.where(bag_id_test == i)
        idx = np.array(idx).flatten()
        prob_ins_a_bag = prob[:, idx]
        m = np.argmax(prob_ins_a_bag)
        y_pred_a_bag, _ = divmod(m, prob_ins_a_bag.shape[1])
        y_pred.append(y_pred_a_bag)

    return y_pred


if __name__ == '__main__':
    time_s = time.time()
    data_path = os.path.join(args.data_path, args.ds)
    index_path = os.path.join(data_path, args.index)

    mat_name = args.ds + '_' + args.ds_suffix + '.mat'
    mat_path = os.path.join(data_path, mat_name)
    ds_name = mat_name[0:-4]
    data_mat = io.loadmat(mat_path)
    data = data_mat['data']
    acc_bag = np.zeros([1, len(all_folds)])
    acc_con_bag = np.zeros([nr_all_te, 3]) 

    for k in range(len(all_folds)):
        x_tr, y_tr, x_te, y_bag_te, y_ins_te, bag_id_te, index_tr, index_te = \
            get_data(data, index_path, k, all_folds, nr_fea, nr_class)

        # Set a large enough preconditioner size to reduce the number of CG iterations run
        preconditioner_size = 100
        find_best_gpu_setting(x_tr, y_tr, preconditioner_size=preconditioner_size)
        checkpoint_size_def = 10000
        likelihood, model = train(x_tr, y_tr, checkpoint_size_def, preconditioner_size, args.epochs)
        y_bag_hat = prediction(likelihood, model, x_te, index_te, bag_id_te)
        y_bag_hat = np.array(y_bag_hat)
        y_bag_te = np.array(y_bag_te)
        acc_bag[0, k] = np.mean(y_bag_hat == y_bag_te)
        print('acc_bag[%d, %d]:' % (1, k + 1), acc_bag[0, k])
    print('Mean Accuracy:', np.mean(acc_bag[0, :]))

    time_e = time.time()
    print('Running time is', time_e - time_s, 'seconds.\n')
    print('Training is finished')


