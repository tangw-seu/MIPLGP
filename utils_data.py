#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import scipy.io as io
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


def get_data(data, fold_path, k, name_folds, fea_num, class_num):
    idx_a_fold = fold_path + '/' + name_folds[k]
    print('\n-----', idx_a_fold, '-----')

    Xtrain, Xtest = [], []
    Xtrain_tmp, Xtest_tmp = np.empty((0, fea_num)), np.empty((0, fea_num))
    lab_tr, lab_te, ins_lab_te = [], [], []
    par_lab_tr, par_lab_tr_tmp = np.zeros((0, class_num)), np.zeros((0, class_num))
    ins_num_tr, ins_num_te = [], []
    bag_id_tr, bag_id_te = [], []
    bag_tr_cnt, bag_te_cnt = 0, 0

    idx = io.loadmat(idx_a_fold)
    idx_tr_np, idx_te_np = idx['trainIndex'],idx['testIndex']
    idx_tr, idx_te = list(np.array(idx_tr_np).flatten()), list(np.array(idx_te_np).flatten())
    random.shuffle(idx_tr)
    random.shuffle(idx_te)
    tr_bags_num = len(idx_tr)
    par_lab_tr = np.empty((tr_bags_num, class_num))

    for i_tr in idx_tr:
        Xtrain = np.vstack((Xtrain_tmp, data[i_tr - 1, 0]))
        Xtrain_tmp = Xtrain

        ins_num_tr_tmp = data[i_tr - 1, 0].shape[0]
        ins_num_tr.append(ins_num_tr_tmp)

        bag_id_tr_tmp = [bag_tr_cnt] * ins_num_tr_tmp
        bag_id_tr = bag_id_tr + bag_id_tr_tmp
        bag_tr_cnt += 1

        lab_tr_tmp = list(data[i_tr - 1, 2].flatten() - 1)
        lab_tr = lab_tr + lab_tr_tmp

        par_lab_tr_list = list(data[i_tr - 1, 1].flatten() - 1)
        par_lab_tr_list.append(class_num - 1)
        par_lab_tr = np.vstack((par_lab_tr_tmp, to_categorical(par_lab_tr_list, class_num)))
        par_lab_tr_tmp = par_lab_tr
    par_lab_tr_ins = np.zeros((0, class_num))
    for i in range(par_lab_tr.shape[0]):
        par_lab_tr_ins_tmp = np.tile(par_lab_tr[i, :], (ins_num_tr[i], 1))
        par_lab_tr_ins = np.vstack((par_lab_tr_ins, par_lab_tr_ins_tmp))

    for i_te in idx_te:
        Xtest = np.vstack((Xtest_tmp, data[i_te - 1, 0]))
        Xtest_tmp = Xtest
        ins_num_te_tmp = data[i_te - 1, 0].shape[0]
        ins_num_te.append(ins_num_te_tmp)
        bag_id_te_tmp = [bag_te_cnt] * ins_num_te_tmp
        bag_id_te = bag_id_te + bag_id_te_tmp
        bag_te_cnt += 1
        lab_te_tmp = list(data[i_te - 1, 2].flatten() - 1)
        lab_te.append(lab_te_tmp)
        ins_lab_te_tmp = lab_te_tmp * ins_num_te_tmp
        ins_lab_te = ins_lab_te + ins_lab_te_tmp

    Xtrain = torch.FloatTensor(Xtrain).to(device)
    Xtest = torch.FloatTensor(Xtest).to(device)
    par_lab_tr_ins = torch.Tensor(par_lab_tr_ins).long().to(device)
    lab_te = list(np.array(lab_te).flatten())
    ins_lab_te = np.array(ins_lab_te)
    bag_id_te = np.array(bag_id_te)

    return Xtrain, par_lab_tr_ins, Xtest, lab_te, ins_lab_te, bag_id_te, idx_tr, idx_te


def to_categorical(y, nr_class):
    y_list = [0] * nr_class
    for i in y:
        y_list[i] = 1
    y_cate = np.array(y_list)

    return y_cate
