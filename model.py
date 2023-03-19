#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import gpytorch
from torch import Tensor
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from typing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExactGPModel_multiGPUs(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, n_devices):
        super(ExactGPModel_multiGPUs, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((num_classes,)))
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices), output_device=device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DisDirichletClassificationLikelihood(FixedNoiseGaussianLikelihood):
    def _prepare_targets_init(self, num_x_tr, targets, num_classes, alpha_epsilon=0.01, dtype=torch.float):
        alpha = alpha_epsilon * torch.ones(num_x_tr, num_classes, dtype=dtype)
        alpha = alpha.to(device)
        for i in range(targets.shape[0]):
            alpha[i, :] = alpha[i, :] + (targets[i, :] > 0) * (1.0 / sum(targets[i, :]))
        sigma2_i = torch.log(1 / alpha + 1.0)
        transformed_targets = alpha.log() - 0.5 * sigma2_i

        return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.type(dtype), alpha

    def _prepare_targets(self, num_x_tr, targets, num_classes, out, alpha_epsilon=0.01, dtype=torch.float):
        alpha = alpha_epsilon * torch.ones(num_x_tr, num_classes, dtype=dtype)
        alpha = alpha.to(device)
        out_candi = out * (targets.T > 0)
        prob = ((out_candi.exp() * (targets.T > 0)) / (out_candi.exp() * (targets.T > 0)).sum(-2, keepdim=True))
        alpha = alpha + prob.T
        sigma2_i = torch.log(1 / alpha + 1.0)
        transformed_targets = alpha.log() - 0.5 * sigma2_i
        transformed_targets = transformed_targets.T

        return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.type(dtype), alpha

    def __init__(self, targets: Tensor, alpha_epsilon: int = 0.01, learn_additional_noise: Optional[bool] = False,
                 dtype: Optional[torch.dtype] = torch.float, num_classes: int = 10, num_x_tr: int = 100, **kwargs, ):
        sigma2_labels, transformed_targets, alpha = self._prepare_targets_init(num_x_tr, targets, num_classes,
                                                                               alpha_epsilon=alpha_epsilon, dtype=dtype)

        super().__init__(noise=sigma2_labels, learn_additional_noise=learn_additional_noise,
                         batch_shape=torch.Size((num_classes,)), **kwargs, )
        self.transformed_targets = transformed_targets.transpose(-2, -1)
        self.num_classes = num_classes
        self.targets = targets
        self.alpha_epsilon = alpha_epsilon

    def __call__(self, *args, **kwargs):
        if "targets" in kwargs:
            targets = kwargs.pop("targets")
            dtype = self.transformed_targets.dtype
            new_noise, _, _ = self._prepare_targets_init(targets, dtype=dtype)
            kwargs["noise"] = new_noise
        return super().__call__(*args, **kwargs)
