from abc import ABC

import torch
import torch.autograd as autograd
import torch.nn as nn


class AnnealedImportanceSampling(nn.Module, ABC):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(AnnealedImportanceSampling, self).__init__()
        self.args = args
        self.target_dist = log_joint
        self.initial_dist = log_variational
        self.deltas = deltas
        self.betas = betas
        #
        self.B = args.batch_size
        self.N = args.n_particles
        self.K = args.n_transitions
        self.D = args.zdim

    def log_gamma(self, beta, z, **kwargs):
        return (1 - beta) * self.initial_dist.log_prob(
            z, **kwargs
        ) + beta * self.target_dist.log_prob(z, **kwargs)

    def grad_log_gamma(self, beta, z, **kwargs):
        with torch.enable_grad():
            z.requires_grad_()
            grad = autograd.grad(
                self.log_gamma(beta, z, **kwargs).sum(),
                z,
                create_graph=self.training,
            )[0]
        return grad
