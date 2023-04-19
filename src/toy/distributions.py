import math

import torch
from torch import distributions as dist
from torch import nn


class GaussianMixtureToyJoint(nn.Module):
    def __init__(self, args, mean=3.0, diagonal_covariance=1.0, num_components=8):
        super(GaussianMixtureToyJoint, self).__init__()
        self.dim = args.zdim
        self.num_components = num_components

        component_distribution = dist.MultivariateNormal(
            torch.zeros(self.dim) + mean,
            covariance_matrix=torch.eye(self.dim) * diagonal_covariance,
        )
        cholesky_factor = torch.eye(self.dim) * math.sqrt(diagonal_covariance)
        covariance_diagonal = torch.diag(cholesky_factor) ** 2
        precision = torch.diag(1.0 / covariance_diagonal)
        mean = component_distribution.sample((num_components,))

        self.register_buffer("mean", mean)
        self.register_buffer("precision", precision)
        self.register_buffer("cholesky_factor", cholesky_factor)
        self.register_buffer(
            "log_det_covariance", torch.sum(torch.log(covariance_diagonal))
        )

    def log_prob(self, z, **kwargs):
        # z: (B, N, D) -> (1, B, N, D)
        z_ = z.unsqueeze(0)

        # mean: (n, D) -> (n, 1, 1, D)
        mean_ = self.mean.unsqueeze(1).unsqueeze(1)

        # log-prob
        diff = z_ - mean_
        M = torch.sum(diff @ self.precision * diff, -1)
        pdf_z_per_component = -0.5 * (
            self.dim * math.log(2 * math.pi) + M + self.log_det_covariance
        )
        pdf_z = pdf_z_per_component.logsumexp(0) - math.log(self.num_components)
        return pdf_z


class SimpleGaussianToyVariational(nn.Module):
    def __init__(self, args, diagonal_covariance=1.0, mean=None):
        super(SimpleGaussianToyVariational, self).__init__()
        self.dim = args.zdim
        self.register_buffer("mean", torch.zeros(self.dim) if mean is None else mean)
        self.register_buffer(
            "covariance_matrix", torch.eye(self.dim) * diagonal_covariance
        )
        self.N = dist.MultivariateNormal(
            loc=self.mean, covariance_matrix=self.covariance_matrix
        )

    def _apply(self, fn):
        # call apply of super class
        super()._apply(fn)
        # deal with parameters of distribution(s)
        self.N.loc = fn(self.N.loc)
        if self.N.scale_tril is not None:
            self.N.scale_tril = fn(self.N.scale_tril)
        if self.N._unbroadcasted_scale_tril is not None:
            self.N._unbroadcasted_scale_tril = fn(self.N._unbroadcasted_scale_tril)
        if self.N.covariance_matrix is not None:
            self.N.covariance_matrix = fn(self.N.covariance_matrix)
        if self.N.precision_matrix is not None:
            self.N.precision_matrix = fn(self.N.precision_matrix)
        return self

    def log_prob(self, z, **kwargs):
        pdf_z = self.N.log_prob(z)
        return pdf_z

    def sample(self, shape):
        return self.N.sample(shape)


class Normal:
    # taken and adapted from the submission of Zhang et al. (2021)
    # https://openreview.net/forum?id=6rqjgrL7Lq
    def __init__(self, mean, covariance):
        assert (
            mean.ndim == 3
            and covariance.D() == 2
            and mean.shape[1] == 1
            and mean.shape[2] == 1
        )
        self.mean = mean
        self.covariance = covariance
        self.precision = torch.inverse(covariance)
        self.cholesky_lower_covariance = torch.linalg.cholesky(covariance)
        self.d = mean.shape[0]
        self.log_det_covariance = torch.logdet(covariance)

    def log_prob(self, x):
        assert x.shape[-1] == self.d
        diff = x - self.mean.T
        M = torch.sum(diff @ self.precision * diff, -1)
        return -0.5 * (self.d * math.log(2 * math.pi) + M + self.log_det_covariance)

    def sample(self, shape: tuple):
        """
        Sample via reparameterization
        :param n: number of desired samples
        :return: n samples in an (n, d) tensor
        """
        Eps = torch.empty(
            (self.d, *shape), dtype=self.mean.dtype, device=self.mean.device
        ).normal_()
        return Eps.permute(
            1, 2, 0
        ) @ self.cholesky_lower_covariance.T + self.mean.permute(
            *torch.arange(self.mean.ndim - 1, -1, -1)
        )


def normal_log_prob(x, mean, precision, log_det_covariance):
    d = x.shape[-1]
    diff = x - mean
    M = torch.sum(diff @ precision * diff, -1)
    return -0.5 * (d * math.log(2 * math.pi) + M + log_det_covariance)


def sample_normal(mean, cholesky_lower_covariance, shape):
    d = mean.shape[-1]
    eps = torch.empty((*shape, d), dtype=mean.dtype, device=mean.device).normal_()
    return eps @ cholesky_lower_covariance.T + mean
