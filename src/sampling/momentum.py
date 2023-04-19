import torch
from torch import nn as nn


class DiagonalMassMatrix(nn.Module):
    def __init__(self, dim):
        super(DiagonalMassMatrix, self).__init__()
        self.dim = dim
        self._diagonal = nn.Parameter(torch.zeros(dim))

    def get_diagonal(self):
        return self._diagonal

    def cholesky(self):
        return torch.diag(torch.exp(1 / 2 * self.get_diagonal()))

    def precision(self):
        return self.inv()

    def logdet(self):
        return torch.sum(self.get_diagonal())

    def inv(self):
        return torch.diag(torch.exp(-self.get_diagonal()))

    def diag(self):
        return torch.exp(self.get_diagonal())

    def forward(self):
        return torch.diag(torch.exp(self.get_diagonal()))

    def dim(self):
        return self.dim


class ScaledDiagonalMassMatrix(DiagonalMassMatrix):
    def __init__(self, dim):
        super(DiagonalMassMatrix, self).__init__()
        self.dim = dim
        self._scalar_diagonal = nn.Parameter(torch.zeros(1))

    def get_diagonal(self):
        return self._scalar_diagonal * torch.ones(
            self.dim, device=self._scalar_diagonal.device
        )


class MomentumRefreshmentFactor(nn.Module):
    def __init__(self):
        super(MomentumRefreshmentFactor, self).__init__()
        self.u = nn.Parameter(torch.ones(1) * 0.95)

    def forward(self):
        return 0.98 * torch.sigmoid(self.u) + 0.01
