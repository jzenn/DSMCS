from torch import nn as nn

from src.sampling.models import get_ais_model


class StaticTarget(nn.Module):
    def __init__(self, args, p, q, betas, deltas):
        super(StaticTarget, self).__init__()
        self.args = args
        self.p = p
        self.q = q

        self.B = args.batch_size
        self.N = args.n_particles
        self.D = args.zdim

        self.ais = get_ais_model(args, p, q, deltas, betas)

    def forward(self, data, **kwargs):
        z = self.q.sample((self.B, self.N))
        return_dict = self.ais(z, x=data, **kwargs)
        return return_dict
