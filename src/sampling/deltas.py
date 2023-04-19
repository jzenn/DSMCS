import torch
from torch import nn


class DeltasNN(nn.Module):
    def __init__(self, args, device):
        super(DeltasNN, self).__init__()
        self.K = args.n_transitions
        self.k_embedding = nn.Embedding(self.K, 32)
        self.hidden_swish = nn.SiLU()
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.scale = args.deltas_nn_output_scale

    def scaled_sigmoid(self, x):
        return self.scale * self.sigmoid(x)

    def forward(self, k):
        out = self.unscaled_forward(k)
        scaled_out = self.scaled_sigmoid(out)
        return scaled_out

    def unscaled_forward(self, k):
        hidden = self.k_embedding(k)
        hidden = self.hidden_swish(hidden)
        out = self.output_layer(hidden)
        return out

    def get_all_deltas(self):
        return torch.tensor(
            [self.forward(torch.tensor([i], device=self.device)) for i in range(self.K)]
        )

    def get_all_pre_activation_deltas(self):
        return torch.tensor(
            [
                self.unscaled_forward(torch.tensor([i], device=self.device))
                for i in range(self.K)
            ]
        )

    def __getitem__(self, k):
        # k in range {1, ..., K}
        delta = self.forward(torch.tensor(k - 1, device=self.device))
        return delta


def get_deltas(args, device):
    deltas = DeltasNN(args, device)
    return deltas
