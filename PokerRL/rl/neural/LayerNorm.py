import torch

from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
