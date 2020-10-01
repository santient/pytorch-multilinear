import math

import torch
from torch import nn
from torch.nn import init


def multilinear(inputs, weight, bias=None):
    assert weight.dim() == len(inputs) + 1
    out = weight
    for input in reversed(inputs):
        out = out @ input
    if bias is not None:
        out = out + bias
    return out

class Multilinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = tuple(in_features)
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, *in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(max(self.in_features))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, *inputs):
        return multilinear(inputs, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
