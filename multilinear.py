import math
import string

import torch
from torch import nn
from torch.nn import init


class Multilinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = tuple(in_features)
        if len(self.in_features) > 24:
            raise ValueError('Up to 24 input vectors supported')
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, *in_features))
        chars = string.ascii_lowercase
        n = len(self.in_features)
        self.einsum_str = '{}{},z{}->z{}'.format(
            chars[n], chars[:n], ',z'.join(chars[:n]), chars[n]
        )
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
        out = torch.einsum(self.einsum_str, self.weight, *inputs)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
