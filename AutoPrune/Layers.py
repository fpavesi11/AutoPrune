from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW, RMSprop
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import init
import math
import seaborn as sns
import matplotlib.pyplot as plt

########################################################################################################################
class OneActLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, weight_val: float, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight_val = weight_val
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor, weight_mask) -> torch.Tensor:
        if self.bias is None:
            out = torch.matmul(input, self.weight.T * weight_mask.T)
        else:
            out = torch.matmul(input, self.weight.T * weight_mask.T) + self.bias
        return out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    """def reset_parameters(self) -> None:
        init.constant_(self.weight, self.weight_val)"""
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

########################################################################################################################
"""
WEIGHT CONSTRAINED LINEAR V2
"""
class weightConstrainedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, neuron_activation=None, weight_function=None,
                 force_positive_init=False, dtype=torch.float64):
        super(weightConstrainedLinear, self).__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.weight_function = weight_function
        self.dtype = dtype
        self.force_positive_init = force_positive_init
        if force_positive_init:  # forces positive weights initialization
            with torch.no_grad():
                self.weight.copy_(torch.abs(self.weight))
        self.neuron_activation = None
        if neuron_activation is not None:
            self.neuron_activation = neuron_activation()

    def forward(self, input):
        assert input.dtype == self.dtype, ("Expected input dtype", self.dtype, "but got", input.dtype)
        if self.weight_function is not None:
            self.weight.data = self.weight_function(self.weight.data)
        out = nn.functional.linear(input, self.weight, self.bias)
        if self.neuron_activation is not None:
            out = self.neuron_activation(out)
        return out


########################################################################################################################
"""
RULE'S WEIGHT DENSE

Defines the rule's weight dense network

NOTE: it's very similar to hidden dense but the last line of forward, 
for clarity is kept separate but merging is quite easy
"""

class ruleDense(nn.Module):
    def __init__(self, input_size, num_neurons, weight_constraint=None, bias=False, force_positive_init=True, dtype=torch.float64):
        super(ruleDense, self).__init__()
        self.input_size = input_size
        if weight_constraint is not None:
            self.weight_constraint = weight_constraint() #remember not to call it when passing arguments
        else:
            self.weight_constraint = None
        self.force_positive = force_positive_init
        self.linear = weightConstrainedLinear(input_size, num_neurons, weight_function=self.weight_constraint,
                                              bias=bias, force_positive_init=force_positive_init, dtype=dtype)

    def forward(self, x):
        x = self.linear(x)
        return x

########################################################################################################################





