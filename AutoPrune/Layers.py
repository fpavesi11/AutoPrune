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



class CustomDimensionalDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDimensionalDropout, self).__init__()
        self.p = p

    def forward(self, x, rules_weights):
        if self.training:
            # Create a mask with random values
            mask = torch.rand(1, 1, x.size(2)).to(x.device)
            # Add rule weight to increase dropout proba
            for param in rules_weights: #<-- needed for how it comes out
                proba = torch.abs(param) #<--- abs value needed
            proba = proba/torch.max(proba) #normalize in 0 1
            proba = 1 - proba
            proba = proba.unsqueeze(0).to(x.device)
            mask += proba
            mask = mask/2 #<--- normalize in 0 1
            # Apply dropout: set some dimensions to zero
            mask = (mask > self.p).float()
            # Expand the mask to match the dimensions of the input tensor
            mask = mask.expand_as(x)
            # Apply the mask
            output = x * mask
        else:
            # During evaluation, just return the input as is
            output = x
        return output


#########################################################################################################################à

class CustomFeatureDropout(nn.Module):
    def __init__(self, p=0.1, topk=8, burn_in=10, distress_period=20, log=True):
        super(CustomFeatureDropout, self).__init__()
        self.p = p
        self.topk = topk
        self.burn_in = burn_in
        self.all_masks = []
        self.distressing = 0
        self.distress_period = distress_period
        self.cut_times = 1
        self.log = log
        self.epoch = 0
        self.log_epoch = 0
        self.cut_drop_epoch = 0
        self.n_batches = 0
        self.observed_batches = 0
        self.final_config = False

    @staticmethod
    def create_mask(rule, topk):
        values, indices = torch.topk(torch.abs(rule).squeeze(), k=topk)
        mask = torch.zeros_like(rule).squeeze(0)
        mask[indices] = 1
        mask = mask.unsqueeze(0)
        return mask

    @staticmethod
    def sum_n_times(val, n):
        v = 0
        for i in range(n):
            v += val
        return v

    def update_distress(self, epoch, reset):
        if epoch > self.epoch:  # <--- avoids performing update more times on same epoch
            self.epoch = epoch
            print('From distress', self.epoch)
            if reset:
                self.distressing = self.distress_period
            else:
                self.distressing -= 1
        elif epoch == self.epoch:
            pass
        else:
            pass

    def update_cut_drop(self, epoch, w):
        if epoch != self.cut_drop_epoch:
            self.cut_drop_epoch = epoch
            print('Update cut drop', self.cut_drop_epoch)
            self.cut_times += 1
            self.drop_n = round(
                w[0].weight.data.size(-1) - self.sum_n_times(self.p, self.cut_times) * w[0].weight.data.size(-1))

    def return_log(self, epoch, string):
        if epoch != self.log_epoch:
            self.log_epoch = epoch
            print('Log epoch', self.log_epoch)
            if self.log:
                print(string)

    def forward(self, w, epoch=None):
        # print('debugging batches', self.observed_batches, 'epoch', epoch)
        if epoch == 0 and self.training:
            assert epoch is not None, 'During training, epoch is needed for burn in period, be sure you are passing epoch to the model and to this layer'
            self.n_batches += 1
            self.observed_batches = self.n_batches
            self.drop_n = round(
                w[0].weight.data.size(-1) - self.p * w[0].weight.data.size(-1))  # <-- set the initial top to keep
            for rule in w:
                self.all_masks.append(torch.ones_like(
                    rule.weight.data))  # <--- mask nothing before burn in period DO NOT REMOVE OR FIRST ITERATION IS BROKEN

        if self.training:
            assert epoch is not None, 'During training, epoch is needed for burn in period, be sure you are passing epoch to the model and to this layer'
            if epoch >= self.burn_in:
                mask_list = []
                if self.distressing == 0:
                    self.observed_batches -= 1
                    for j, rule in enumerate(w):
                        # self.return_log(epoch, 'Cutting ' + str(w[0].weight.data.size(-1) - self.drop_n) + '')
                        param = rule.weight.data
                        param = param * self.all_masks[j]
                        mask = self.create_mask(param, self.drop_n)
                        mask_list.append(mask)
                    self.all_masks = mask_list
                    if self.drop_n == self.topk:
                        self.drop_n = self.topk
                        if self.log:
                            self.return_log(epoch, 'Objective reached ' + str(self.drop_n))
                        self.distressing = -1  # <----- signals the cutting has to be stopped
                        self.final_config = True
                    if self.observed_batches == 0 and self.final_config == False:
                        print('Cutting ' + str(w[0].weight.data.size(-1) - self.drop_n) + ' DONE!')
                        self.observed_batches = self.n_batches
                        self.distressing = self.distress_period
                        self.cut_times += 1
                        self.drop_n = max(self.topk,
                                          round(
                                              w[0].weight.data.size(-1) - self.sum_n_times(self.p, self.cut_times) * w[
                                                  0].weight.data.size(-1)))


                elif self.distressing > 0:
                    self.observed_batches -= 1
                    mask_list = self.all_masks
                    if self.observed_batches == 0:
                        self.observed_batches = self.n_batches
                        self.distressing -= 1
                        print('Distressing', self.distressing)
                else:
                    mask_list = self.all_masks  # <------ when the number of weights reaches the desired quantity, cutting is stopped


            else:
                mask_list = []
                for rule in w:
                    mask_list.append(torch.ones_like(rule.weight.data))  # <--- mask nothing before burn in period

        else:
            mask_list = self.all_masks
        return mask_list