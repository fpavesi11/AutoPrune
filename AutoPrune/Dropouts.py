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

class RegularFeatureDropout(nn.Module):
    def __init__(self, p=0.1, topk=8, burn_in=10, distress_period=20, log=True):
        super(RegularFeatureDropout, self).__init__()
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

    def forward(self, w, epoch=None, loss=None):
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
                        print('Epoch ' + str(epoch) + ': Cutting ' + str(w[0].weight.data.size(-1) - self.drop_n) + ' DONE!')
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
                else:
                    mask_list = self.all_masks  # <------ when the number of weights reaches the desired quantity, cutting is stopped


            else:
                mask_list = []
                for rule in w:
                    mask_list.append(torch.ones_like(rule.weight.data))  # <--- mask nothing before burn in period

        else:
            mask_list = self.all_masks
        return mask_list


########################################################################################################################


class LossAwareFeatureDropout(nn.Module):
    def __init__(self, p=0.1, topk=8, burn_in=10, threshold=None, convergence_window_length=100, log=False):
        super(LossAwareFeatureDropout, self).__init__()
        self.p = p
        self.topk = topk
        self.burn_in = burn_in
        self.all_masks = []
        self.distressing = 0
        self.threshold = threshold
        self.cut_times = 1
        self.log = log
        self.epoch = 0
        self.log_epoch = 0
        self.cut_drop_epoch = 0
        self.n_batches = 0
        self.observed_batches = 0
        self.final_config = False
        self.first_cut = True
        self.count_cuts = 0
        self.convergence_window_length = convergence_window_length
        self.convergence_window_true = []
        self.convergence_window_check = []

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

    def window_slider(self, loss_list, loss):
        loss_list.append(loss)
        if len(loss_list) > self.convergence_window_length:
            loss_list = loss_list[-self.convergence_window_length:]  # <-- this way window is always same length
        return loss_list

    @staticmethod
    def delta(x):
        deltas = []
        for i in range(len(x)):
            if i - 1 >= 0:
                deltas.append(x[i] - x[i - 1])
        return deltas

    def check_cut_eligibility(self): # This function can be too conservative when burn in period is long
        cut_eligibility = False
        if len(self.convergence_window_check) >= self.convergence_window_length:
            if self.threshold is None: # Uses burn in convergence
                delta_true = torch.tensor(self.delta(self.convergence_window_true))
                abs_max_delta_true = torch.max(torch.abs(delta_true))
                abs_avg_delta_check = (torch.sum(torch.abs(torch.tensor(self.delta(self.convergence_window_check)))) / len(
                    self.convergence_window_check)).item()
                if abs_avg_delta_check <= abs_max_delta_true:
                    cut_eligibility = True
            else: # Uses predefined threshold
                abs_avg_delta_check = (
                            torch.sum(torch.abs(torch.tensor(self.delta(self.convergence_window_check)))) / len(
                        self.convergence_window_check)).item()
                if abs_avg_delta_check <= self.threshold:
                    cut_eligibility = True
        return cut_eligibility

    def __calculate_drop_n(self, w):
        if isinstance(self.p, int):
            drop_n = self.p
        else:
            drop_n = round()

    def forward(self, w, epoch=None, loss=None):
        # print('debugging batches', self.observed_batches, 'epoch', epoch)
        if epoch == 0 and self.training:
            assert epoch is not None, 'During training, epoch is needed for burn in period, be sure you are passing epoch to the model and to this layer'
            self.n_batches += 1
            self.observed_batches = self.n_batches
            # set initial top to keep
            if isinstance(self.p, int):
                self.drop_n = round(w[0].weight.data.size(-1) - self.p)
            else:
                self.drop_n = round(w[0].weight.data.size(-1) - self.p * w[0].weight.data.size(-1))

            for rule in w:
                self.all_masks.append(torch.ones_like(
                    rule.weight.data))  # <--- mask nothing before burn in period DO NOT REMOVE OR FIRST ITERATION IS BROKEN

        if self.training:
            assert epoch is not None, 'During training, epoch is needed for burn in period, be sure you are passing epoch to the model and to this layer'
            if epoch >= self.burn_in:
                mask_list = []
                self.convergence_window_check = self.window_slider(self.convergence_window_check, loss)
                cut_eligibility_check = self.check_cut_eligibility()
                if (cut_eligibility_check and not self.final_config) or self.first_cut:
                    self.first_cut=False
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
                            self.return_log(epoch,
                                            'Objective ' + str(self.topk) + ' features reached ' + str(self.drop_n))
                        self.final_config = True
                    if self.final_config == False:
                        if self.log:
                            print('Epoch ' + str(epoch) + ': Cutting ' + str(w[0].weight.data.size(-1) - self.drop_n) + ' DONE!')
                        self.cut_times += 1
                        if isinstance(self.p, int):
                            drop_n = round(w[0].weight.data.size(-1) - self.p*self.cut_times)
                        else:
                            drop_n = round(w[0].weight.data.size(-1) - self.sum_n_times(self.p, self.cut_times) * w[0].weight.data.size(-1))
                        self.drop_n = max(self.topk,
                                          drop_n)
                    self.convergence_window_check = []  # <--- window is resetted, otherwise it considers values before cut

                else:
                    mask_list = self.all_masks

            else:
                mask_list = []
                for rule in w:
                    mask_list.append(torch.ones_like(rule.weight.data))  # <--- mask nothing before burn in period
                    if loss is not None:
                        self.convergence_window_true = self.window_slider(self.convergence_window_true, loss)

        else: # Evaluation
            mask_list = self.all_masks
        return mask_list



########################################################################################################################


"""
TARGETED DROPOUT
"""


class TargetFeatureDropout(nn.Module):
    def __init__(self, p=0.5, threshold=0.7, burn_in=10):
        super(TargetFeatureDropout, self).__init__()
        self.p = p
        self.threshold = threshold
        self.burn_in = burn_in
        self.drop_history = None

    def forward(self, w, epoch=None, loss=None):
        if self.drop_history is None:
            self.drop_history = []
            for rule in w:
                self.drop_history.append(torch.ones_like(rule.weight.data))
        if self.training:
            assert epoch is not None, 'During training, epoch is needed for burn in period, be sure you are passing epoch to the model and to this layer'
            if epoch >= self.burn_in:
                mask_list = []
                for n_rule, rule in enumerate(w):
                    param = rule.weight.data
                    mask = torch.rand(1, param.size(1)).to(param.device)
                    # mask = torch.ones((1, param.size(1))).to(param.device)
                    proba = torch.abs(param).to(param.device)
                    proba = proba / torch.max(proba)  # normalize in 0 1
                    mask += proba
                    mask = mask / 2  # <--- normalize in 0 1
                    # Apply dropout: set some dimensions to zero
                    mask = (mask > self.p).float()
                    # Expand the mask to match the dimensions of the input tensor
                    mask = mask.expand_as(proba)
                    # print(proba.sort(descending=True))
                    # rule.weight.data = param * mask

                    self.drop_history[n_rule] += mask
                    self.drop_history[n_rule] /= 2

                    mask_list.append(mask)
            else:
                mask_list = []
                for rule in w:
                    mask_list.append(torch.ones_like(rule.weight.data))  # <--- mask nothing before burn in period

        else:
            mask_list = []
            for drop_mask in self.drop_history:
                mask = drop_mask >= self.threshold
                # rule = rule.weight.data * mask
                mask_list.append(mask)
        return mask_list