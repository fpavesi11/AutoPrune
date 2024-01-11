from itertools import compress
from sklearn.utils import shuffle
import numpy as np

def OnevsAllBalancedSample(data, targets, target_class, n_classes=None, seed=None):
    if n_classes is None:
        n_classes = len(np.unique(targets))

    train_data_1 = list(compress(data, list(targets == 1)))
    train_targets_1 = [target_class] * len(train_data_1)

    train_data_2 = []
    train_targets_2 = []
    len_sample = round(len(train_data_1) / (n_classes - 1))
    for j in range(n_classes):
        if j == target_class:
            continue
        train_data_j = list(compress(data, list(targets == j)))
        if len(train_data_j) < len_sample:
            sample_index = [True] * len(train_data_j)
            raise RuntimeWarning('Class {n_class} has only {n_obs} observations'.format(n_class=j, n_obs=len(train_data_j)))
        else:
            sample_index = [True] * len_sample
            sample_index.extend([False] * (len(train_data_j) - len_sample))
            sample_index = shuffle(sample_index, random_state=seed)

        sample_data = list(compress(train_data_j, sample_index))
        train_targets_j = [j] * len(sample_data)
        train_data_2.extend(sample_data)
        train_targets_2.extend(train_targets_j)

    train_data = train_data_1 + train_data_2
    train_targets = train_targets_1 + train_targets_2

    train_data, train_targets = shuffle(train_data, train_targets, random_state=seed)

    return train_data, train_targets

