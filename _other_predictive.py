from AutoPrune.Losses import *
from AutoPrune.Trainers import *
from AutoPrune.Utils import *
import torch
import numpy as np
from torch import nn

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(data_home='data/',
                                      subset='train')
newsgroups_test = fetch_20newsgroups(data_home='data/',
                                     subset='test')

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from AutoPrune.DataUtils import OnevsAllBalancedSample
from scipy.sparse import csr_matrix


target_class = 20
train_data, train_targets_original = OnevsAllBalancedSample(newsgroups_train.data, newsgroups_train.target, target_class=target_class)
test_data, test_targets_original = OnevsAllBalancedSample(newsgroups_test.data, newsgroups_test.target, target_class=target_class)

train_labels = [1 if t == target_class else 0 for t in train_targets_original]
test_labels = [1 if t == target_class else 0 for t in test_targets_original]


top_k = 10000
vectorizer = TfidfVectorizer(lowercase=True, max_features=top_k)

# ONE VS. ALL TARGETS
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)


# from sparse numpy to dense torch tensor
A = csr_matrix(train_vectors)
B = csr_matrix(test_vectors)
train_encodings = torch.tensor(A.todense(), dtype=torch.float32)
test_encodings = torch.tensor(B.todense(), dtype=torch.float32)

# BINARIZE ENCODINGS
train_encodings = (train_encodings > 0).to(torch.float32)
test_encodings = (test_encodings > 0).to(torch.float32)

#%%

# SUPPORT VECTOR MACHINE - RADIAL BASIS FUNCTIONS KERNEL

from sklearn import svm

supp_vm = svm.SVC(kernel='rbf')

supp_vm.fit(train_encodings.numpy(), np.array(train_labels, dtype=np.float32))


acc_svm = supp_vm.score(test_encodings.numpy(),
                        np.array(test_labels, dtype=np.float32))
print(acc_svm)

predicted_svm = supp_vm.predict(test_encodings.numpy())

#%%

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500,
                             criterion='gini')

rfc.fit(train_encodings.numpy(),
        np.array(train_labels, dtype=np.float32))

acc_rf = rfc.score(test_encodings.numpy(),
                   np.array(test_labels, dtype=np.float32))

print(acc_rf)

predicted_rf = rfc.predict(test_encodings.numpy())


