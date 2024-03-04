from AutoPrune.Losses import *
from AutoPrune.Trainers import *
from AutoPrune.Utils import *
import torch
import numpy as np

"""
NOTES ON PERFORMANCES
Predictive model: 
    - MLP (Dense(top_k,512)LeakyRelU->Dense(512,128)LeakyReLU->Dense(128,64)LeakyReLU->Dense(64,1)Sigmoid
    - 30 epochs training with L2 regularization (lambda=0.3)
Surrogate model:
    - 8 clusters, each with 8 rules with objective 8 features (always reached)
    - p=1 (1 feature cut at time), epochs=5000, burn_in=150, window_length=10

- Class 0: fidelity around 95%. Predictive model has 83% test accuracy. 
- Class 1: fidelity around 97%. Predictive model 87% test accuracy.
- Class 2: fidelity around 95%. Predictive model has 87% test accuracy.
- Class 3: fidelity around 95%. Predictive model has 85% test accuracy.
- Class 4: fidelity around 94%. Predictive model has 84% test accuracy.
- Class 5: fidelity around 94%. Predictive model has 88% test accuracy.
- Class 6: fidelity around 93%. Predictive model has 85% test accuracy.
- Class 7: fidelity around 95%. Predictive model has 86% test accuracy.
- Class 8: fidelity around 95%. Predictive model has 86% test accuracy.
- Class 9: fidelity around 95%. Predictive model has 85% test accuracy.
- Class 10: fidelity around 96%. Predictive model has 84% test accuracy.
- Class 11: fidelity around 95%. Predictive model has 84% test accuracy.
- Class 12: fidelity around 94%. Predictive model has 84% test accuracy.
- Class 13: fidelity around 91%. Predictive model has 82% test accuracy.
- Class 14: fidelity around 92%. Predictive model has 86% test accuracy.
- Class 15: fidelity around 93%. Predictive model has 86% test accuracy.
- Class 16: fidelity around 96%. Predictive model has 85% test accuracy.
- Class 17: fidelity around 92%. Predictive model has 86% test accuracy.
- Class 18: fidelity around 97%. Predictive model has 87% test accuracy.
- Class 19: fidelity around 96%. Predictive model has 86% test accuracy.
- Class 20: fidelity around 95%. Predictive model has 86% test accuracy.
"""

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(data_home='data/',
                                      subset='train')
newsgroups_test = fetch_20newsgroups(data_home='data/',
                                     subset='test')


#%%

from AutoPrune.DataUtils import OnevsAllBalancedSample

target_class = 20
train_data, train_targets_original = OnevsAllBalancedSample(newsgroups_train.data, newsgroups_train.target, target_class=target_class)
test_data, test_targets_original = OnevsAllBalancedSample(newsgroups_test.data, newsgroups_test.target, target_class=target_class)

train_labels = [1 if t == target_class else 0 for t in train_targets_original]
test_labels = [1 if t == target_class else 0 for t in test_targets_original]




#%%

from sklearn.feature_extraction.text import TfidfVectorizer

top_k = 10000
vectorizer = TfidfVectorizer(lowercase=True, max_features=top_k)

# ONE VS. ALL TARGETS
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

#%%

from scipy.sparse import csr_matrix

# from sparse numpy to dense torch tensor
A = csr_matrix(train_vectors)
B = csr_matrix(test_vectors)
train_encodings = torch.tensor(A.todense(), dtype=torch.float32)
test_encodings = torch.tensor(B.todense(), dtype=torch.float32)

# BINARIZE ENCODINGS
train_encodings = (train_encodings > 0).to(torch.float32)
test_encodings = (test_encodings > 0).to(torch.float32)

train_set = TrainDataset(train_encodings, train_labels)
test_set = TrainDataset(test_encodings, test_labels)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=True)

#%%

class deepNN(nn.Module):
    def __init__(self, in_features):
        super(deepNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, 512),
                                   nn.LeakyReLU(),
                                   nn.Linear(512, 128),
                                   nn.LeakyReLU(),
                                   nn.Linear(128, 64),
                                   nn.LeakyReLU(),
                                   nn.Linear(64, 1),
                                   nn.Sigmoid())
        self.num_rules = 0  # <--- this allows to use model trainer, must be fixed

    def forward(self, x, unused=None, unused2=None):
        x = self.model(x)
        return x

    def get_hidden_rules_params(self):
        return self.parameters()

    def get_rules_weights_params(self):
        return None


dnn = deepNN(in_features=top_k)

loss_fn = AlmostPenalizedLoss(loss_fn=nn.BCELoss(reduction='sum'),
                              l1_lambda_hidden_rules=0.0,
                              l2_lambda_hidden_rules=0.3,
                              l1_lambda_rules_weights=0.0,
                              l2_lambda_rules_weights=0.0,
                              gini_lambda_rules_weights=0.0)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')

trainer = RegularModelTrainer(model=dnn,
                              loss_fn=loss_fn,
                              optimizer=AdamW)

trainer.train_model(train_data_loader=train_loader,
                    num_epochs=30,
                    device=device,
                    learning_rate=1e-3,
                    display_log='epoch',
                    store_weights_history=False,
                    store_weights_grad=False)


loss_dnn, accuracy_dnn, f1_0_dnn, f1_1_dnn = trainer.eval_model(test_loader, device=device)
print(f'Average loss: {loss_dnn}')
print(f'Accuracy: {accuracy_dnn}')
print(f'F1 score on label 0: {f1_0_dnn}')
print(f'F1 score on label 1: {f1_1_dnn}')

predicted_dnn = trainer.predict(test_loader, device=device, prob=False)

# build the dataset of predictions to perform explanation
pred_set_dnn = TrainDataset(test_encodings,
                               torch.tensor(predicted_dnn.detach().numpy(), dtype=torch.float32))

pred_loader_dnn = DataLoader(pred_set_dnn, batch_size=512, shuffle=True)


#%%

from AutoPrune.DataUtils import SentenceTransformerKMeansClustering
from sklearn.feature_extraction.text import TfidfVectorizer

n_clusters = 8
top_k = 300
st_kmean = SentenceTransformerKMeansClustering(n_clusters=n_clusters,
                                               top_k=top_k,
                                               vectorizer=TfidfVectorizer,
                                               sentence_transformer='all-MiniLM-L6-v2',
                                               lowercase=True)

cluster_test_encodings = st_kmean.fit(test_data)
translation_vocabs = st_kmean.get_translation_vocabs() #returns vocabularies to translate back to words

# build the dataset of predictions to perform explanation
pred_set_dnn = TrainDataset(cluster_test_encodings,
                            torch.tensor(predicted_dnn.detach().numpy(), dtype=torch.float32))

pred_loader_dnn = DataLoader(pred_set_dnn, batch_size=512, shuffle=True)


#%%

dropout_module = LossAwareFeatureDropout(p=1,
                                         topk=8,
                                         burn_in=150,
                                         threshold=0.005,
                                         convergence_window_length=10,
                                         log=True)

almost_ohnn_model = ClusterOneHotNNv2(input_size=top_k,
                                    n_clusters=n_clusters,
                                   num_rules=8,
                                   dropout_module=dropout_module,
                                   burn_in=300,
                                   initial_weight_val=None, # useless, to be removed
                                   final_activation=nn.Sigmoid,
                                   out_weight_constraint=None,
                                   rule_bias=False,
                                   force_positive_out_init=False,
                                   rule_activation=nn.Tanh,
                                   out_bias=False,
                                   dropout_perc=0.05,
                                   dtype=torch.float32)

loss_fn = AlmostPenalizedLoss(loss_fn=nn.BCELoss(reduction='sum'),
                              l1_lambda_hidden_rules=0.0,
                              l2_lambda_hidden_rules=0.0,
                              l1_lambda_rules_weights=0.0,
                              l2_lambda_rules_weights=0.0,
                              gini_lambda_rules_weights=0.0,
                              gini_lambda_hidden_rules=0.0)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')

trainer = RegularModelTrainer(model=almost_ohnn_model,
                             loss_fn=loss_fn,
                             optimizer=AdamW)

trainer.train_model(train_data_loader=pred_loader_dnn,
                    num_epochs=5000,
                    device=device,
                    learning_rate=1e-3,
                    display_log='epoch',
                    store_weights_history=False,
                    store_weights_grad=False)


fig1 = sns.lineplot(trainer.history['train_loss'])
plt.show()

fig2 = sns.lineplot(trainer.history['train_accuracy'])
plt.show()

#%%

avg_loss, accuracy, f1_1, f1_0 = trainer.eval_model(pred_loader_dnn, device=device)

#%%

from AutoPrune.ExplanationEvaluation import RuleFrequencyCluster

rfreq = RuleFrequencyCluster(trainer.model.model, pred_loader_dnn, device=device)
print(rfreq)

#%%

from AutoPrune.ExplanationEvaluation import CoverageCluster

covclust = CoverageCluster(trainer.model.model, pred_loader_dnn, device=device)
print(covclust)

#%%

from AutoPrune.ExplanationEvaluation import OverlappingCluster

OverlappingCluster(trainer.model.model)



#%%

from AutoPrune.ExplanationEvaluation import OverallFeatureOverlapping

OverallFeatureOverlapping(trainer.model.model, translation_vocabs)


