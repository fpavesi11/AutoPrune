from AutoPrune.AutoModels import LanguageAutoPruneMulticlass
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')


from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(data_home='data/',
                                      subset='train')
newsgroups_test = fetch_20newsgroups(data_home='data/',
                                     subset='test')

AutoPruneModel = LanguageAutoPruneMulticlass(n_clusters=64,
                                             n_rules=1,
                                             target_n_features=8)

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

results = []
for target in range(21):
    print('Target', target)
    results.append(AutoPruneModel(train_data_x=newsgroups_train.data,
                                  train_data_y=newsgroups_train.target,
                                  test_data_x=newsgroups_test.data,
                                  test_data_y=newsgroups_test.target,
                                  target_class = target,
                                  vectorizer_type=TfidfVectorizer,
                                  top_k_predictive=10000,
                                  top_k_surrogate=300,
                                  predictive_model=deepNN,
                                  p=1,
                                  burn_in=150,
                                  threshold=0.005,
                                  convergence_window_length=10,
                                  num_epochs_predictive=30,
                                  num_epochs_surrogate=5000,
                                  device=device)
                   )

#%%

import pickle

# Save to a file
with open('C:\\Users\\drikb\\Desktop\\CRISP\\XAI project\\AutoPrunePackage\\models\\results\\20news_results.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
import pickle
#Load from a file
with open('C:\\Users\\drikb\\Desktop\\CRISP\\XAI project\\AutoPrunePackage\\models\\results\\20news_results.pkl', 'rb') as file:
    loaded_list = pickle.load(file)

#%%

for element in loaded_list:
    print(element['surrogate_accuracy'])
