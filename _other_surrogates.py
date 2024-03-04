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

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt


tree_surrogate = DecisionTreeClassifier(criterion='gini',
                                        max_depth=7,
                                        max_features=1,
                                        max_leaf_nodes=64)

tree_surrogate.fit(test_encodings.numpy(),
                   predicted_dnn.detach().numpy())

print(tree_surrogate.score(test_encodings.numpy(),
                           predicted_dnn.detach().numpy()))

sorted_vectorizer = sorted_dict = dict(sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1]))

plt.figure(figsize=(12, 8))
plot_tree(tree_surrogate, filled=True, feature_names=list(sorted_vectorizer.keys()), class_names=['0','1'], rounded=True)
plt.show()


#%%

feature_importances = tree_surrogate.feature_importances_
used_features = feature_importances != 0
coverage = ((test_encodings.numpy() * used_features).sum(-1) > 0).sum()/len(test_encodings)

#%%

def extract_rules_from_tree(tree_surrogate):
    # Get the tree structure
    tree_structure = tree_surrogate.tree_

    # Get the number of features
    num_features = tree_structure.n_features

    # Initialize a list for each rule with the importance of each feature
    rule_arrays = []

    def traverse_tree(node_id, current_rule):
        if node_id == -1:
            return

        # Check if the node is a leaf
        if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
            rule_arrays.append(current_rule.copy())
            return

        # Add current split condition to the rule
        current_rule[tree_structure.feature[node_id]] = tree_structure.threshold[node_id]

        # Recursively traverse left and right children
        traverse_tree(tree_structure.children_left[node_id], current_rule)
        traverse_tree(tree_structure.children_right[node_id], current_rule)

        # Remove current split condition after backtracking
        current_rule[tree_structure.feature[node_id]] = 0

    # Start traversing the tree from the root
    traverse_tree(0, [0] * num_features)

    return rule_arrays

rule_arrays = extract_rules_from_tree(tree_surrogate)

#%%

rule_freq = []
n_instances = len(test_encodings.numpy())
for rule_array in rule_arrays:
    n_activated = (test_encodings.numpy() * rule_array != 0).sum(-1) != 0
    perc_activated = n_activated.sum()/n_instances
    rule_freq.append(perc_activated)

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rulefit import RuleFit

# Assuming 'X' is your feature matrix and 'y' is your target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RuleFit model
rulefit_model = RuleFit(max_iter=10000,
                        max_rules=8)

# Fit the model on the training data
rulefit_model.fit(test_encodings.numpy(), predicted_dnn.detach().numpy())

# Make predictions on the test data
y_pred = rulefit_model.predict(test_encodings.numpy())
#%%
# Evaluate accuracy
accuracy = accuracy_score(predicted_dnn.detach().numpy(), np.round(y_pred))
print("Accuracy:", accuracy)
#%%
# Get the rules generated by RuleFit
rules = rulefit_model.get_rules()
print("Rules:", rules)
