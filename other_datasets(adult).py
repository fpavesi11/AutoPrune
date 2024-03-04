from AutoPrune.AutoModels import LanguageAutoPruneMulticlass
import torch
from torch import nn
import pandas as pd
import numpy as np
from AutoPrune.Losses import *
from AutoPrune.Trainers import *
from AutoPrune.Utils import *
import torch
import numpy as np
from sklearn.model_selection import train_test_split


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')

from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
adult_features = adult.data.features
adult_targets = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

df = pd.concat([adult_features, adult_targets], axis=1)
df.replace('?', np.nan, inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)
df.income = df.income.map(lambda x: x.replace('.',''))

#%%
# PREDICTIVE DATASET
# One-hot-encode
pr_df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation',
                                'relationship', 'race', 'native-country'])

# Binarize
pr_df = pd.get_dummies(pr_df, columns=['sex', 'income'], drop_first=True)


# to drop (fnlwgt, education-num)
pr_df.drop(['education-num', 'fnlwgt'], axis=1, inplace=True)

X = pr_df.drop('income_>50K', axis=1)
y = pr_df['income_>50K']

random_state = 42

# Determine the desired balance ratio
balance_ratio = 0.5

# Calculate the number of observations needed for each class
n_class_min = min(sum(y==0), sum(y==1))

# Sample from each class to achieve the desired balance
X_balanced = pd.concat([
    X[y == 0].sample(n=n_class_min, random_state=random_state),
    X[y == 1].sample(n=n_class_min, random_state=random_state)
], axis=0)

y_balanced = pd.concat([
    y[y == 0].sample(n=n_class_min, random_state=random_state),
    y[y == 1].sample(n=n_class_min, random_state=random_state)
], axis=0)

# Split the balanced dataset into training and testing sets with stratification over both X_balanced and y_balanced
train_x, test_x, train_y, test_y = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=random_state, stratify=y_balanced)

train_x, test_x, train_y, test_y = train_x.to_numpy(dtype=np.float32), test_x.to_numpy(dtype=np.float32), train_y.to_numpy(dtype=np.float32), test_y.to_numpy(dtype=np.float32)

#%%

train_x, test_x, train_y, test_y = torch.tensor(train_x), torch.tensor(test_x), torch.tensor(train_y), torch.tensor(test_y)
train_set = TrainDataset(train_x, train_y)
test_set = TrainDataset(test_x, test_y)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=True)


#%%

class deepNN(nn.Module):
    def __init__(self, in_features):
        super(deepNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, 1028),
                                   nn.LeakyReLU(),
                                   nn.Linear(1028, 512),
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

in_features = train_x.size(-1)
dnn = deepNN(in_features=in_features)

loss_fn = AlmostPenalizedLoss(loss_fn=nn.BCELoss(reduction='sum'),
                              l1_lambda_hidden_rules=0.1,
                              l2_lambda_hidden_rules=0.1,
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
                    num_epochs=100,
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


#%%
# EXPLAINER DATASET
# One-hot-encode
ex_df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation',
                                'relationship', 'race', 'native-country'])

# Binarize
ex_df = pd.get_dummies(ex_df, columns=['sex', 'income'], drop_first=True)

# Divide in classes and one hot encode
ex_df['capital-gain'] = pd.qcut(ex_df['capital-gain'], q=10, labels=False, duplicates='drop')
ex_df['capital-loss'] = pd.qcut(ex_df['capital-loss'], q=10, labels=False, duplicates='drop')
ex_df['hours-per-week'] = pd.qcut(ex_df['hours-per-week'], q=10, labels=False, duplicates='drop')
ex_df['age'] = pd.qcut(ex_df['age'], q=10, labels=False, duplicates='drop')

ex_df = pd.get_dummies(ex_df, columns=['capital-gain', 'capital-loss', 'hours-per-week', 'age'])

# to drop (fnlwgt, education-num)
ex_df.drop(['education-num', 'fnlwgt'], axis=1, inplace=True)

X = ex_df.drop('income_>50K', axis=1)
y = ex_df['income_>50K']

random_state = 42

# Determine the desired balance ratio
balance_ratio = 0.5

# Calculate the number of observations needed for each class
n_class_min = min(sum(y==0), sum(y==1))

# Sample from each class to achieve the desired balance
X_balanced_ex = pd.concat([
    X[y == 0].sample(n=n_class_min, random_state=random_state),
    X[y == 1].sample(n=n_class_min, random_state=random_state)
], axis=0)

y_balanced_ex = pd.concat([
    y[y == 0].sample(n=n_class_min, random_state=random_state),
    y[y == 1].sample(n=n_class_min, random_state=random_state)
], axis=0)

# Split the balanced dataset into training and testing sets with stratification over both X_balanced and y_balanced
train_x_ex, test_x_ex, train_y_ex, test_y_ex = train_test_split(X_balanced_ex, y_balanced_ex, test_size=0.2, random_state=random_state, stratify=y_balanced_ex)

train_x_ex, test_x_ex, train_y_ex, test_y_ex = train_x_ex.to_numpy(dtype=np.float32), test_x_ex.to_numpy(dtype=np.float32), train_y_ex.to_numpy(dtype=np.float32), test_y_ex.to_numpy(dtype=np.float32)

train_x_ex, test_x_ex, train_y_ex, test_y_ex = torch.tensor(train_x_ex), torch.tensor(test_x_ex), torch.tensor(train_y_ex), torch.tensor(test_y_ex)

#%%

# TREE 
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt


tree_surrogate = DecisionTreeClassifier(criterion='gini',
                                        max_depth=8,
                                        max_features=1,
                                        max_leaf_nodes=64)

tree_surrogate.fit(test_x,
                   predicted_dnn.numpy())

print(tree_surrogate.score(test_x,
                           predicted_dnn.numpy()))

plt.figure(figsize=(12, 8))
plot_tree(tree_surrogate, filled=True, class_names=['0','1'], rounded=True)
plt.show()


#%%
# K-MEANS CLUSTERING
from sklearn.cluster import KMeans

n_clusters = 8
KMean = KMeans(n_clusters=4)
KMean.fit(test_x_ex)
obs_clusters = torch.tensor(KMean.labels_).unsqueeze(-1)
test_x_ex = torch.cat([test_x_ex, obs_clusters], dim=-1)


#%%

# build the dataset of predictions to perform explanation
pred_set_dnn = TrainDataset(test_x_ex,
                               torch.tensor(predicted_dnn.detach().numpy(), dtype=torch.float32))

pred_loader_dnn = DataLoader(pred_set_dnn, batch_size=512, shuffle=True)


#%%

top_k = 8
input_size = test_x_ex.size(-1)-1

dropout_module = LossAwareFeatureDropout(p=1,
                                         topk=top_k,
                                         burn_in=400,
                                         threshold=0.01,
                                         convergence_window_length=10,
                                         log=True)


almost_ohnn_model = ClusterOneHotNNv2(input_size=input_size,
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

# TREE 
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

selected_cluster = 0
reduced_x = test_x_ex[test_x_ex[:,-1] == selected_cluster]
predicted_restricted = predicted_dnn[test_x_ex[:,-1] == selected_cluster]


tree_surrogate = DecisionTreeClassifier(criterion='gini',
                                        max_depth = 8,
                                        max_leaf_nodes = 8)

tree_surrogate.fit(reduced_x.numpy(),
                   predicted_restricted.numpy())

print(tree_surrogate.score(reduced_x.numpy(),
                           predicted_restricted.numpy()))

"""plt.figure(figsize=(12, 8))
plot_tree(tree_surrogate, filled=True, class_names=['0','1'], rounded=True)
plt.show()"""

#%%


from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

print(sum(tree_surrogate.tree_.children_left < 0))
# start pruning from the root
prune_index(tree_surrogate.tree_, 0, 10)
print(sum(tree_surrogate.tree_.children_left < 0))

print(tree_surrogate.tree_.max_depth)

print(tree_surrogate.score(test_x_ex.numpy(),
                           predicted_dnn.numpy()))
#%%

plt.figure(figsize=(12, 8))
plot_tree(tree_surrogate, filled=True, class_names=['0','1'], rounded=True)
plt.show()