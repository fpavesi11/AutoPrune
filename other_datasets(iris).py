#%%
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
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

iris = sklearn.datasets.load_iris()
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80, random_state=42)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)
rf_preds = rf.predict(test)

print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))

#%%
import torch
from sklearn.preprocessing import KBinsDiscretizer
rf_preds_bin = torch.tensor(rf_preds == 2)

discretizer = KBinsDiscretizer(n_bins=4)
discretizer.fit(iris.data)
data_ex = discretizer.transform(iris.data).toarray()
train_ex, test_ex, _, _ = sklearn.model_selection.train_test_split(data_ex, iris.target, train_size=0.80, random_state=42)


test_ex, rf_preds_bin = torch.tensor(test_ex, dtype=torch.float32), torch.tensor(rf_preds_bin, dtype=torch.float32)

# build the dataset of predictions to perform explanation
pred_set_dnn = TrainDataset(test_ex, rf_preds_bin)

pred_loader_dnn = DataLoader(pred_set_dnn, batch_size=16, shuffle=True)

#%%

top_k = 4
input_size = test_ex.size(-1)

dropout_module = LossAwareFeatureDropout(p=1,
                                         topk=top_k,
                                         burn_in=300,
                                         threshold=0.005,
                                         convergence_window_length=30,
                                         log=True)


almost_ohnn_model = almostOneHotNNv2(input_size=input_size,
                                num_rules=4,
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
                    num_epochs=2000,
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


from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt


tree_surrogate = DecisionTreeClassifier(criterion='gini',
                                        max_depth=4,
                                        max_features=1,
                                        max_leaf_nodes=4)

tree_surrogate.fit(test,
                   rf_preds_bin.numpy())

print(tree_surrogate.score(test,
                           rf_preds_bin.numpy()))

plt.figure(figsize=(12, 8))
plot_tree(tree_surrogate, filled=True, feature_names=iris.target_names, class_names=['0','1'], rounded=True)
plt.show()