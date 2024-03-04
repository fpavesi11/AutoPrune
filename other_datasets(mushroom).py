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


data = np.genfromtxt('data\\agaricus-lepiota.data.txt', delimiter=',', dtype='<U20')
labels = data[:,0]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:]

categorical_features = range(22)
feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(',')

categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
fibrous=f,grooves=g,scaly=y,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises=t,no=f
almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
attached=a,descending=d,free=f,notched=n
close=c,crowded=w,distant=d
broad=b,narrow=n
black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
enlarging=e,tapering=t
bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
fibrous=f,scaly=y,silky=k,smooth=s
fibrous=f,scaly=y,silky=k,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
partial=p,universal=u
brown=n,orange=o,white=w,yellow=y
none=n,one=o,two=t
cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')
for j, names in enumerate(categorical_names):
    values = names.split(',')
    values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
    data[:,j] = np.array(list(map(lambda x: values[x], data[:,j])))
    
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_
    
data = data.astype(float)

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80, random_state=42)

encoder = sklearn.preprocessing.OneHotEncoder()

encoder.fit(data)
encoded_train = encoder.transform(train).toarray()
encoded_test = encoder.transform(test).toarray()

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(encoded_train, labels_train)

predict_fn = lambda x: rf.predict_proba(encoder.transform(x))

print(sklearn.metrics.accuracy_score(labels_test, rf.predict(encoded_test)))

rf_preds = rf.predict(encoder.transform(test))

#%%

test_ex, rf_preds_ex = torch.tensor(encoded_test, dtype=torch.float32), torch.tensor(rf_preds, dtype=torch.float32)

pred_set_dnn = TrainDataset(test_ex, rf_preds_ex)

pred_loader_dnn = DataLoader(pred_set_dnn, batch_size=512, shuffle=True)


#%%

top_k = 8
input_size = test_ex.size(-1)

dropout_module = LossAwareFeatureDropout(p=1,
                                         topk=top_k,
                                         burn_in=300,
                                         threshold=0.005,
                                         convergence_window_length=30,
                                         log=True)


almost_ohnn_model = almostOneHotNNv2(input_size=input_size,
                                     output_size=1,
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
                                        max_depth=8,
                                        max_features=1,
                                        max_leaf_nodes=8,
                                        min_samples_leaf=30)

tree_surrogate.fit(test,
                   rf_preds_ex.numpy())

print(tree_surrogate.score(test,
                           rf_preds_ex.numpy()))

plt.figure(figsize=(12, 8))
plot_tree(tree_surrogate, filled=True, feature_names=feature_names, class_names=['0','1'], rounded=True)
plt.show()