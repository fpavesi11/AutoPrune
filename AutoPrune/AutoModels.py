from AutoPrune.Losses import *
from AutoPrune.Trainers import *
from AutoPrune.Utils import *
import torch
import numpy as np
from AutoPrune.DataUtils import OnevsAllBalancedSample
from scipy.sparse import csr_matrix
from AutoPrune.DataUtils import SentenceTransformerKMeansClustering


class LanguageAutoPruneMulticlass:
    def __init__(self, n_clusters, n_rules, target_n_features):
        self.n_clusters = n_clusters
        self.n_rules = n_rules
        self.target_n_features = target_n_features

    def __call__(self, train_data_x, train_data_y, test_data_x, test_data_y, target_class, vectorizer_type,
                 top_k_predictive, top_k_surrogate, predictive_model, p=1, burn_in=150, threshold=0.005,
                 convergence_window_length=10, num_epochs_predictive=30, num_epochs_surrogate=5000, device=None):
        train_data, train_targets_original = OnevsAllBalancedSample(train_data_x, train_data_y, target_class=target_class)
        test_data, test_targets_original = OnevsAllBalancedSample(test_data_x, test_data_y, target_class=target_class)

        train_labels = [1 if t == target_class else 0 for t in train_targets_original]
        test_labels = [1 if t == target_class else 0 for t in test_targets_original]

        vectorizer = vectorizer_type(lowercase=True, max_features=top_k_predictive)

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

        train_set = TrainDataset(train_encodings, train_labels)
        test_set = TrainDataset(test_encodings, test_labels)

        train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=512, shuffle=True)

        # FIT PREDICTIVE
        dnn = predictive_model(in_features=top_k_predictive)

        loss_fn = AlmostPenalizedLoss(loss_fn=nn.BCELoss(reduction='sum'),
                                      l1_lambda_hidden_rules=0.0,
                                      l2_lambda_hidden_rules=0.3,
                                      l1_lambda_rules_weights=0.0,
                                      l2_lambda_rules_weights=0.0,
                                      gini_lambda_rules_weights=0.0)

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
        predicted_dnn = trainer.predict(test_loader, device=device, prob=False)


        #K means Sentence Transformer clustering
        st_kmean = SentenceTransformerKMeansClustering(n_clusters=self.n_clusters,
                                                       top_k=top_k_surrogate,
                                                       vectorizer=vectorizer_type,
                                                       sentence_transformer='all-MiniLM-L6-v2',
                                                       lowercase=True)

        cluster_test_encodings = st_kmean.fit(test_data)
        translation_vocabs = st_kmean.get_translation_vocabs()  # returns vocabularies to translate back to words

        # build the dataset of predictions to perform explanation
        pred_set_dnn = TrainDataset(cluster_test_encodings,
                                    torch.tensor(predicted_dnn.detach().numpy(), dtype=torch.float32))

        pred_loader_dnn = DataLoader(pred_set_dnn, batch_size=512, shuffle=True)


        # FIT SURROGATE

        dropout_module = LossAwareFeatureDropout(p=p,
                                                 topk=self.target_n_features,
                                                 burn_in=burn_in,
                                                 threshold=threshold,
                                                 convergence_window_length=convergence_window_length,
                                                 log=False)

        almost_ohnn_model = ClusterOneHotNNv2(input_size=top_k_surrogate,
                                              n_clusters=self.n_clusters,
                                              num_rules=self.n_rules,
                                              dropout_module=dropout_module,
                                              burn_in=300, #<--- useless, to be removed
                                              initial_weight_val=None, #<--- useless, to be removed
                                              final_activation=nn.Sigmoid,
                                              out_weight_constraint=None, #<--- of no interest, to be removed
                                              rule_bias=False,
                                              force_positive_out_init=False, #<--- of no interest, to be removed
                                              rule_activation=nn.Tanh,
                                              out_bias=False,
                                              dropout_perc=0.05, #<---- useless, to be removed
                                              dtype=torch.float32)

        loss_fn = AlmostPenalizedLoss(loss_fn=nn.BCELoss(reduction='sum'),
                                      l1_lambda_hidden_rules=0.0,
                                      l2_lambda_hidden_rules=0.0,
                                      l1_lambda_rules_weights=0.0,
                                      l2_lambda_rules_weights=0.0,
                                      gini_lambda_rules_weights=0.0,
                                      gini_lambda_hidden_rules=0.0)

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

        convergence_reached = False
        # it is enough to check on the first cluster, for the first rule
        first_cluster_masks = almost_ohnn_model.model[0][-1].all_masks
        num_neurons_conv = first_cluster_masks[0].sum(dim=-1)
        if num_neurons_conv == self.target_n_features:
            convergence_reached = True

        return_dict = {'predictive_loss': loss_dnn,
                       'predictive_accuracy': accuracy_dnn,
                       'predictive_f1_0': f1_0_dnn,
                       'predictive_f1_1': f1_1_dnn,
                       'surrogate_loss_history': trainer.history['train_loss'],
                       'surrogate_accuracy_history': trainer.history['train_accuracy'],
                       'convergence_reached': convergence_reached}

        return return_dict


