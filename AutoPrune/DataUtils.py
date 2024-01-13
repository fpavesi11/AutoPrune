from itertools import compress
from sklearn.utils import shuffle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import torch
from scipy.sparse import csr_matrix



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


class SentenceTransformerKMeansClustering:
    def __init__(self, n_clusters, top_k, vectorizer, sentence_transformer='all-MiniLM-L6-v2', lowercase=True):
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.lowercase = lowercase
        self.vectorizer = vectorizer
        self.transformer = SentenceTransformer(sentence_transformer)
        self.all_vectorizers = []

    def get_translation_vocabs(self):
        translation_vocabs = []
        for vectorizer in self.all_vectorizers:
            vocab = {}
            for k, v in vectorizer.vocabulary_.items():
                vocab[v] = k
            translation_vocabs.append(vocab)
        return translation_vocabs

    def fit(self, data, device=None, log=False):
        self.transformer = self.transformer.to(device)
        embeddings = torch.tensor(self.transformer.encode(data)).to(device)
        Kmean = KMeans(n_clusters=self.n_clusters)
        Kmean.fit(embeddings.cpu())

        sentences_clusters = torch.tensor(Kmean.labels_).unsqueeze(-1)

        test_vectors = torch.zeros((len(sentences_clusters), self.top_k))
        for cluster in range(self.n_clusters):
            index = sentences_clusters == cluster
            vectorizer = self.vectorizer(lowercase=self.lowercase, max_features=self.top_k)
            sentence_slice = [data[i] for i, x in enumerate(index) if x]
            slice_vectors = vectorizer.fit_transform(sentence_slice)
            self.all_vectorizers.append(vectorizer)
            B = csr_matrix(slice_vectors)
            slice_encodings = torch.tensor(B.todense(), dtype=torch.float32)
            slice_encodings = (slice_encodings > 0).to(torch.float32)
            if log:
                print(slice_encodings.size())
            index_positions = [i for i, x in enumerate(index) if x]
            for j, idx in enumerate(index_positions):
                test_vectors[idx, :] = slice_encodings[j, :]

        cluster_test_encodings = torch.cat([test_vectors, sentences_clusters], dim=-1)

        return cluster_test_encodings





