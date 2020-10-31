# copied or adapted from https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/metapath2vec-link-prediction.html

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec

from multiprocessing import cpu_count

CPU_COUNT = cpu_count()


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
	return [
		binary_operator(transform_node(src), transform_node(dst))
		for src, dst in link_examples
	]


# 2. training classifier
def link_prediction_classifier(max_iter=6000):
	lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
	return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


def train_link_prediction_model(
		link_examples, link_labels, get_embedding, binary_operator
):
	clf = link_prediction_classifier()
	link_features = link_examples_to_features(
		link_examples, get_embedding, binary_operator
	)
	clf.fit(link_features, link_labels)
	return clf


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
		clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
	link_features_test = link_examples_to_features(
		link_examples_test, get_embedding, binary_operator
	)
	score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
	return score


def evaluate_roc_auc(clf, link_features, link_labels):
	predicted = clf.predict_proba(link_features)

	# check which class corresponds to positive links
	positive_column = list(clf.classes_).index(1)
	return roc_auc_score(link_labels, predicted[:, positive_column])


def operator_l1(u, v):
	return np.abs(u - v)


def operator_l2(u, v):
	return (u - v) ** 2


# general utils
def make_walks(
		graph, metapaths, *,
		num_walks=10, walk_length=100):

	rw = UniformRandomMetaPathWalk(graph)
	return rw.run(
		graph.nodes(),
		n=num_walks,
		length=walk_length,
		metapaths=metapaths
	)


def metapath2vec_embedding(
		walks, *,
		context_window_size=10, num_iter=1, workers=CPU_COUNT, dimensions=64):
	model = Word2Vec(
		walks,
		size=dimensions,
		window=context_window_size,
		min_count=0,
		sg=1,
		workers=workers,
		iter=num_iter,
	)

	def get_embedding(u):
		return model.wv[u]

	return get_embedding
