import random
from collections import namedtuple

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
import networkx as nx
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from typing import List

import extract_graphs
import functions

PredictionInfo = namedtuple("PredictionInfo", "true_labels, pred_labels, probs")

METAPATHS = [
	["author", "topic", "author"],
	["author", "institute", "author"],
	["author", "institute", "location", "institute", "author"],
	["author", "institute", "type", "institute", "author"],
	["author", "author", "author"],
]


def to_examples(p_graph, n_graph):
	tr = [(e, 1) for e in p_graph.edges] + [(e, 0) for e in n_graph.edges]
	random.shuffle(tr)
	return [x for x, _ in tr], [y for _, y in tr]


def main():
	base = extract_graphs.get_base_graph()
	# print(base.info())
	folds: List[extract_graphs.FoldGraphs] = extract_graphs.get_fold_graphs()

	results = []
	for fold_i, (tr_p, tr_n, te_p, te_n) in enumerate(folds):
		train_graph = StellarGraph.from_networkx(nx.compose(base, tr_p))
		test_graph = StellarGraph.from_networkx(nx.compose(base, te_p))

		walks = functions.make_walks(train_graph, METAPATHS)
		get_embedding = functions.metapath2vec_embedding(walks)

		bin_op = functions.operator_l2
		tr_x, tr_y = to_examples(tr_p, tr_n)

		clf: Pipeline = functions.train_link_prediction_model(
			tr_x,
			tr_y,
			get_embedding=get_embedding,
			binary_operator=bin_op
		)

		te_x, te_y = to_examples(te_p, te_n)

		test_walks = functions.make_walks(test_graph, METAPATHS)
		get_embedding_test = functions.metapath2vec_embedding(test_walks)

		te_x_vec = functions.link_examples_to_features(
			te_x,
			transform_node=get_embedding_test,
			binary_operator=bin_op
		)

		probs = [p[1] for p in clf.predict_proba(te_x_vec)]
		results.append(PredictionInfo(
			true_labels=te_y,
			pred_labels=[1 if x >= .5 else 0 for x in probs],
			probs=probs
		))

	scores = {
		"auc roc": [roc_auc_score(p.true_labels, p.probs) for p in results],
		"auc pr": [average_precision_score(p.true_labels, p.probs) for p in results],
		"precision": [precision_score(p.true_labels, p.pred_labels) for p in results],
		"recall": [recall_score(p.true_labels, p.pred_labels) for p in results],
		"f1s": [f1_score(p.true_labels, p.pred_labels) for p in results]
	}

	print("label, [scores], mean, std")
	for k, v in scores.items():
		print(k, v, np.average(v), np.std(v))

	with open("coauthor_results.txt", "w") as f:
		f.write("label, [scores], mean, std\n")
		for k, v in scores.items():
			f.write("{} {} {} {}\n".format(k, v, np.average(v), np.std(v)))


if __name__ == '__main__':
	main()
