from stellargraph import StellarGraph


from typing import List

import extract_graphs
import functions


METAPATHS = [
	["author", "topic", "author"],
	["author", "institute", "author"],
	["author", "institute", "location", "institute", "author"],
	["author", "institute", "type", "institute", "author"],
	["author", "institute", "type", "institute", "author"],
	# ["author", "author"]
]


def to_examples(p_graph, n_graph):
	tr_x = list(p_graph.edges) + list(n_graph.edges)
	tr_y = [1] * len(p_graph.edges) + [0] * len(n_graph.edges)

	return tr_x, tr_y


def main():
	base = StellarGraph.from_networkx(extract_graphs.get_base_graph())
	# print(base.info())
	folds: List[extract_graphs.FoldGraphs] = extract_graphs.get_fold_graphs()

	for fold_i, (tr_p, tr_n, te_p, te_n) in enumerate(folds):
		walks = functions.make_walks(base, METAPATHS)
		get_embedding = functions.metapath2vec_embedding(walks)

		bin_op = functions.operator_l2

		tr_x, tr_y = to_examples(tr_p, tr_n)

		clf = functions.train_link_prediction_model(
			tr_x,
			tr_y,
			get_embedding=get_embedding,
			binary_operator=bin_op
		)

		te_x, te_y = to_examples(tr_p, tr_n)

		print(functions.evaluate_link_prediction_model(
			clf=clf,
			binary_operator=bin_op,
			get_embedding=get_embedding,
			link_examples_test=te_x,
			link_labels_test=te_y
		))


if __name__ == '__main__':
	main()
