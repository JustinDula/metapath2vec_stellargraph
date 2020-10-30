import re
from collections import namedtuple
from os.path import join as pjoin
from shutil import rmtree
import networkx as nx

NUM_FOLDS = 5
FOLDS = ["data/5folds/fold" + str(i + 1) for i in range(NUM_FOLDS)]

FoldGraphs = namedtuple("FoldGraphs", "tr_p, tr_n, te_p, te_n")


def get_base_graph():
	base_graph = nx.Graph()

	raw_pattern = r"""([\w]+)\("([\w_]+)","([\w_]+)"\)\."""
	reg = re.compile(raw_pattern)

	with open("data/5folds/train_facts.txt", "r") as f:
		for i, line in enumerate(f.readlines()):
			line = line.strip()
			if match := reg.match(line):
				rel, a, b = match.groups()

				base_graph.add_edge(a, b, label=rel)
				if rel == "Affiliation":
					base_graph.nodes[a]["label"] = "author"
					base_graph.nodes[b]["label"] = "institute"
				elif rel == "InstituteType":
					base_graph.nodes[a]["label"] = "institute"
					base_graph.nodes[b]["label"] = "type"
				elif rel == "ResearchTopic":
					base_graph.nodes[a]["label"] = "author"
					base_graph.nodes[b]["label"] = "topic"
				elif rel == "Location":
					base_graph.nodes[a]["label"] = "institute"
					base_graph.nodes[b]["label"] = "location"
				else:
					print(f"ERROR: unrecognized rel @ line {i+1}: {rel} : {a}, {b}")

	return base_graph


def extract_coauthors(f):
	raw_pattern = r"""CoAuthor\("([\w_]+)","([\w_]+)"\)\."""
	re_pattern = re.compile(raw_pattern)

	g = nx.Graph()
	for line in f.readlines():
		if line.strip() == "":
			pass
		a, b = re_pattern.match(line).groups()
		g.add_edge(a, b, label="CoAuthor")
		g.nodes[a]["label"] = "author"
		g.nodes[b]["label"] = "author"
	return g


def get_fold_graphs():

	pos_graphs = []
	neg_graphs = []

	for in_path in FOLDS:
		with open(pjoin(in_path, "pos.txt"), "r") as f:
			pos_graphs.append(extract_coauthors(f))

		with open(pjoin(in_path, "neg.txt"), "r") as f:
			neg_graphs.append(extract_coauthors(f))

	folds = []

	for i, (pos_test, neg_test) in enumerate(zip(pos_graphs, neg_graphs)):
		tr_p, tr_n = nx.Graph(), nx.Graph()
		for j, (pos_graph, neg_graph) in enumerate(zip(pos_graphs, neg_graphs)):
			if j != i:
				tr_p = nx.compose(tr_p, pos_graph)
				tr_n = nx.compose(tr_n, pos_graph)
		folds.append(FoldGraphs(tr_p=tr_p, tr_n=tr_n, te_p=pos_test, te_n=neg_test))

	return folds