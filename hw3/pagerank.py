import networkx as nx
import matplotlib.pyplot as plt
TRAIN_FILE = "militaryAid.csv"

G = nx.DiGraph()
with open(TRAIN_FILE) as f:
	for row in f:
		actor1, actor2 = row.strip("\n").split(",")
		G.add_edge(actor1, actor2)

pr = nx.pagerank(G)
print pr
sort_pr = sorted(pr.keys(), key = lambda x :pr[x], reverse=True)
print sort_pr[:10]