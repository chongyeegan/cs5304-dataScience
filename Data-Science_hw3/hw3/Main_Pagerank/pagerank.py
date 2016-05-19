import networkx as nx
import csv
import matplotlib.pyplot as plt
TRAIN_FILE = "militaryAid.csv"
#TRAIN_FILE = "militaryAid_received_byAvgTone.csv"
#TRAIN_FILE = "militaryAid_byAvgTone_1990To2000.csv"
#TRAIN_FILE = "militaryAid_byAvgTone_2000To2010.csv"

G = nx.DiGraph()
with open(TRAIN_FILE) as f:
	reader = csv.reader(f)
	reader.next()
	for row in reader:
		actor1, actor2 = row[0], row[3] #pagerank with actor name
		#actor1, actor2 = row[1], row[5] #pagerank with country name
		G.add_edge(actor1, actor2)

pr = nx.pagerank(G)
sort_pr = sorted(pr.keys(), key = lambda x :pr[x], reverse=True)
print sort_pr[:10]