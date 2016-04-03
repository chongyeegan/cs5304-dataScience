import csv, random
import numpy as np
from collections import deque
from operator import itemgetter
import sys
import pickle
import os.path
from collections import defaultdict
DATA_PATH = "../data/" 
TRAIN_FILE = "train_10M.csv" #45840617 lines

def buildCount(f_name):
	dict_list = [defaultdict(int) for _ in xrange(26)]
	increment = 10000000/100
	with open(f_name) as f:
		reader = csv.reader(f)
		#increment = len(reader)/100.0
		for idx, row in enumerate(reader):
			#print idx
			for i in xrange(14, len(row)):
				#print len(row)-1-14
				if row[i] != "x":
					dict_list[i-14][row[i]] += 1
			if idx % increment == 0:
				print idx/increment, "%"
	for mydict in dict_list:
		print len(mydict)
	with open("../data/dict.pickle" , "wb") as f:
		pickle.dump(dict_list, f)
if not os.path.exists("../data/dict.pickle" ):
	buildCount(DATA_PATH+TRAIN_FILE)
with open("../data/dict.pickle") as f:
	dict_list = pickle.load(f)