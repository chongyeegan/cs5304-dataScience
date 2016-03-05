import csv, random
import numpy as np
from collections import deque
DATA_PATH = "../data/" 
TRAIN_FILE = "train.txt" #45840617 lines
def partitioning(f_name):
	head, ext = TRAIN_FILE.split(".")
	data = []
	idx_10M, idx_5M, idx_2M, idx_3M = genIndices(f_name)
	train_10M = DATA_PATH + head + "_10M.csv"
	train_5M = DATA_PATH + head + "_5M.csv"
	train_2M = DATA_PATH + head + "_2M.csv"
	train_3M = DATA_PATH + head + "_3M.csv"
	with open(f_name) as f:
		f_10M = open(train_10M, "wb")
		f_5M = open(train_5M, "wb")
		f_2M = open(train_2M, "wb")
		f_3M = open(train_3M, "wb")
		reader = csv.reader(f)
		w_10M = csv.writer(f_10M)
		w_5M = csv.writer(f_5M)
		w_2M = csv.writer(f_2M)
		w_3M = csv.writer(f_3M)
		increment = 45840617/100
		for idx, row in enumerate(reader):
			if idx % increment == 0:
				print idx*100/45840617.0, "%"
			#print len(idx_10M), idx_10M[0], idx
			if idx_10M and idx == idx_10M[0]:
				#print idx_10M[0]
				idx_10M.popleft()
				w_10M.writerow(row)
				if idx_5M and idx == idx_5M[0]:
					idx_5M.popleft()
					w_5M.writerow(row)
				if idx_2M and idx == idx_2M[0]:
					idx_2M.popleft()
					w_2M.writerow(row)
				if idx_3M and idx == idx_3M[0]:
					idx_3M.popleft()
					w_3M.writerow(row)
		f_10M.close()
		f_5M.close()
		f_2M.close()
		f_3M.close()
		#print len(data), len(data[0])
def lineOfFile(f_name):
	with open(f_name) as f:
		num_lines = sum(1 for line in f)
	print " has %d line" %num_lines
def genIndices(f_name):
	#num_lines = lineOfFile(f_name)
	print "gen indices"
	num_lines = 45840617
	idx_10M = np.random.permutation(45840617)[:10000000].tolist()
	idx_5M = idx_10M[:5000000]
	idx_2M = idx_10M[5000000:7000000]
	idx_3M = idx_10M[7000000:]
	idx_10M.sort()
	#idx_10M.reverse()
	idx_5M.sort()
	#idx_5M.reverse()
	idx_2M.sort()
	#idx_2M.reverse()
	idx_3M.sort()
	#idx_3M.reverse()
	print "gen indices finished"
	#print idx_10M
	return deque(idx_10M), deque(idx_5M), deque(idx_2M), deque(idx_3M)
#partitioning(DATA_PATH+TRAIN_FILE)
#lineOfFile(DATA_PATH+TRAIN_FILE)
lineOfFile(DATA_PATH+"train_10M.csv")
lineOfFile(DATA_PATH+"train_5M.csv")
lineOfFile(DATA_PATH+"train_2M.csv")
lineOfFile(DATA_PATH+"train_3M.csv")
#print np.random.permutation(10).tolist()
