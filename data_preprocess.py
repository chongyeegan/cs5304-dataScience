import csv, random
import numpy as np
from collections import deque
from operator import itemgetter
import sys
import pickle
import os.path
DATA_TYPE_FILE = "../data/datatype.pickle"
IDX_FILE = "../data/idx.pickle"
DATA_PATH = "../data/" 
TRAIN_FILE = "train.txt" #45840617 lines
def partitioning(f_name):
	dict_type = {0:"int", 1:"str"}
	type_data_counter = [[0,0] for _ in xrange(40)]
	type_data = [0] * 40
	missing_counter = [0] * 40
	head, ext = TRAIN_FILE.split(".")
	data = []
	idx_10M, idx_5M, idx_2M, idx_3M = genIndices(f_name)
	idx_38M = deque(np.delete(np.array(range(45840617)), np.array(idx_10M)).tolist())
	test_38M = DATA_PATH + head + "_38M.csv"
	'''
	train_10M = DATA_PATH + head + "_10M.csv"
	train_10M = DATA_PATH + head + "_10M.csv"
	train_5M = DATA_PATH + head + "_5M.csv"
	train_2M = DATA_PATH + head + "_2M.csv"
	train_3M = DATA_PATH + head + "_3M.csv"
	train_5K = DATA_PATH + head + "_5K.csv"
	'''
	with open(f_name) as f:
		'''
		f_10M = open(train_10M, "wb")
		f_5M = open(train_5M, "wb")
		f_2M = open(train_2M, "wb")
		f_3M = open(train_3M, "wb")
		f_5K = open(train_5K, "wb")
		'''
		f_38M = open(test_38M, "wb")
		reader = csv.reader(f, delimiter="\t")
		'''
		w_10M = csv.writer(f_10M)
		w_5M = csv.writer(f_5M)
		w_2M = csv.writer(f_2M)
		w_3M = csv.writer(f_3M)
		w_5K = csv.writer(f_5K)
		'''
		w_38M = csv.writer(f_38M)
		increment = 45840617/100
		print "checking data type"

		if os.path.exists(DATA_TYPE_FILE):
			with open(DATA_TYPE_FILE) as f_data_type:
				type_data, missing_counter = pickle.load(f_data_type)
		else:
			for idx, row in enumerate(reader):
				if idx % increment == 0:
					print idx*100/45840617.0, "%"
				for data_idx,data in enumerate(row):
					#print data, data_idx
					#print type_data_counter[data_idx]
					if data == "":
						missing_counter[data_idx]+=1
						continue
					try:
						# print "try int data"
						int(data)
						# print "int success"
						# print data
						type_data_counter[data_idx][1] +=1 
					except:
						# print "int fail"
						type_data_counter[data_idx][0] +=1 
			#for row in type_data_counter:
			#	print row
			for i in xrange(len(type_data_counter)):
				type_data[i], dump = max(enumerate(type_data_counter[i]), key = itemgetter(1))
			with open(DATA_TYPE_FILE, "wb") as f_data_type:
				pickle.dump((type_data, missing_counter), f_data_type)
		print "finished checking data type"
		print type_data
		return 
		f.seek(0)
		#w_5K.writerow(type_data)
		for idx, row in enumerate(reader):
			for i in xrange(len(row)):
				if type_data[i] == 1: # this column is int
					if row[i] == "":
						pass
					try:
						int(row[i]) 
					except:
						#pass
						row[i] = 0 # fail
				else:
				
					if row[i] == "":
						row[i] = "x"
					else:
						try:
							int(row[i])
							row[i] = "x"
						except:
							pass
			if idx % increment == 0:
				print idx*100/45840617.0, "%"
			#print len(idx_10M), idx_10M[0], idx
			'''
			if idx < 5000:
				w_5K.writerow(row)
			'''
			'''
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
			'''
			if idx_38M and idx == idx_38M[0]:
				idx_38M.popleft()
				w_38M.writerow(row)
		'''
		f_10M.close()
		f_5M.close()
		f_2M.close()
		f_3M.close()
		f_5K.close()
		'''
		f_38M.close()
		#print len(data), len(data[0])
def lineOfFile(f_name):
	with open(f_name) as f:
		num_lines = sum(1 for line in f)
	print " has %d line" %num_lines
def genIndices(f_name):
	#num_lines = lineOfFile(f_name)
	print "gen indices"
	if os.path.exists(IDX_FILE):
		with open(IDX_FILE) as f_idx:
			idx_10M, idx_5M, idx_2M, idx_3M = pickle.load(f_idx)
	else:
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
		with open(IDX_FILE, "wb") as f_idx:
			pickle.dump((idx_10M, idx_5M, idx_2M, idx_3M),f_idx)
	#idx_3M.reverse()
	print "gen indices finished"
	#print idx_10M
	return deque(idx_10M), deque(idx_5M), deque(idx_2M), deque(idx_3M)
partitioning(DATA_PATH+TRAIN_FILE)
#lineOfFile(DATA_PATH+TRAIN_FILE)
#lineOfFile(DATA_PATH+"train_10M.csv")
#lineOfFile(DATA_PATH+"train_5M.csv")
#lineOfFile(DATA_PATH+"train_2M.csv")
#lineOfFile(DATA_PATH+"train_3M.csv")
#print np.random.permutation(10).tolist()
