import csv
import numpy as np
FILE = "reddit_200000.csv"
total = 200000
trim_size = 100000
t1_count, t3_count = 0, 0
with open(FILE) as f:
	reader = csv.reader(f)
	count = 0
	data = []
	randidx = np.random.randint(0, total, trim_size)
	randidx.sort()
	for idx, row in enumerate(reader):
		'''
		head, _ = row[1].split("_")
		if head =="t1":
			t1_count+=1
		if head == "t3":
			t3_count+=1
		'''

		data.append(row)
		count+=1
		if count >= trim_size:
			break

with open("reddit_" + str(trim_size)+".csv", "w") as f:
	writer = csv.writer(f)
	for row in data:
		writer.writerow(row)
