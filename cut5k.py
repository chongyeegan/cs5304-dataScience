import csv
count = 0
with open("../data/preprocessed_train_5M.csv") as f_5M:
	with open("../data/preprocessed_train_12K.csv", "wb") as f_12K:
		reader = csv.reader(f_5M, delimiter=",")
		writer = csv.writer(f_12K)
		for idx, row in enumerate(reader):
			#print row
			if idx >= 12000:
				break
			writer.writerow(row)