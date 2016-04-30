import settings
import numpy as np
import operator
from scipy.linalg import svd
from sklearn.cluster import KMeans as kmeans
import pickle
import os
def readData():
	if os.path.exists("data.pickle"):
		with open("data.pickle") as f:
			user_dict, movie_dict, data = pickle.load(f)
	else:
		data = []
		user_dict = {}
		user_count = 0
		with open(settings.RATINGS_10M) as f:
			for row in f:
				user, movie, rating, timestamp = row.strip("\n").split("::")
				if user not in user_dict:
					user_dict[user] = user_count
					user_count+= 1
				data.append([user, movie, rating, timestamp])
		data.sort(key = operator.itemgetter(3))

		movie_dict = {}
		movie_count = 0
		with open(settings.MOVIE_LIST) as f:
			for row in f:
				movie_id, movie_name, feature = row.strip("\n").split("::")
				if movie_id not in movie_dict:
					movie_dict[movie_id] = movie_count
					movie_count += 1
		with open("data.pickle", "wb") as f:
			pickle.dump((user_dict, movie_dict, data), f)
	
	return user_dict, movie_dict, np.array(data)

def buildMatrix(data, user_dict, movie_dict, num_user, num_movie):
	matrix = [[0.0] * num_movie for _ in xrange(num_user)]
	for user, movie, rating, timestamp in data:
		matrix[user_dict[user]][movie_dict[movie]] = float(rating)
	return np.array(matrix)

def dim_reduction(X, w, l, u = None, v_T = None):
	print X
	if not u and not v_T:
		u, s, v_T = svd(X)
	X_reduce = ((u[:, :w].transpose()).dot(X)).dot(v_T[:l, :].transpose())
	return X_reduce, u, v_T

print "Reading data"
user_dict, movie_dict, data = readData()
cut1, cut2 = int(len(data)*0.6), int(len(data)*0.8)
train, validate,test = data[:cut1, :], data[cut1:cut2, :], data[cut2:, :]
num_user, num_movie = len(user_dict), len(movie_dict)

w, l = int(num_user/2), int(num_movie/2)

print "train kmeans classifier"
train_reduce = [0,0,0.0] * len(train)
user_clf = kmeans(n_clusters=w, max_iter= 30, n_jobs=4)
print "train user"
train_category = user_clf.fit_predict(train[:,[0,2]])
for i in xrange(len(train_category)):
	train_reduce[i][0] = train_category[i]
	train_reduce[i][1] = train[i, 1]
	train_reduce[i][2] = user_clf.cluster_centers_[train_category[i], 1]
print "train product"
product_clf = kmeans(n_clusters=l, max_iter= 30, n_jobs=4)
train_category = product_clf.fit_predict(train_reduce[:,[1,2]])
for i in xrange(len(train_category)):
	train_reduce[i][1] = train_category[i]
	train_reduce[i][2] = product_clf.cluster_centers_[train_category[i], 1]

print "validate_reduction"
validate_reduce = [0,0,0.0] * len(validate)
validate_category = user_clf.predict(validate[:,[0,2]])
for i in xrange(len(validate_category)):
	validate_reduce[i][0] = validate_category[i]
	validate_reduce[i][1] = validate[i, 1]
	validate_reduce[i][2] = user_clf.cluster_centers_[validate_category[i], 1]

validate_category = product_clf.predict(validate_reduce[:,[1,2]])
for i in xrange(len(validate_category)):
	validate_reduce[i][1] = validate_category[i]
	validate_reduce[i][2] = product_clf.cluster_centers_[validate_category[i], 1]

print "test_reduction"
test_reduce = [0,0,0.0] * len(test)
test_category = user_clf.predict(test[:,[0,2]])
for i in xrange(len(test_category)):
	test_reduce[i][0] = test_category[i]
	test_reduce[i][1] = test[i, 1]
	test_reduce[i][2] = user_clf.cluster_centers_[test_category[i], 1]

test_category = product_clf.predict(test_reduce[:,[1,2]])
for i in xrange(len(test_category)):
	test_reduce[i][1] = test_category[i]
	test_reduce[i][2] = product_clf.cluster_centers_[test_category[i], 1]

print "save data"
with open(DATA_FOLDER + "train.dat", "wb") as f:
	for user, product, rating in train_reduce:
		f.write(str(user) + "::" + str(product) + "::" + str(rating) + "\n")

with open(DATA_FOLDER + "validate.dat", "wb") as f:
	for user, product, rating in validate_reduce:
		f.write(str(user) + "::" + str(product) + "::" + str(rating) + "\n")

with open(DATA_FOLDER + "test.dat", "wb") as f:
	for user, product, rating in test_reduce:
		f.write(str(user) + "::" + str(product) + "::" + str(rating) + "\n")