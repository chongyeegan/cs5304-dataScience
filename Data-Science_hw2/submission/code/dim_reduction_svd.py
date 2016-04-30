import settings
import numpy as np
import operator
from scipy.linalg import svd
#from scipy.sparse import svds as svd
from scipy.sparse import csr_matrix

def readData():
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
	
	return user_dict, movie_dict, np.array(data)

def buildMatrix(data, user_dict, movie_dict, num_user, num_movie):
	
	matrix = np.array([[0.0] * num_movie for _ in xrange(num_user)])
	for user, movie, rating, timestamp in data:
		matrix[user_dict[user], movie_dict[movie]] = float(rating)
	return matrix

def dim_reduction(X, w, l, u = np.array([]), v_T = np.array([])):
	if not u.any() and not v_T.any():
		u, s, v_T = svd(X)
	X_reduce = ((u[:, :w].transpose()).dot(X)).dot(v_T[:l, :].transpose())
	return X_reduce, u, v_T

print "Reading data"
user_dict, movie_dict, data = readData()
cut1, cut2 = int(len(data)*0.6), int(len(data)*0.8)
train, validate,test = data[:cut1, :], data[cut1:cut2, :], data[cut2:, :]
num_user, num_movie = len(user_dict), len(movie_dict)

w, l = int(num_user/2), int(num_movie/2)
print w,l
print "SVD, train_reduction"
train_matrix = buildMatrix(train, user_dict, movie_dict, num_user, num_movie)
train_reduce, u, v_T = dim_reduction(train_matrix, w, l)
print "save train"
with open(DATA_FOLDER + "train.dat", "wb") as f:
	for x in xrange(train_reduce.shape[0]):
		for y in xrange(train_reduce.shape[1]):
			f.write(str(x) + "::" + str(y) + "::" + str(train_reduce[x,y]) + "\n")

print "validate_reduction"
validate_matrix = buildMatrix(validate, user_dict, movie_dict, num_user, num_movie)
validate_reduce, u, v_T = dim_reduction(validate_matrix, w, l, u, v_T)
print "save validate"
with open(DATA_FOLDER + "validate.dat", "wb") as f:
	for x in xrange(validate_reduce.shape[0]):
		for y in xrange(validate_reduce.shape[1]):
			f.write(str(x) + "::" + str(y) + "::" + str(validate_reduce[x,y]) + "\n")


print "test_reduction"
test_matrix = buildMatrix(test, user_dict, movie_dict, num_user, num_movie)
test_reduce, u, v_T = dim_reduction(test_matrix, w, l, u, v_T)

print "save test"
with open(DATA_FOLDER + "test.dat", "wb") as f:
	for x in xrange(test_reduce.shape[0]):
		for y in xrange(test_reduce.shape[1]):
			f.write(str(x) + "::" + str(y) + "::" + str(test_reduce[x,y]) + "\n")

