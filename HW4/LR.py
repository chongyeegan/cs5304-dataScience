from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import pickle

mem = Memory("./mycache")

@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]

print "loading train data"
train_X, train_y = get_data("./data/a4_smvl_trn")
print "loading validation data"
validation_X, validation_y = get_data("./data/a4_smvl_val")
print "loading test data"
test_X, test_y = get_data("./data/a4_smvl_tst")
print "loading data finished"

print "start training LR model"
LR_clf = LogisticRegression(n_jobs = -1, warm_start= True)
LR_clf.fit(train_X, train_y)

with open("LR_param.dat", "w") as f:
	pickle.dump((LR_clf.coef_, LR_clf.intercept_), f)