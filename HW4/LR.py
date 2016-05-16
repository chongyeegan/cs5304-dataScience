from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import pickle
import numpy as np

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
LR_clf = LogisticRegression(n_jobs = -1)
LR_clf.fit(train_X, train_y)

with open("LR_param.dat", "w") as f:
	pickle.dump((LR_clf.coef_, LR_clf.intercept_), f)

result = LR_clf.predict(test_X)
acc = np.sum([1 if y_truth == y_predict else 0 for y_truth, y_predict in zip(test_y, result)]).astype("float32")/len(test_y)
print "Accuracy: ", acc
probas = LR_clf.predict_proba(test_X)
lost = log_loss(test_y, probas)
print 'Lost', lost

print "testing...."
test_labels = LR_clf.predict(test_X)
test_probs = LR_clf.predict_proba(test_X)
print "plotting ROC"
print test_labels.shape, test_probs[:,1].shape
tpr, fpr, th = roc_curve(test_labels, test_probs[:,0])
print tpr.shape
print fpr.shape
#print zip(tpr[:], fpr[:])
plt.plot( fpr[:], tpr[:], color = 'r',linewidth=5.0)
print "testing finished"

max_acc = 0
best_clf = None
max_C = 0
c_pool = [0.01, 0.05,0.1, 0.5, 1.0, 5.0, 10.0]
for c in c_pool:
    tmp_LR_clf = LogisticRegression(C=c, n_jobs = -1)
    tmp_LR_clf.fit(train_X, train_y)
    tmp_test_labels = tmp_LR_clf.predict(validation_X)
    tmp_acc = np.sum([1 if y_truth == y_predict else 0 for y_truth, y_predict in zip(validation_y, tmp_test_labels)])*1.0/len(validation_y)
    
    if tmp_acc > max_acc:
        best_clf = tmp_LR_clf
        max_acc = tmp_acc
        max_C = c
print "validation finished, with the highest acc: ", max_acc, " with C= ", max_C
with open("best_clf_param.dat", "w") as f:
	pickle.dump((best_clf.coef_, best_clf.intercept_), f)

clf_cv = CalibratedClassifierCV(best_clf, cv=5, method='isotonic')
clf_cv.fit(train_X, train_y)
result_cv = clf_cv.predict(test_X)
acc = np.sum([1 if y_truth == y_predict else 0 for y_truth, y_predict in zip(test_y, result_cv)]).astype("float32")/len(test_y)
print "Accuracy: ", acc
probas_cv = clf_cv.predict_proba(test_X)
cv_score = log_loss(test_y, probas_cv)
print 'calibrated score (5-fold:)', cv_score

with open("cv_clf_param.dat", "w") as f:
	pickle.dump((cv_clf.coef_, cv_clf.intercept_), f)

