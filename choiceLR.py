from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint as DataPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD as LogisticRegression
import sys
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import csv
TEST_DATA = "../data/preprocessed_train_12K.csv"
iteration = 100
def ParseData(row):
    data = [float(feature) for feature in row.split(",")]
    return DataPoint(data[0], data[7:8])

data = sc.textFile(TEST_DATA).map(ParseData)
train, test = data.randomSplit([0.8, 0.2], seed = 11L)
#train.cache()

LR = LogisticRegression.train(train, iteration)
predictions = LR.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test.count())
print "testErr = %1f" %testErr