from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint as DataPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD as LogisticRegression
import sys

TEST_DATA = "CT/Data\ Science/data/preprocessed_train_5K.csv"
iteration = 10
def ParseData(row):
	data = [float(feature) for feature in row.split(",")]

	#preosjpojapje
	# label, feature = preprocess(row)
	return DataPoint(0.0, data[1:])

#if __name__ == "__main__":
	#conf = SparkConf().setAppName("regressiontest").setMaster("org.apache.spark.deploy.master.Master")
	#sc = SparkContext(conf=conf)
data = sc.textFile(TEST_DATA).map(ParseData)
train, test = data.randomSplit([0.6, 0.4], seed = 11L)
training.cache()
LR = LogisticRegression.train(train, iteration)
predictions = LR.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test.count())
print testErr
metrics = BinaryClassificationMetrics(predsAndLabels)
print metrics.roc()
print LR.predict([1])
sc.stop()