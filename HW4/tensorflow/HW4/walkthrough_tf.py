#!/usr/bin/python
import tensorflow as tf
import numpy as np
import random
PATH = "../../data/"
#gunzip -c a4_smvl_trn.gz | head -n 100000 |  gshuf > sample.txt
'''
x = []
y = []
features = set()
for line in file("sample.txt"):
    entries = line.strip().split(" ")
    if int(entries[0]) == -1:
        y.append([0,1])
    else:
        y.append([1, 0])
    x.append({int(k.split(":")[0]):int(k.split(":")[1]) for k in entries[1:]})
    features.update(x[-1].keys())
x_y = zip(x,y)
'''
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
    return data[0], map(lambda x: [0,1] if x == -1 else [1,0], data[1])

print "loading train data"
train_X, train_y = get_data(PATH + "a4_smvl_trn")
#train_x_y = zip(train_X,train_y)
'''
print "loading validation data"
validation_X, validation_y = get_data(PATH + "a4_smvl_val")
'''
print "loading test data"
test_X, test_y = get_data(PATH+"a4_smvl_tst")
#test_x_y = zip(test_X, test_y)
print "loading data finished"

print "read train: X:", train_X.shape,"y: ", train_y.shape

feature_count = train_X.shape
learning_rate = 0.001

sess = tf.InteractiveSession()


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)



x_tensor = tf.placeholder("float", [None, feature_count])
y_tensor = tf.placeholder("float", [None, 2])

def weight_variable(shape, name):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)


def bias_variable(shape, name):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
 #       tf.histogram_summary(name, var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read, and
    adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim], layer_name+"_weights")
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim], layer_name+"_biases")
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
#            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
#        tf.histogram_summary(layer_name + '/activations', activations)
        return activations

inter = nn_layer(x_tensor, feature_count, 300, 'layer1')
y = nn_layer(inter, 300, 2, 'layer2', act=tf.nn.softmax)

with tf.name_scope('cross_entropy'):
    diff = y_tensor * tf.log(y)
    with tf.name_scope('total'):
        cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_tensor, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()
init = tf.initialize_all_variables()
sess.run(init)
train_writer = tf.train.SummaryWriter('/tmp/log', sess.graph)

batch_size = 100
counter = 0
saver = tf.train.Saver()
for epoch in range(50):
    random.shuffle(train_x_y)
    start = 0
    while start < len(train+x_y):
        #batch = np.array(train_x_y[start:start+batch_size])
        batch_X = train_X[start:start+batch_size, :]
        batch_y = train_y[tart:start+batch_size, :]
        start = start+batch_size
        labels = []
        #feats_np = np.zeros((len(batch),feature_count))
        '''
        for i,f_l in enumerate(batch):
            feats,label  = f_l
            for k,v in feats.iteritems():
                feats_np[i,k-1] = v
            labels.append(label)
        labels_np = np.vstack(labels)
        '''
        feats_np = batch_X.todense()
        labels_np = batch_y
        feed_dict = {x_tensor:feats_np, y_tensor:labels_np}
        # print x_tensor,y_tensor,feats_np.shape,labels_np.shape
        summary,_ = sess.run([merged,train_step],feed_dict=feed_dict)
        train_writer.add_summary(summary, counter)
        counter += 1
        # prediction = sess.run(y,feed_dict=feed_dict)
        # print len(prediction)
    print epoch
    if (epoch+1)%10 == 0:
        save_path = saver.save(sess, "/tmp/model_"+str(epoch+1)+".ckpt")
        print "save ", epoch, "th variables to ", save_path