
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 4
# ------------
# 
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.
# 
# The goal of this assignment is make the neural network convolutional.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import argparse
import autoencoder_nn as nn
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.', required=True)
parser.add_argument('--out', help='Output filename .ckpt file', default="out.ckpt")
parser.add_argument('--learning_rate', help='Learning rate', default=1e-8)
parser.add_argument('--decay_rate', help='decay rate', default=0.96)
parser.add_argument('--decay_steps', help='decay steps', default=100000)

args = parser.parse_args()

pickle_file = args.pickle
outvariablesfilename = args.out
learning_rate = args.learning_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps

f = open(pickle_file, 'rb')
data = pickle.load(f)
train_dataset = data["train_dataset"]
train_labels = data["train_labels"]
valid_dataset = data["valid_dataset"]
valid_labels = data["valid_labels"]
test_dataset = data["test_dataset"]
test_labels = data["test_labels"]
img_head = data["img_head"]
img_size = img_head["sizes"]
img_head_label = data["img_head_label"]
img_size_label = img_head_label["sizes"]

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

# In[ ]:

# in_depth = img_size[3] #zdim
# in_height = img_size[2] #ydim
# in_width = img_size[1] #xdim
# num_channels = img_size[0] #numchannels
# num_channels_labels = 1

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (depth * width * height * channels)
# - We know that nrrd format 
# - labels as float 1-hot encodings.

def reformat(dataset, labels):
  dataset = dataset.reshape(tuple([-1]) + tuple(reversed(img_size))).astype(np.float32)
  labels = labels.reshape(tuple([-1]) + tuple(reversed(img_size_label))).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.

# In[ ]:

batch_size = 64
# patch_size = 8
# depth = 32
# depth2 = 64
# num_hidden = 256
# stride = [1, 1, 1, 1]

# def evaluate_accuracy(prediction, labels):    
#   accuracy = tf.reduce_sum(tf.squared_difference(prediction, labels))
#   return accuracy.eval()

graph = tf.Graph()

with graph.as_default():

# run inference on the input data
  x = tf.placeholder(tf.float32, shape=(tuple([batch_size]) + tuple(reversed(img_size))))
  y_ = tf.placeholder(tf.float32, shape=(tuple([batch_size]) + tuple(reversed(img_size_label))))
  keep_prob = tf.placeholder(tf.float32)

  tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)

  y_conv = nn.inference(x, img_size, keep_prob, batch_size)

# calculate the loss from the results of inference and the labels
  loss = nn.loss(y_conv, y_)

# setup the training operations
  train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)

  # setup the summary ops to use TensorBoard
  summary_op = tf.merge_all_summaries()

  # intersection_sum, label_sum, example_sum = evaluation(y_conv, y_)

  # valid_prediction = model(tf_valid_dataset)
  #cross_entropy = tf.reduce_sum(tf.squared_difference(y_conv, y_))

  #regularizers = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
  #cross_entropy += 0.1 * regularizers

  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  #train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)  

  # accuracy = cross_entropy

  # valid_prediction = model(tf_valid_dataset)
  # evaluation(valid_prediction)
  # test_prediction = model(tf_test_dataset)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(session, model)


    for i in range(20000):
      offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = valid_dataset[offset:(offset + batch_size), :]
      batch_labels = valid_labels[offset:(offset + batch_size), :]

      _, loss_value, summary = sess.run([train_step, loss, summary_op], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
      #train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

      if i%100 == 0:
        
        print("step %d, loss %g"%(i, loss_value))
        valid_accuracy = evaluate_accuracy(valid_prediction.eval(feed_dict={keep_prob: 1.0}), valid_labels)
        print("\tvalid accuracy %g"%valid_accuracy)        

    #test_accuracy = evaluate_accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels)
    #print("test accuracy %g"%test_accuracy)
    
  
  
