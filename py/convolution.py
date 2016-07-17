
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

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.', required=True)
parser.add_argument('--out', help='Output filename .ckpt file', default="out.ckpt")

args = parser.parse_args()

pickle_file = args.pickle
outvariables = args.out

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

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

# In[ ]:

num_labels = 2

in_depth = img_size[3] #zdim
in_height = img_size[2] #ydim
in_width = img_size[1] #xdim
num_channels = img_size[0] #numchannels

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, in_depth, in_height, in_width, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[ ]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.

# In[ ]:

batch_size = 16
patch_size = 5
depth = 32
depth2 = 32
num_hidden = 64

#[batch, depth, height, width, channels]
stride = [1, 1, 1, 1, 1]

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, in_depth, in_height, in_width, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal([2, patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))

  layer2_weights = tf.Variable(tf.truncated_normal([2, patch_size, patch_size, depth, depth2], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))

  # Fully connected
  layer3_weights = tf.Variable(tf.truncated_normal([ int(round(in_width/4.0))*int(round(in_height/4.0))*depth2, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):


    conv = tf.nn.conv3d(data, layer1_weights, stride, padding='SAME')
    conv = tf.nn.avg_pool3d(conv, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)

    hidden = tf.nn.dropout(hidden, 0.75)

    conv = tf.nn.conv3d(hidden, layer2_weights, stride, padding='SAME')
    conv = tf.nn.avg_pool3d(conv, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)

    hidden = tf.nn.dropout(hidden, 0.75)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3] * shape[4]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 5e-9
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[ ]:

num_steps = 100000

with tf.Session(graph=graph) as session:
  saver = tf.train.Saver()
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
      save_path = saver.save(session, outvariables)
      print("Current model saved in file: %s" % save_path)
  
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  save_path = saver.save(session, outvariables)
  print("Model saved in file: %s" % save_path)



  

# ---
# Problem 1
# ---------
# 
# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.
# 
# ---

# ---
# Problem 2
# ---------
# 
# Try to get the best performance you can using a convolutional net. Look for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, adding Dropout, and/or adding learning rate decay.
# 
# ---
