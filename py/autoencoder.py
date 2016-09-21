
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
outvariablesfilename = args.out

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

in_depth = img_size[3] #zdim
in_height = img_size[2] #ydim
in_width = img_size[1] #xdim
num_channels = img_size[0] #numchannels
num_channels_labels = 1

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
patch_size = 8
depth = 32
depth2 = 64
num_hidden = 256
stride = [1, 1, 1, 1]

def evaluate_accuracy(prediction, labels):
  correct_prediction = tf.equal(prediction, labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy.eval()

graph = tf.Graph()

with graph.as_default():

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

  def deconv3d(x, W, outputshape):
    return tf.nn.conv3d_transpose(x, W, outputshape, strides=[1, 2, 2, 2, 1], padding='SAME')

  def max_pool_3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

  def max_unpool_3d(x):
    sh = x.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(x, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat(i, [out, tf.zeros_like(out)])
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

  x = tf.placeholder(tf.float32, shape=(tuple([batch_size]) + tuple(reversed(img_size))))
  y_ = tf.placeholder(tf.float32, shape=(tuple([batch_size]) + tuple(reversed(img_size_label))))
  
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  keep_prob = tf.placeholder(tf.float32)

  W_conv1 = weight_variable([patch_size, patch_size, patch_size, num_channels, depth])
  b_conv1 = bias_variable([depth])
  
  W_conv2 = weight_variable([patch_size, patch_size, patch_size, depth, depth2])
  b_conv2 = bias_variable([depth2])

  W_fc1 = weight_variable([patch_size*patch_size*patch_size*depth2, num_hidden])
  b_fc1 = bias_variable([num_hidden])
  
  W_fc2 = weight_variable([num_hidden, num_hidden])
  b_fc2 = bias_variable([num_hidden])

  W_fc3 = weight_variable([num_hidden, patch_size*patch_size*patch_size*depth2])
  b_fc3 = bias_variable([patch_size*patch_size*patch_size*depth2])

  W_unconv1 = weight_variable([patch_size, patch_size, patch_size, depth, depth2])
  b_unconv1 = bias_variable([depth])
  
  W_unconv2 = weight_variable([patch_size, patch_size, patch_size, num_channels_labels, depth])
  b_unconv2 = bias_variable([num_channels_labels])
  
  # Model.
  def model(x_image):

    #convolutional layers
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_3d(h_conv1)
    
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_3d(h_conv2)

    #fully connected
    shape = h_pool2.get_shape().as_list()
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, shape[1]*shape[2]*shape[3]*shape[4]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    h_fc3_flat = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
    h_fc3 = tf.reshape(h_fc3_flat, [-1, shape[1], shape[2], shape[3], shape[4]])
    
    h_conv3 = tf.nn.relu(deconv3d(h_fc3, W_unconv1, [shape[0], in_width//2, in_height//2, in_depth//2, depth]) + b_unconv1)
    #h_pool1.get_shape().as_list()

    h_conv4 = tf.nn.relu(deconv3d(h_conv3, W_unconv2, [shape[0], in_width, in_height, in_depth, num_channels_labels]) + b_unconv2)

    return tf.reshape(h_conv4, [-1, in_width, in_height, in_depth])
  
  y_conv = model(x)
  
  cross_entropy = tf.reduce_mean(y_conv - y_)

  regularizers = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) 
  cross_entropy += 0.1 * regularizers

  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(y_conv, y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  valid_prediction = model(tf_valid_dataset)
  test_prediction = model(tf_test_dataset)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    for i in range(20000):
      offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_data, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        valid_accuracy = evaluate_accuracy(valid_prediction.eval(feed_dict={keep_prob: 1.0}), valid_labels)
        print("\tvalid accuracy %g"%valid_accuracy)
        save_path = saver.save(sess, outvariablesfilename)
        print("Current model saved in file: %s" % save_path)
      train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

    test_accuracy = evaluate_accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels)
    print("test accuracy %g"%test_accuracy)
    
  
  
