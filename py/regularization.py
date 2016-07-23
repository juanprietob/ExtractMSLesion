
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 3
# ------------
# 
# Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.
# 
# The goal of this assignment is to explore regularization techniques.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import nrrd


# First reload the data we generated in _notmist.ipynb_.

# In[ ]:

# pickle_file = 'notMNIST.pickle'
# with open(pickle_file, 'rb') as f:
#   data = pickle.load(f)
#   train_dataset = data['train_dataset']
#   train_labels = data['train_labels']
#   valid_dataset = data['valid_dataset']
#   valid_labels = data['valid_labels']
#   test_dataset = data['test_dataset']
#   test_labels = data['test_labels']
#   del data  # hint to help gc free up memory

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.')
parser.add_argument('--out', help='Output filename .ckpt file', default="out.ckpt")
parser.add_argument('--img', help='Evaluate an image')
parser.add_argument('--model', help='The deep learning model')

args = parser.parse_args()

pickle_file = args.pickle
outvariables = args.out
img = args.img
model = args.model

num_labels = 2

if img:
  image, head = nrrd.read(img)
  img_size = head['sizes']
  image = image.reshape((-1, img_size[0]*img_size[1]*img_size[2]*img_size[3])).astype(np.float32)
elif not pickle_file:
  parser.print_help()
  quit()
else:
  f = open(pickle_file, 'rb')
  data = pickle.load(f)
  train_dataset = data["train_dataset"]
  train_labels = data["train_labels"]
  valid_dataset = data["valid_dataset"]
  valid_labels = data["valid_labels"]
  test_dataset = data["test_dataset"]
  test_labels = data["test_labels"]
  img_size = data["img_head"]["sizes"]

  def reformat(dataset, labels):
    dataset = dataset.reshape((-1, img_size[0]*img_size[1]*img_size[2]*img_size[3])).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
  train_dataset, train_labels = reformat(train_dataset, train_labels)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
  test_dataset, test_labels = reformat(test_dataset, test_labels)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)




def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# ---
# Problem 1
# ---------
# 
# Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.
# 
# ---


batch_size = 2048
hidden_nodes = [1024, 1024, 512]

num_steps = 15000
reg_constant = 0.1
weights = []
biases = []

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, img_size[0]*img_size[1]*img_size[2]*img_size[3]))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  keep_prob = tf.placeholder(tf.float32)
  tf_image = tf.placeholder(tf.float32,shape=(1, img_size[0]*img_size[1]*img_size[2]*img_size[3]))

  if not img:
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

  if len(hidden_nodes) > 0:
    inputlayersize = img_size[0]*img_size[1]*img_size[2]*img_size[3]

    for idx, hn in enumerate(hidden_nodes):

      wn = tf.Variable(tf.truncated_normal([inputlayersize, hn]))
      bn = tf.Variable(tf.zeros([hn]))

      weights.append(wn)
      biases.append(bn)
      
      if idx + 1 == len(hidden_nodes):
        wend = tf.Variable(tf.truncated_normal([hn, num_labels]))
        bend = tf.Variable(tf.zeros([num_labels]))
        weights.append(wend)
        biases.append(bend)
      else:
        inputlayersize = hn

  else:
    wn = tf.Variable(tf.truncated_normal([img_size[0]*img_size[1]*img_size[2]*img_size[3], num_labels]))
    bn = tf.Variable(tf.zeros([num_labels]))

    weights.append(wn)
    biases.append(bn)


  # Variables.
  

  # Relu
  # def layer(ds, w1, b1, w2, b2, dropprob=1):
  #   logits = tf.matmul(ds, w1) + b1
  #   droped = tf.nn.dropout(tf.nn.relu(logits), dropprob)
  #   return tf.matmul(droped, w2) + b2

  def layer(ds, w, b, relu=False):
    output = tf.matmul(ds, w) + b
    if(relu):
      output = tf.nn.relu(output)
    return output

  def layers(ds):
    layerout = ds
    for idx, (w, b) in enumerate(zip(weights, biases)):
      relu = idx < len(weights) - 1
      layerout = layer(layerout, w, b, relu)
    return layerout

  def regularize():
    output = 0
    for w in weights:
      output += reg_constant*tf.nn.l2_loss(w)
    return output
  
  logits = layers(tf_train_dataset)
  logits = tf.nn.dropout(logits, keep_prob)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + regularize()
  # Optimizer.
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 1e-9
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(layers(tf_train_dataset))
  if not img:
    valid_prediction = tf.nn.softmax(layers(tf_valid_dataset))
    test_prediction = tf.nn.softmax(layers(tf_test_dataset))
  predict_image = tf.nn.softmax(layers(tf_image))


# Let's run it:

# In[ ]:

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  if img and model:
    saver.restore(session, model)
    feed_dict = {tf_image : image, keep_prob: 1}
    print(predict_image.eval(feed_dict=feed_dict)[0][0],",",predict_image.eval(feed_dict=feed_dict)[0][1])

  else:
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 500 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval({keep_prob: 1}), valid_labels))
        save_path = saver.save(session, outvariables)
        print("Current model saved in file: %s" % save_path)

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval({keep_prob: 1}), test_labels))

