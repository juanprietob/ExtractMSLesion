
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
from datetime import datetime
import nrrd

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model file computed with autoencoder_train.py', required=True)
parser.add_argument('--sample', help='Evaluate an image sample in nrrd format')
parser.add_argument('--out', help='Write output of evaluation', default="", type=str)
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.')
parser.add_argument('--batch', help='Batch size for evaluation', default=64)


args = parser.parse_args()

pickle_file = args.pickle
model = args.model
batch_size = args.batch
sample = args.sample
outfilename = args.out

if sample:
  img, head = nrrd.read(sample)

  def reformat(image):
    dataset = np.ndarray(shape=tuple([1]) + tuple(image.shape), dtype=np.float32)
    dataset[0] = image
    dataset = np.rollaxis(dataset,1,5)
    return dataset

  batch_data = reformat(img)
  img_size = batch_data[0].shape

  graph = tf.Graph()

  with graph.as_default():

  # run inference on the input data
    x = tf.placeholder(tf.float32, shape=(tuple([1]) + tuple(img_size)))

    y_conv = nn.inference(x, img_size, 1, 1)

    y_conv = tf.argmax( y_conv , axis=4)
    #y_conv = tf.nn.softmax( y_conv )

    with tf.Session() as sess:
      # Add ops to save and restore all the variables.
      saver = tf.train.Saver()
      saver.restore(sess, model)

      prediction = sess.run([y_conv], feed_dict={x: batch_data})

    if outfilename != "":
      print("Writting output prediction to", outfilename)
      nrrd.write(outfilename, prediction[0][0])

else:

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

  #batch_size = 64
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

    y_conv = nn.inference(x, img_size, keep_prob, batch_size)

    intersection_sum, label_sum, example_sum = nn.evaluation(y_conv, y_)

    # setup the summary ops to use TensorBoard
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      saver = tf.train.Saver()
      saver.restore(sess, model)
      
      # specify where to write the log files for import to TensorBoard
      summary_writer = tf.train.SummaryWriter(os.path.dirname(model), sess.graph)

      int_sum = 0
      l_sum = 0
      e_sum = 0

      for step in range(int(len(valid_dataset)/batch_size)):

        offset = (step * batch_size)
        batch_data = valid_dataset[offset:(offset + batch_size), :]
        batch_labels = valid_labels[offset:(offset + batch_size), :]

        ins, ls, es = sess.run([intersection_sum, label_sum, example_sum], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1})

        int_sum += ins
        l_sum += ls
        e_sum += es

      precision = (2.0 * int_sum) / ( l_sum + e_sum )
      print('OUTPUT: %s: Dice metric = %.3f' % (datetime.now(), precision))

      # create summary to show in TensorBoard
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='2Dice metric', simple_value=precision)
      summary_writer.add_summary(summary, global_step)

