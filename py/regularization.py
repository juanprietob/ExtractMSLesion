
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
parser.add_argument('--summary', help='Create a summary of the graph', default=False)
parser.add_argument('--img', help='Evaluate an image')
parser.add_argument('--imgLabel', help='Evaluate an image and use label map')
parser.add_argument('--neighborhood', help='Set the image neighborhood, required when using img', nargs='+', type=int)
parser.add_argument('--model', help='The deep learning model')
parser.add_argument('--outImageLabel', help='Output image with probability')
parser.add_argument('--sample', help='Evaluate an sample image')

args = parser.parse_args()

pickle_file = args.pickle
outvariables = args.out
createsummary = args.summary
img = args.img
imgLabel = args.imgLabel
model = args.model
neighborhood = args.neighborhood
outImageLabel = args.outImageLabel
sample = args.sample

num_labels = 2

img_data = None
img_size = None
img_size_in = None
img_size_all = 1
img_data_label = None
img_head_label = None

if model:

  if img and neighborhood:
    try:
      img_data, img_head = nrrd.read(img)
      img_size_in = img_head["sizes"]
      img_size = np.array([num_labels, neighborhood[0]*2+1, neighborhood[1]*2+1, neighborhood[2]*2+1])
      
      if imgLabel:
        img_data_label, img_head_label = nrrd.read(imgLabel)
        
    except Exception as e:
      print('Could not read:', img, e)
      quit()
  elif sample:
    try:
      img_data, img_head = nrrd.read(sample)
      img_size = img_head["sizes"]
      
    except Exception as e:
      print('Could not read:', img, e)
      quit()
  else:
    print('type --help to learn how to use this program')
    quit()
  
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

batch_size = 2048
hidden_nodes = [1024, 1024, 512]

num_steps = 15000
reg_constant = 0.1
weights = []
biases = []

img_size_all = img_size[0]*img_size[1]*img_size[2]*img_size[3]

def evaluate_accuracy(prediction, labels):
  return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1)), tf.float32))

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, img_size[0]*img_size[1]*img_size[2]*img_size[3]))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  keep_prob = tf.placeholder(tf.float32)
  tf_image = tf.placeholder(tf.float32,shape=(1, img_size[0]*img_size[1]*img_size[2]*img_size[3]))

  if not img and not sample:
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

  def variable_summaries(var, name):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('sttdev/' + name, stddev)
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
      tf.histogram_summary(name, var)

  def layer(ds, w, b):
    output = tf.matmul(ds, w) + b
    return output

  def layers(ds, name=""):
    layerout = ds
    for idx, (w, b) in enumerate(zip(weights, biases)):
      layerout = layer(layerout, w, b)

      if idx < len(weights) - 1:
        act = tf.nn.relu(layerout)

      if not name == "":
        layer_name = name + str(idx)
        with tf.name_scope(layer_name):
          variable_summaries(w, layer_name + '/weights')
          variable_summaries(b, layer_name + '/biases')
          tf.histogram_summary(layer_name + '/pre_activations', layerout)
          if idx < len(weights) - 1:
            tf.histogram_summary(layer_name + '/activations', act)

      if idx < len(weights) - 1:
        layerout = act

    return layerout  
      
    

  def regularize():
    output = 0
    for w in weights:
      output += reg_constant*tf.nn.l2_loss(w)
    return output
  
  if createsummary:
    layerout = layers(tf_train_dataset, "train")
  else:
    layerout = layers(tf_train_dataset)

  logits = tf.nn.dropout(layerout, keep_prob)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  reg = regularize()

  with tf.name_scope('cross_entropy'):
    tf.scalar_summary('cross entropy', loss)
    tf.scalar_summary('regularization', reg)


  # Optimizer.
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 1e-9
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss + reg, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(layerout)
  if not img and not sample:
    if createsummary:
      valid_prediction = tf.nn.softmax(layers(tf_valid_dataset, "valid"))
      test_prediction = tf.nn.softmax(layers(tf_test_dataset, "test"))
      with tf.name_scope('accuracy'):
        tf.scalar_summary('minibatch_accuracy', evaluate_accuracy(train_prediction, tf_train_labels))
    else:
      valid_prediction = tf.nn.softmax(layers(tf_valid_dataset))
      test_prediction = tf.nn.softmax(layers(tf_test_dataset))
  else:
    predict_image = tf.nn.softmax(layers(tf_image))

def getNeighborhood(image, x, y, z, neighborhood, array):
  for i, xx in zip(range(0, neighborhood[0]*2 + 1), range(-neighborhood[0], neighborhood[0])):
    for j, yy in zip(range(0, neighborhood[1]*2 + 1), range(-neighborhood[1], neighborhood[1])):
      for k, zz in zip(range(0, neighborhood[2]*2 + 1), range(-neighborhood[2], neighborhood[2])):
        array[0][i][j][k] = image[0][xx + x][yy + y][zz + z]
        array[1][i][j][k] = image[1][xx + x][yy + y][zz + z]
  return array

def predictImage(image, size, neighborhood, imglabel, imgheadlabel):
  neighborhood_size = (neighborhood[0]*2 + 1)*(neighborhood[1]*2 + 1)*(neighborhood[2]*2 + 1)*num_labels
  neigh = np.zeros((num_labels, neighborhood[0]*2 + 1, neighborhood[1]*2 + 1, neighborhood[2]*2 + 1))
  labeledImage = np.zeros(size)
  for i in range(neighborhood[0], size[1]-neighborhood[0]):
    for j in range(neighborhood[1], size[2]-neighborhood[1]):
      for k in range(neighborhood[2], size[3]-neighborhood[2]):
        predict = True
        
        if not imglabel is None:
          predict = imglabel[i][j][k] == 6

        if predict:
          neigh = getNeighborhood(image, i, j, k, neighborhood, neigh)
          prediction = predict_image.eval({keep_prob: 1, tf_image: neigh.reshape(-1, neighborhood_size).astype(np.float32)})
          labeledImage[0][i][j][k] = prediction[0][0]
          labeledImage[1][i][j][k] = prediction[0][1]

  return labeledImage


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  
  if createsummary:
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./train', session.graph)

  if model:
    
    saver.restore(session, model)
    
    if img_data is not None:
      if img:
        labelImg = predictImage(img_data, img_size_in, neighborhood, img_data_label, img_head_label)
        if outImageLabel:
          nrrd.write(outImageLabel, labelImg)
      elif sample is not None:
        prediction = predict_image.eval({keep_prob: 1, tf_image: img_data.reshape(-1,img_size_all ).astype(np.float32)})
        print(prediction);


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
      if createsummary:
        summary, _, l, predictions = session.run([merged, optimizer, loss, train_prediction], feed_dict=feed_dict)
      else:
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

      if (step % 500 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        minibatch_accuracy = evaluate_accuracy(predictions, batch_labels)
        validation_accuracy = evaluate_accuracy(valid_prediction.eval({keep_prob: 1}), valid_labels)

        print("Minibatch accuracy: %.3f" % minibatch_accuracy.eval())
        print("Validation accuracy: %.3f" % validation_accuracy.eval())
        save_path = saver.save(session, outvariables)
        print("Current model saved in file: %s" % save_path)
        if createsummary:
          train_writer.add_summary(summary, step)
              
    test_accuracy = evaluate_accuracy(test_prediction.eval({keep_prob: 1}), test_labels)
    print("Test accuracy: %.3f" % test_accuracy.eval())

