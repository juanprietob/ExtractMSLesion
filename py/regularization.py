
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
import multiprocessing
from multiprocessing import Pool
import argparse
import glob, os

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.')
parser.add_argument('--num_labels', help='Set the number of labels of the training data', default=2, type=int)
parser.add_argument('--out', help='Output filename .ckpt file', default="out.ckpt")
parser.add_argument('--summary', help='Create a summary of the graph', default=False)
parser.add_argument('--img', help='Evaluate an image')
parser.add_argument('--imgLabel', help='Evaluate an image and use label map')
parser.add_argument('--neighborhood', help='Set the image neighborhood, required when using --img', nargs='+', type=int)
parser.add_argument('--model', help='The deep learning model')
parser.add_argument('--outImageLabel', help='Output image with probability')
parser.add_argument('--sample', help='Evaluate a sample image')
parser.add_argument('--evaluateDir', help='Set a directory to evaluate the image sampes')

args = parser.parse_args()

pickle_file = args.pickle
num_labels = args.num_labels
outvariables = args.out
createsummary = args.summary
img = args.img
imgLabel = args.imgLabel
model = args.model
neighborhood = args.neighborhood
outImageLabel = args.outImageLabel
sample = args.sample
evaluateDir = args.evaluateDir

img_data = None
img_size = None
img_size_in = None
img_size_all = 1
img_data_label = None
img_head_label = None
evaluate_img = None

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
  elif not evaluateDir is None:
    image_files = glob.glob(os.path.join(evaluateDir, '*.nrrd'))
    evaluate_img_size = None
    if(len(image_files) > 0):
      data, head = nrrd.read(image_files[0])
      img_size = head["sizes"]
      print("Using image size: ", img_size)
    else:
      raise Exception("No .nrrd files in directory " + d)
    
    evaluate_img = np.ndarray(shape=(len(image_files), img_size[0], img_size[1], img_size[2], img_size[3]), dtype=np.float32)
    num_images = 0

    for file in image_files:
      print(file)
      try:
        data, head = nrrd.read(file)
        evaluate_img[num_images, :, :, :] = data
        num_images += 1
      except Exception as e:
        print('Could not read:', file, '- it\'s ok, skipping.', e)

    evaluate_img = evaluate_img[0:num_images, :, :]
    
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

  if not img and not sample and not evaluateDir:
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
  starter_learning_rate = 1e-8
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss + reg, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(layerout)
  if not img and not sample and not evaluateDir:
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

def getNeighborhood(image, x, y, z, neighborhood):
  neigh = np.zeros((num_labels, neighborhood[0]*2 + 1, neighborhood[1]*2 + 1, neighborhood[2]*2 + 1))
  for i in range(-neighborhood[0], neighborhood[0]+1):
    for j in range(-neighborhood[1], neighborhood[1]+1):
      for k in range(-neighborhood[2], neighborhood[2]+1):
        ni = i + neighborhood[0]
        nj = j + neighborhood[1]
        nk = k + neighborhood[2]
        neigh[0][ni][nj][nk] = image[0][x+i][y+j][z+k]
        neigh[1][ni][nj][nk] = image[1][x+i][y+j][z+k]
  return neigh

def predictImage(image, region, neighborhood, imglabel):
  neighborhood_size = (neighborhood[0]*2 + 1)*(neighborhood[1]*2 + 1)*(neighborhood[2]*2 + 1)*num_labels
  for i in range(region[0], region[1]):
    for j in range(region[2], region[3]):
      for k in range(region[4], region[5]):
        predict = True
        if not img_data_label is None:
          predict = img_data_label[i][j][k] == 8 or img_data_label[i][j][k] == 6
        if predict:
          array = getNeighborhood(image, i, j, k, neighborhood)
          prediction = predict_image.eval({keep_prob: 1, tf_image: array.reshape(-1, neighborhood_size).astype(np.float32)})
          if(prediction[0][0] == 1):
            imglabel[i][j][k] = 1
  return True

def predictImageStar(param):
  return predictImage(*param)


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  saver = tf.train.Saver()
  
  if createsummary:
    merged = tf.merge_all_summaries()
    summarydir = os.path.splitext(pickle_file)[0] + "summary"
    try:
      os.mkdir(summarydir)
    except Exception as e:
      print(e)
    train_writer = tf.train.SummaryWriter(summarydir, session.graph)

  if model:
    
    saver.restore(session, model)
    
    if img_data is not None:
      if img:

        imglabel = np.zeros((img_size_in[1], img_size_in[2], img_size_in[3]))
        region = [0, img_size_in[1], 0, img_size_in[2], 0, img_size_in[3]]
        predictImage(img_data, region, neighborhood, imglabel)

        if outImageLabel:
          nrrd.write(outImageLabel, imglabel)

        # if __name__ == '__main__':
        #   numthreads = multiprocessing.cpu_count()
        #   params = []
        #   numsplit = numthreads
        #   region_size_x = img_size_in[1]/numsplit

        #   while region_size_x < 0 and numsplit > 1:
        #     --numsplit
        #     region_size_x = img_size_in[1]/numsplit
          
        #   imglabel = np.zeros((num_labels, img_size_in[1], img_size_in[2], img_size_in[3]))

        #   for ns in range(0, numsplit):
        #     region = [region_size_x*ns, region_size_x*ns + region_size_x, 0, img_size_in[2], 0, img_size_in[3]]
        #     params.append([img_data, region, neighborhood, imglabel])

        #   p = Pool(processes=1)
        #   print(p.map(predictImageStar, params))
          
        #   if outImageLabel:
        #     nrrd.write(outImageLabel, imglabel)


      elif sample is not None:
        prediction = predict_image.eval({keep_prob: 1, tf_image: img_data.reshape(-1,img_size_all ).astype(np.float32)})
        print(prediction[0][0]);
    elif evaluateDir is not None:
      for eval_img in evaluate_img:
        prediction = predict_image.eval({keep_prob: 1, tf_image: eval_img.reshape(-1, img_size_all ).astype(np.float32)})
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

