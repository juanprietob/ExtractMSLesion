from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn import neighbors, linear_model
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.', required=True)

args = parser.parse_args()

pickle_file = args.pickle
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

num_labels = 2

in_depth = img_size[3] #zdim
in_height = img_size[2] #ydim
in_width = img_size[1] #xdim
num_channels = img_size[0] #num channels

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

(samples, depth, height, width, num_channels) = train_dataset.shape
train_dataset = np.reshape(train_dataset,(samples,depth*height*width*num_channels))[0:samples]

(samples, depth, height, width, num_channels) = valid_dataset.shape
valid_dataset = np.reshape(valid_dataset,(samples,depth*height*width*num_channels))[0:samples]

(samples, depth, height, width, num_channels) = test_dataset.shape
test_dataset = np.reshape(test_dataset,(samples,depth*height*width*num_channels))[0:samples]

print('Training set reshaped', train_dataset.shape, train_labels.shape)
print('Validation set reshaped', valid_dataset.shape, valid_labels.shape)
print('Validation set reshaped', test_dataset.shape, test_labels.shape)


pickle_file = 'ms_wm_sgdfit.pickle'
try:
  f = open(pickle_file, 'rb')
  data = pickle.load(f)
  sgdc = data['sgdc']
except Exception as e:
  print(e)
  sgdc = linear_model.SGDClassifier(n_iter=25, n_jobs=-1, loss='log')
  sgdc.fit(train_dataset, train_labels)

try:
  f = open(pickle_file, 'wb')
  save = {
    'sgdc': sgdc
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

print('SGDC test score: %f' % sgdc.score(test_dataset, test_labels))
print('SGDC validation score: %f' % sgdc.score(valid_dataset, valid_labels))