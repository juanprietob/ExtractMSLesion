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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
h = .02 

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

num_labels = 3

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


basedir = os.path.dirname(pickle_file);
filename = os.path.splitext(os.path.basename(pickle_file))[0]

pickle_file = os.path.join(basedir, filename + "sgdfit.pickle")
try:
  f = open(pickle_file, 'rb')
  data = pickle.load(f)
  sgdc = data['sgdc']
except Exception as e:
  print(e)
  sgdc = linear_model.SGDClassifier(n_iter=100, n_jobs=-1, loss='log')
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


pickle_file = os.path.join(basedir, filename + 'pca.pickle')

try:
  f = open(pickle_file, 'rb')
  data = pickle.load(f)
  reduced_data = data['pca']
except Exception as e:
  print('Decomposing data:')
  reduced_data = PCA(n_components=2).fit_transform(train_dataset)
  try:
    print('Saving data:')
    f = open(pickle_file, 'wb')
    save = {
      'pca': reduced_data
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


pickle_file = os.path.join(basedir, filename + '_kmeans.pickle')

try:
  f = open(pickle_file, 'rb')
  data = pickle.load(f)
  kmeans = data['kmeans']
except Exception as e:
  print('Clustering data:')
  kmeans = KMeans(init='k-means++', n_clusters=num_labels, n_init=10)
  kmeans.fit(reduced_data)
  
  try:
    f = open(pickle_file, 'wb')
    save = {
      'kmeans': kmeans
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1



plt.figure(1)
plt.clf()
# plt.imshow(reduced_data, interpolation='nearest',
#            extent=(x_min, x_max, y_min, y_max),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

def getColor(l):
  if l == 0:
    return 'magenta'
  elif l == 1:
    return 'yellow'
  return 'cyan'

colors = [getColor(l) for l in train_labels]

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors)
#Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            c=['r','b'], zorder=10)
#plt.title('K-means clustering on the digits dataset (PCA-reduced data) Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

