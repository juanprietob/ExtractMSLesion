"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from six.moves import cPickle as pickle
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

num_labels = 2

def reformat(dataset):
  dataset = dataset.reshape((-1, img_size[0]*img_size[1]*img_size[2]*img_size[3])).astype(np.float32)
  return dataset
train_dataset = reformat(train_dataset)
valid_dataset = reformat(valid_dataset)
test_dataset = reformat(test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# we create 40 separable points
np.random.seed(0)
X = valid_dataset
Y = valid_labels

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-500, 3000)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

def getColor(l):
  if l == 0:
    return 'magenta'
  return 'cyan'

colors = [getColor(l) for l in Y]

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=colors)

plt.axis('tight')
plt.show()
