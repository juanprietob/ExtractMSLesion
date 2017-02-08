from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.', required=True)

args = parser.parse_args()

pickle_file = args.pickle

print("Reading:", pickle_file)

f = open(pickle_file, 'rb')
data = pickle.load(f)
train_dataset = data["train_dataset"]
train_labels = data["train_labels"]
valid_dataset = data["valid_dataset"]
valid_labels = data["valid_labels"]
test_dataset = data["test_dataset"]
test_labels = data["test_labels"]
img_head = data["img_head"]
img_head_label = None
if "img_head_label" in data:
	img_head_label = data["img_head_label"]


datasplit = {}
datasplit["train_dataset"] = train_dataset
datasplit["train_labels"] = train_labels
datasplit["img_head"] = img_head
if img_head_label is not None:
	datasplit["img_head_label"] = img_head_label

print("Split train...")
f = open(os.path.splitext(pickle_file)[0] + "_train.pickle", 'wb')
try:
	pickle.dump(datasplit, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	f.close()
	print("Can't save compressed file, trying without compression")
	f = open(os.path.splitext(pickle_file)[0] + "_train.pickle", 'wb')
	pickle.dump(datasplit, f)
	f.close()


print("Split valid and test...")
datasplit = {}
datasplit["valid_dataset"] = valid_dataset
datasplit["valid_labels"] = valid_labels
datasplit["test_dataset"] = test_dataset
datasplit["test_labels"] = test_labels
datasplit["img_head"] = img_head
if img_head_label is not None:
	datasplit["img_head_label"] = img_head_label

f = open(os.path.splitext(pickle_file)[0] + "_valid.pickle", 'wb')
try:
	pickle.dump(datasplit, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	f.close()
	print("Can't save compressed file, trying without compression")
	f = open(os.path.splitext(pickle_file)[0] + "_valid.pickle", 'wb')
	pickle.dump(datasplit, f)
	f.close()

