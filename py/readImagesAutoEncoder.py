import glob, os
import numpy as np
import nrrd
import argparse
import fnmatch
import random

from six.moves import cPickle as pickle


parser = argparse.ArgumentParser(description='Finds images in directory <dir> named like <pattern>.nrrd and the corresponding <prefix><pattern>.nrrd in order to create the input and output for trainning a deep learning network.')
parser.add_argument('--dir', help='Input directory', required=True)
parser.add_argument('--pattern', help='Filename pattern. Finds this pattern recursively in the input directory. ex. --pattern 60-124-120', required=True)
parser.add_argument('--prefix', help='Filename prefix. Differentiates the labeled image (output) from the input. ex. --prefix pvec', required=True)
parser.add_argument('--outdir', help='Output directory for pickle file')
parser.add_argument("--force", help="Force pickle file generation")
parser.add_argument("--trainSize", help="Output training size dataset", default=0.8, type=float)
parser.add_argument("--validSize", help="Output validation size dataset", default=0.1, type=float)
parser.add_argument("--testSize", help="Output test size dataset", default=0.1, type=float)

args = parser.parse_args()

rootdir = args.dir
pattern = args.pattern
prefix = args.prefix
outdir = args.outdir
force = args.force
train_size = args.trainSize
valid_size = args.validSize
test_size = args.testSize
img_head = None
img_head_label = None
img_size = None
img_size_label = None


if not outdir:
	outdir = rootdir

def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        results.extend(os.path.join(base, f) for f in fnmatch.filter(files, pattern))
    return results

def maybe_pickle(rootdir, outdir, pattern):
	dataset_names = []	

	set_filename = os.path.join(outdir, pattern + '_data.pickle')
	if os.path.exists(set_filename) and not force:
		# You may override by setting force=True.
		print('%s already present - Skipping pickling.' % set_filename)
		with open(set_filename, 'rb') as f:
			data = pickle.load(f)
	else:
		print('Pickling %s.' % set_filename)

		image_files = recursive_glob(rootdir, pattern + ".nrrd")

		if(len(image_files) > 0):
			image_data, img_head = nrrd.read(image_files[0])
			print img_head
			img_size = img_head["sizes"]
			print("Using image size: ", img_size)

			label_file = os.path.join(os.path.dirname(image_files[0]), prefix + os.path.basename(image_files[0]))
			image_data_label, img_head_label = nrrd.read(label_file)
			print img_head_label
			img_size_label = img_head_label["sizes"]
			print("Using image label size: ", img_size_label)
		else:
			raise Exception("No .nrrd files in directory " + d)

		data = {}
		data["img_head"] = img_head
		data["img_head_label"] = img_head_label

		dataset = np.ndarray(shape=tuple([len(image_files)]) + tuple(img_size), dtype=np.float32)
		dataset_label = np.ndarray(shape=tuple([len(image_files)]) + tuple(img_size_label), dtype=np.float32)
		num_images = 0

		for file in image_files:

			label_file = os.path.join(os.path.dirname(file), prefix + os.path.basename(file))
				
			print(file)
			print(label_file)

			try:
				
				image_data, img_head = nrrd.read(file)
				image_data_label, img_head_label = nrrd.read(label_file)

				dataset[num_images, :, :, :] = image_data
				dataset_label[num_images, :, :, :] = image_data_label
				num_images += 1

			except Exception as e:
				print('Could not read:', file, '- it\'s ok, skipping.', e)

		dataset = dataset[0:num_images, :]
		dataset_label = dataset_label[0:num_images, :]
		data["dataset"] = dataset
		data["dataset_label"] = dataset_label

		try:
			with open(set_filename, 'wb') as f:
				pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', set_filename, ':', e)

	return data


all_dataset = maybe_pickle(rootdir, outdir, pattern)


def make_arrays(nb_rows, imgsize, imgsizelabel):
  if nb_rows:
    dataset = np.ndarray(tuple([nb_rows]) + tuple(imgsize), dtype=np.float32)
    labels = np.ndarray(tuple([nb_rows]) + tuple(imgsizelabel), dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def maybe_randomize(alldataset):

	pickle_file = os.path.join(outdir, pattern + '.pickle')

	if os.path.exists(pickle_file) and not force:
		# You may override by setting force=True.
		print('%s already present - Skipping shuffling.' % pickle_file)
	else:

		print('Shuffling and creating train, valid and test sets')

		dataset = alldataset["dataset"]
		dataset_label = alldataset["dataset_label"]
		img_head = alldataset["img_head"]
		img_size = img_head["sizes"]
		img_head_label = alldataset["img_head_label"]
		img_size_label = img_head_label["sizes"]

		size = len(dataset)
		trainsize = int(size*train_size)
		validsize = int(size*valid_size)
		testsize = int(size*test_size)

		train_dataset, train_labels = make_arrays(trainsize, img_size, img_size_label)
		valid_dataset, valid_labels = make_arrays(validsize, img_size, img_size_label)
		test_dataset, test_labels = make_arrays(testsize, img_size, img_size_label)
		

		r = range(size)
		random.shuffle(r)
		for i in range(0, trainsize):
			random_index = r[i]
			train_dataset[i] = dataset[random_index]
			train_labels[i] = dataset_label[random_index]

		for i in range(0, validsize):
			random_index = r[i + trainsize]
			train_dataset[i] = dataset[random_index]
			train_labels[i] = dataset_label[random_index]

		for i in range(0, testsize):
			random_index = r[i + trainsize + validsize]
			test_dataset[i] = dataset[random_index]
			test_labels[i] = dataset_label[random_index]
		

		try:
		  f = open(os.path.join(rootdir, pickle_file), 'wb')
		  save = {
		    'train_dataset': train_dataset,
		    'train_labels': train_labels,
		    'valid_dataset': valid_dataset,
		    'valid_labels': valid_labels,
		    'test_dataset': test_dataset,
		    'test_labels': test_labels,
		    'img_head': img_head,
		    'img_head_label': img_head_label
		    }
		  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		  f.close()
		except Exception as e:
		  print('Unable to save data to', pickle_file, ':', e)
		  raise
	return pickle_file


pickle_file = maybe_randomize(all_dataset)
# In[ ]:

statinfo = os.stat(os.path.join(rootdir, pickle_file))
print('Compressed pickle size:', statinfo.st_size)
