import glob, os
import numpy as np
import nrrd
import argparse
import fnmatch
import random
import collections

from six.moves import cPickle as pickle


parser = argparse.ArgumentParser(description='Finds images in directory <dir> with extension .nrrd to create the input and output for trainning a deep learning network. Image named *_label.nrrd are ignored')
parser.add_argument('--dir', help='Input directory', required=True)
parser.add_argument('--out', help='Output filename for pickle file', default="out.pickle")
parser.add_argument("--force", help="Force pickle file generation")
parser.add_argument("--trainSize", help="Output training size dataset in percentage default 0.8 ", default=0.8, type=float)
parser.add_argument("--validSize", help="Output validation size dataset in percentage default 0.1 ", default=0.1, type=float)
parser.add_argument("--testSize", help="Output test size dataset in percentage default 0.1", default=0.1, type=float)
parser.add_argument("--readLabels", help="Read label images, put in the same directory with sufix *_label.nrrd", default=False, type=bool)

args = parser.parse_args()

rootdir = args.dir
outfilename = args.out
force = args.force
train_size = args.trainSize
valid_size = args.validSize
test_size = args.testSize
readLabels = args.readLabels

img_head = None
img_head_label = None
img_size = None
img_size_label = None

def recursive_glob(treeroot, pattern):
	print ("Reading files from: ", treeroot, pattern)
	results = []
	for base, dirs, files in os.walk(treeroot):
		results.extend(os.path.join(base, f) for f in fnmatch.filter(files, pattern))
	return results

def maybe_pickle(rootdir, dirs):
	datasets = []

	for d in dirs:

		set_filename = os.path.join(rootdir, d + '.pickle')

		if os.path.exists(set_filename) and not force:
			# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % set_filename)
			
			with open(set_filename, 'rb') as f:
				data = pickle.load(f)

			img_head = data["img_head"]
			global img_size
			img_size = img_head["sizes"]

			if(readLabels):
				img_head_label = data["img_head_label"]
				global img_size_label
				img_size_label = img_head_label["sizes"]

		else:
			print('Pickling %s.' % set_filename)

			image_files = recursive_glob(os.path.join(rootdir, d), "*.nrrd")

			if(len(image_files) > 0):
				image_data, img_head = nrrd.read(image_files[0])
				print img_head
				img_size = img_head["sizes"]
				print("Using image size: ", img_size)
			else:
				raise Exception("No .nrrd files in directory " + d)

			data = {}
			data["img_head"] = img_head

			dataset = np.ndarray(shape=tuple([len(image_files)]) + tuple(img_size), dtype=np.float32)
			num_images = 0
			datasetlabel = None

			for file in image_files:

				if(readLabels and file.find("_label.nrrd") != -1):
					continue

				print file

				try:

					image_data, img_head = nrrd.read(file)

					dataset[num_images, :, :, :] = image_data
					num_images += 1

				except Exception as e:
					print('Could not read:', file, '- it\'s ok, skipping.', e)
					continue


				if(readLabels):

					try:
							
						filelabel = file.replace(".nrrd", "_label.nrrd")
						
						print "Reading label file", filelabel

						image_data_label, img_head_label = nrrd.read(filelabel)
						img_size_label = img_head_label["sizes"]

						if(datasetlabel == None):
							data["img_head_label"] = img_head_label
							datasetlabel = np.ndarray(shape=(len(image_files), img_size_label[0], img_size_label[1], img_size_label[2]), dtype=np.float32)

						datasetlabel[num_images, :, :, :] = image_data_label

					except Exception as e:

						print('Could not read:', filelabel, '- it\'s not ok, terminating')
						quit()


			dataset = dataset[0:num_images, :]
			data["dataset"] = dataset

			if(readLabels):
				datasetlabel = datasetlabel[0:num_images, :, :]
				data["datasetlabel"] = datasetlabel

			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)

		datasets.append(data)

	return datasets

print ('Reading samples from:', rootdir)
dirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]

alldatasets = maybe_pickle(rootdir, dirs)

def make_arrays(nb_rows):
  if nb_rows:
    dataset = np.ndarray(tuple([nb_rows]) + tuple(img_size), dtype=np.float32)
    if(readLabels):
    	labels = np.ndarray(tuple([nb_rows]) + tuple(img_size_label), dtype=np.float32)
    else:
    	labels = np.ndarray(tuple([nb_rows]), dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def maybe_randomize(alldataset, outfilename):

	if(len(alldataset) == 0):
		raise Exception("No datasets to randomize")

	if os.path.exists(outfilename) and not force:
		# You may override by setting force=True.
		print('%s already present - Skipping shuffling.' % outfilename)
	else:

		print('Shuffling and creating train, valid and test sets')

		size = 0
		for data in alldataset:
			size+=len(data["dataset"])

		print('Total number of samples= ', size)

		trainsize = int(size*train_size)
		validsize = int(size*valid_size)
		testsize = int(size*test_size)

		number_of_labels=len(alldataset)
		train_dataset, train_labels = make_arrays(trainsize)
		valid_dataset, valid_labels = make_arrays(validsize)
		test_dataset, test_labels = make_arrays(testsize)

		print ("Generating trainsize, validsize, testsize", trainsize, validsize, testsize)

		r = range(size)
		random.shuffle(r)#we randomize all the index from the total size of the dataset

		for i in range(trainsize):
			random_index = r[i]
			dataset_index = 0
			dataset_length=0
			label=0
			for l in range(number_of_labels):#We search the current index in all the labels
				current_dataset_length=len(alldataset[l]["dataset"])
				dataset_length+=current_dataset_length
				if(random_index < dataset_length):#When the random_index is less then it is in the interval for the given label
					label=l
					dataset_index = int(random_index*current_dataset_length/size)
					break
			train_dataset[i] = alldataset[l]["dataset"][dataset_index]
			if(readLabels):
				train_labels[i] = alldataset[l]["datasetlabel"][dataset_index]
			else:
				train_labels[i] = label

		if(readLabels == False):
			print ("Train labels distribution: ", collections.Counter(train_labels))

		for i in range(validsize):
			random_index = r[i + trainsize]
			dataset_index = 0
			dataset_length=0
			label=0
			for l in range(number_of_labels):
				current_dataset_length=len(alldataset[l]["dataset"])
				dataset_length+=current_dataset_length
				if(random_index < dataset_length):
					label=l
					dataset_index = int(random_index*current_dataset_length/size)
					break
			valid_dataset[i] = alldataset[l]["dataset"][dataset_index]
			if(readLabels):
				valid_labels[i] = alldataset[l]["datasetlabel"][dataset_index]
			else:
				valid_labels[i] = label


		if(readLabels == False):
			print ("Valid labels distribution: ", collections.Counter(valid_labels))

		for i in range(0, testsize):
			random_index = r[i + trainsize + validsize]
			dataset_index = 0
			dataset_length=0
			label=0
			for l in range(number_of_labels):
				current_dataset_length=len(alldataset[l]["dataset"])
				dataset_length+=current_dataset_length
				if(random_index < dataset_length):
					label=l
					dataset_index = int(random_index*current_dataset_length/size)
					break
			test_dataset[i] = alldataset[l]["dataset"][dataset_index]
			if(readLabels):
				test_labels[i] = alldataset[l]["datasetlabel"][dataset_index]
			else:
				test_labels[i] = label

		if(readLabels == False):
			print ("Test labels distribution: ", collections.Counter(test_labels))
		

		try:
		  f = open(os.path.join(rootdir, outfilename), 'wb')
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
		  print('Unable to save data to', outfilename, ':', e)
		  raise
	return outfilename


pickle_file = maybe_randomize(alldatasets, outfilename)

# In[ ]:

statinfo = os.stat(os.path.join(rootdir, pickle_file))
print('Compressed pickle size:', statinfo.st_size)