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
parser.add_argument("--trainSize", help="Output training size dataset in percentage default 0.8 ", default=0.8, type=float)
parser.add_argument("--validSize", help="Output validation size dataset in percentage default 0.1 ", default=0.1, type=float)
parser.add_argument("--testSize", help="Output test size dataset in percentage default 0.1", default=0.1, type=float)
parser.add_argument("--readLabels", help="Read label images, put in the same directory with sufix *_label.nrrd", default=False, type=bool)
parser.add_argument("--extractLabel", help="Threshold the labels. readLabels must be enabled as well. If different than -1, the resulting labels will be binary, 1 for the extractLabel, 0 for the rest", default=-1, type=int, nargs='+')
parser.add_argument("--sampleFile", help="Instead of recursing a directory tree, provide a txt file with the samples name. Should be located besides each directory class, e.x., If directory name is wm then the file should be wm.txt.", default=False, type=bool)
parser.add_argument("--writeRandomImage", help="While performing the sanity checks if set to True, it will write random images from train, valid and test datasets", default=False, type=bool)
args = parser.parse_args()

rootdir = args.dir
outfilename = args.out
train_size = args.trainSize
valid_size = args.validSize
test_size = args.testSize
readLabels = args.readLabels
extractLabel = args.extractLabel
sampleFile = args.sampleFile
writeRandomImage = args.writeRandomImage

img_head = None
img_head_label = None
img_size = None
img_size_label = None

def threshold_labels(labels, extract_label):
    for l in np.nditer(labels, op_flags=['readwrite']):
    	for li in range(len(extract_label)):
	        if(l == extract_label[li]):
	            l[...] = li + 1
	        else:
	            l[...] = 0
    return labels

def recursive_glob(treeroot, pattern):
	print ("Reading files from: ", treeroot, pattern)
	results = []
	for base, dirs, files in os.walk(treeroot):
		results.extend(os.path.join(base, f) for f in fnmatch.filter(files, pattern))
	return results

def readSampleFile(sampleFile):
	print ("Reading from sample file: ", sampleFile)
	results = []
	
	try:
		text_file = open(sampleFile, "r")
		results = text_file.read().splitlines()
		text_file.close()
	except Exception as e:
		print('Could not read:', sampleFile)

	return results

def maybe_pickle(rootdir, dirs):
	datasets = []

	for d in dirs:

		set_filename = os.path.join(rootdir, d + '.pickle')

		if os.path.exists(set_filename):
			
			print('%s already present - Skipping pickling.' % set_filename)
			
			with open(set_filename, 'rb') as f:
				data = pickle.load(f)
			global img_head
			img_head = data["img_head"]
			global img_size
			img_size = img_head["sizes"]

			if(readLabels):
				global img_head_label
				img_head_label = data["img_head_label"]
				global img_size_label
				img_size_label = img_head_label["sizes"]

			f.close()
		else:
			print('Pickling %s.' % set_filename)

			if(sampleFile):
				image_files = readSampleFile(os.path.join(rootdir, d + ".txt"))
			else:
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

					if(readLabels):

						try:
								
							filelabel = file.replace(".nrrd", "_label.nrrd")
							
							print "Reading label file", filelabel

							image_data_label, img_head_label = nrrd.read(filelabel)
							img_size_label = img_head_label["sizes"]

							if(datasetlabel is None):
								data["img_head_label"] = img_head_label
								datasetlabel = np.ndarray(shape=(len(image_files), img_size_label[0], img_size_label[1], img_size_label[2]), dtype=np.float32)

							datasetlabel[num_images, :, :, :] = image_data_label

						except Exception as e:
							print('Could not read:', filelabel, '- it\'s not ok, removing previous image')
							num_images -= 1

					num_images += 1

				except Exception as e:
					print('Could not read:', file, '- it\'s ok, skipping.', e)
					continue

			dataset = dataset[0:num_images, :]
			data["dataset"] = dataset

			if(readLabels):
				datasetlabel = datasetlabel[0:num_images, :, :]
				data["datasetlabel"] = datasetlabel

			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(data, f)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)

		datasets.append(data)

	return datasets

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


		if(readLabels and extractLabel != -1):
			print("Threshold train labels...")
			train_labels = threshold_labels(train_labels, extractLabel)
			print("Threshold valid labels...")
			valid_labels = threshold_labels(valid_labels, extractLabel)
			print("Threshold test labels...")
			test_labels = threshold_labels(test_labels, extractLabel)		

		try:
			print("Saving merged datasets:", outfilename)
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
		  	pickle.dump(save, f)
		  	f.close()
		  	sanity_checks(save)
		except Exception as e:
			print('Unable to save data to', outfilename, ':', e)
			raise


def sanity_checks(dataset):
	print("Sannity checks...")
	print("Train size (data length, labels length, labels sum): ", len(dataset["train_dataset"]), len(dataset["train_labels"]), np.sum(dataset["train_labels"]))
	print("Valid size (data length, labels length, labels sum): ", len(dataset["valid_dataset"]), len(dataset["valid_labels"]), np.sum(dataset["valid_labels"]))
	print("Test size (data length, labels length, labels sum): ", len(dataset["test_dataset"]), len(dataset["test_labels"]), np.sum(dataset["test_labels"]))
	print("Image head:", dataset["img_head"])
	print("Image head:", dataset["img_head_label"])

	if writeRandomImage:

		head = dataset["img_head"]
		
		index_train = random.randint(0, len(dataset["train_dataset"]))
		img = dataset["train_dataset"][index_train]
		nrrd.write(path.join(path.dirname(outfilename), "train.nrrd"), img, head)
		index_label = random.randint(0, len(dataset["valid_dataset"]))
		img = dataset["valid_dataset"][index_label]
		nrrd.write(path.join(path.dirname(outfilename), "valid.nrrd"), img, head)
		index_test = random.randint(0, len(dataset["test_dataset"]))
		img = dataset["test_dataset"][index_test]
		nrrd.write(path.join(path.dirname(outfilename), "test.nrrd"), img, head)


		if(readLabels):
			img_label = dataset["train_labels"][index_train]
			nrrd.write(path.join(path.dirname(outfilename), "train_label.nrrd"), img_label, head)
			img_label = dataset["valid_labels"][index_label]
			nrrd.write(path.join(path.dirname(outfilename), "valid_label.nrrd"), img_label, head)
			img_label = dataset["test_labels"][index_test]
			nrrd.write(path.join(path.dirname(outfilename), "test_label.nrrd"), img_label, head)
		else:
			print("The train image corresponds to class", dataset["train_labels"][index_train])
			print("The valid image corresponds to class", dataset["valid_labels"][index_valid])
			print("The test image corresponds to class", dataset["test_labels"][index_test])



if os.path.exists(outfilename):
	print('%s already present - Skipping reading samples and shuffling.' % outfilename)
	
	with open(outfilename, 'rb') as f:
		data = pickle.load(f)
	sanity_checks(data)
	f.close()
else:

	print ('Reading samples from:', rootdir)
	dirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]

	alldatasets = maybe_pickle(rootdir, dirs)

	maybe_randomize(alldatasets, outfilename)

	# In[ ]:

statinfo = os.stat(os.path.join(rootdir, outfilename))
print('Compressed pickle size:', statinfo.st_size)