import glob, os
import numpy as np
import nrrd
import argparse
from six.moves import cPickle as pickle


parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='Input directory', required=True)
#parser.add_argument("--force", help="Force pickle file generation")
parser.add_argument("--force", help="Force pickle file generation")
parser.add_argument('-s','--imgsize', nargs=4, help='Set the image size, [numberOfComponents, sizex, sizey, sizez]', default=[2, 19, 19, 3], type=int)
parser.add_argument("--trainSize", help="Output training size dataset", default=200000, type=int)
parser.add_argument("--validSize", help="Output validation size dataset", default=10000, type=int)
parser.add_argument("--testSize", help="Output test size dataset", default=10000, type=int)

args = parser.parse_args()

imgsize = args.imgsize;
rootdir = args.dir
force = args.force
train_size = args.trainSize
valid_size = args.validSize
test_size = args.testSize
img_head = {}

dirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]

def maybe_pickle(dirs, force):
	dataset_names = []

	for d in dirs:

		set_filename = os.path.join(rootdir, d + '.pickle')
		if os.path.exists(set_filename) and not force:
			# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % set_filename)
		else:
			print('Pickling %s.' % set_filename)

			image_files = glob.glob(os.path.join(rootdir, d, '*.nrrd'))

			dataset = np.ndarray(shape=(len(image_files), imgsize[0], imgsize[1], imgsize[2], imgsize[3]), dtype=np.float32)
			num_images = 0

			for file in image_files:
				print(file)

				try:
					
					image_data, img_head = nrrd.read(file)

					dataset[num_images, :, :, :] = image_data
					num_images += 1

				except IOError as e:
					print('Could not read:', file, '- it\'s ok, skipping.')

			dataset = dataset[0:num_images, :, :]

			try:
				with open(set_filename, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print('Unable to save data to', set_filename, ':', e)

		dataset_names.append(set_filename)

	return dataset_names

train_datasets = maybe_pickle(dirs, force)


def make_arrays(nb_rows, imgsize):
  if nb_rows:
    dataset = np.ndarray((nb_rows, imgsize[0], imgsize[1], imgsize[2], imgsize[3]), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, imgsize)
  train_dataset, train_labels = make_arrays(train_size, imgsize)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
           

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
#_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
#test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


nrrd.write("out.nrrd", train_dataset[10], img_head)
