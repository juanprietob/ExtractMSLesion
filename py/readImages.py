import glob, os
from optparse import OptionParser
import numpy as np
import nrrd

parser = OptionParser()
parser.add_option("-d", "--dir", dest="directory",
                  help="Input directory", metavar="FILE")

(options, args) = parser.parse_args()
print options

image_files = glob.glob(os.path.join(options.directory, '*.nrrd'))

if len(image_files) > 0:

	image_data, options = nrrd.read(image_files[0])
	imgsize = options["sizes"];

	dataset = np.ndarray(shape=(len(image_files), imgsize[0], imgsize[1], imgsize[2], imgsize[3]), dtype=np.float32)
	image_index = 0

	print options

	for file in image_files:
		print(file)

		try:
			
			image_data, options = nrrd.read(file)

		  	dataset[image_index, :, :] = image_data

		  	image_index += 1

		except IOError as e:
			print('Could not read:', file, '- it\'s ok, skipping.')

	num_images = image_index
	dataset = dataset[0:num_images, :, :]
