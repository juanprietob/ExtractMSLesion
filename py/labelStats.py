import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labelStats', help='Label statistics file, with count, sizex, sizey, sizez', required=True)

args = parser.parse_args()

data = np.loadtxt(args.labelStats, delimiter=',', skiprows = 1)

data = data.transpose()

print("mean,sdev")
for row in range(len(data)):
	print(np.mean(data[row]), np.std(data[row]))

