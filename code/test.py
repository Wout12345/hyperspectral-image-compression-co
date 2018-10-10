import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import st_hosvd

def load_indian_pines():
	return loadmat("../data/Indian_pines.mat")["indian_pines"]

def plot_intensity(data):
	# Shows the cumulative intensity per pixel using grayscale
	plt.imshow(np.sum(data, axis=2))
	plt.gray()
	plt.show()

def plot_comparison(data1, data2):
	# Shows the cumulative intensity per pixel using grayscale for two images simultaneously
	plt.imshow(np.sum(data1, axis=2))
	plt.imshow(np.sum(data2, axis=2))
	plt.gray()
	plt.show()

def main():
	data = load_indian_pines()
	compressed = st_hosvd.compress(data)
	decompressed = st_hosvd.decompress(compressed)
	plot_comparison(decompressed)

main()
