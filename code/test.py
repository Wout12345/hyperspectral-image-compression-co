import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functools import reduce

import st_hosvd

def load_indian_pines():
	return loadmat("../data/Indian_pines.mat")["indian_pines"]

def plot_intensity(data):
	# Shows the cumulative intensity per pixel using grayscale
	plt.imshow(np.sum(data, axis=2))
	plt.gray()
	plt.show()

def plot_comparison(original, compressed):
	# Shows the cumulative intensity per pixel using grayscale for two images simultaneously
	plt.imshow(np.sum(original, axis=2))
	plt.gray()
	plt.show()
	plt.imshow(np.sum(compressed, axis=2))
	plt.gray()
	plt.show()

def print_difference(original, compressed):
	diff = np.linalg.norm(original - compressed)
	norm = np.linalg.norm(original)
	print("Absolute error:", diff)
	print("Initial norm:", norm)
	print("Relative error:", abs(diff)/norm)

def print_compression_rate(original, compressed):
	original_size = original.ndim + reduce((lambda x, y: x * y), original.shape)
	factor_matrices, core_tensor = compressed
	compressed_size = 2*original.ndim + sum([factor_matrix.shape[0]*factor_matrix.shape[1] for factor_matrix in factor_matrices]) + reduce((lambda x, y: x * y), core_tensor.shape)
	print("Sizes in number of floats:")
	print("Original shape:", original.shape, "\toriginal size:", original_size)
	print("Compressed shape:", core_tensor.shape, "\tcompressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

def main():
	data = load_indian_pines()
	compressed = st_hosvd.compress(data)
	decompressed = st_hosvd.decompress(compressed)
	print_difference(data, decompressed)
	print_compression_rate(data, compressed)
	plot_comparison(data, decompressed)

main()
