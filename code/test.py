import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from time import time, clock

import st_hosvd
import hodct

def load_indian_pines():
	return loadmat("../data/Indian_pines.mat")["indian_pines"]

def load_cuprite():
	return loadmat("../data/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat")["X"]

def load_botswana():
	return loadmat("../data/Botswana.mat")["Botswana"]

def plot_intensity(data):
	# Shows the cumulative intensity per pixel using grayscale
	plt.imshow(np.sum(data, axis=2))
	plt.gray()
	plt.show()

def plot_comparison(original, decompressed):
	# Shows the cumulative intensity per pixel using grayscale for two images simultaneously
	plot_intensity(original)
	plot_intensity(decompressed)

def print_difference(original, decompressed):
	diff = np.linalg.norm(original - decompressed)
	norm = np.linalg.norm(original)
	#print("Absolute error:", diff)
	#print("Initial norm:", norm)
	print("Relative error:", diff/norm)

def test_compression_ratio():
	
	data = load_cuprite()
	#compressed = st_hosvd.compress_tucker(data, 0.0294533218462, print_progress=True)
	compressed = st_hosvd.load_tucker("../data/tucker_cuprite_0.025.npz")
	decompressed = st_hosvd.decompress_tucker(compressed)
	print_difference(data, decompressed)
	st_hosvd.print_compression_rate_tucker(data, compressed)
	#plot_comparison(data, decompressed)
	#st_hosvd.plot_core_tensor_magnitudes(compressed)
	
	#st_hosvd.save_tucker(compressed, "../data/tucker_cuprite_0.025.npz")
	
	compressed_quantize = st_hosvd.compress_quantize2(compressed)
	decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_quantize2(compressed_quantize))
	print_difference(data, decompressed)
	st_hosvd.print_compression_rate_quantize2(data, compressed_quantize)
	plot_comparison(data, decompressed)
	#st_hosvd.plot_core_tensor_magnitudes(compressed)

def test_time():
	# Tests performance on random data, ignoring compression ratio
	data = np.random.rand(101, 101, 10001)
	compressed = st_hosvd.compress_tucker(data, 0, rank=(22, 21, 19), print_progress=True)
	decompressed = st_hosvd.decompress_tucker(compressed)
	print_difference(data, decompressed)
	st_hosvd.print_compression_rate_tucker(data, compressed)
	plot_comparison(data, decompressed)

def main():
	test_compression_ratio()

main()
