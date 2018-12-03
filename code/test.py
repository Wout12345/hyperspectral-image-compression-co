import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from time import time, clock

import st_hosvd
#import blockwise_st_hosvd
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
	fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
	axes[0].imshow(np.sum(original, axis=2), cmap="gray")
	axes[1].imshow(np.sum(decompressed, axis=2), cmap="gray")
	axes[2].imshow(np.sqrt(np.sum((original - decompressed)**2, axis=2)), cmap="gray")
	plt.show()

def print_difference(original, decompressed):
	diff = np.linalg.norm(original - decompressed)
	norm = np.linalg.norm(original)
	print("Relative error: %s\n"%(diff/norm))

def test_compression_ratio():
	
	data = load_cuprite()
	compressed = st_hosvd.compress_tucker(data, 0.025, print_progress=True, rank=(501, 596, 12))#(501, 596, 12))
	#st_hosvd.save_tucker(compressed, "../data/tucker_cuprite_0.025.npz")
	#compressed = st_hosvd.load_tucker("../data/tucker_cuprite_0.025.npz")
	decompressed = st_hosvd.decompress_tucker(compressed)
	st_hosvd.print_compression_rate_tucker(data, compressed)
	print_difference(data, decompressed)
	#plot_comparison(data, decompressed)
	
	"""method = 2
	if method == 2:
		compressed_quantized = st_hosvd.compress_quantize2(compressed)
		st_hosvd.print_compression_rate_quantize2(data, compressed_quantized)
		decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_quantize2(compressed_quantized))
		print_difference(data, decompressed)
		#plot_comparison(data, decompressed)
	else:
		compressed_quantized = st_hosvd.compress_quantize3(compressed)
		st_hosvd.print_compression_rate_quantize3(data, compressed_quantized)
		print_difference(data, st_hosvd.decompress_tucker(st_hosvd.decompress_quantize3(compressed_quantized)))"""

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
