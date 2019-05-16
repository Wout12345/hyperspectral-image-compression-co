import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imsave
from math import floor, sqrt

# Limit memory usage
# Code from https://stackoverflow.com/questions/41105733/limit-ram-usage-to-python-program
import resource

def limit_memory():
	soft, hard = resource.getrlimit(resource.RLIMIT_AS)
	resource.setrlimit(resource.RLIMIT_AS, (round(get_memory() * 1024 * 0.95), hard))

def get_memory():
	with open("/proc/meminfo", "r") as mem:
		free_memory = 0
		for i in mem:
			sline = i.split()
			if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
				free_memory += int(sline[1])
	return free_memory

limit_memory()

# End of memory limit

def custom_norm(data):
	
	# Calculates Frobenius norm of data but without copying the whole matrix to float32 like numpy.linalg.norm does
	sq_total = 0
	buffer_size = 128*1047*224*4 # In bytes
	block_size = max(1, floor((buffer_size/(data.size*data.itemsize))*data.shape[0]))
	for i in range(0, data.shape[0], block_size):
		sq_total += np.linalg.norm(data[i:i + block_size, :, :])**2
	return sqrt(sq_total)

# Loaders

def load_indian_pines():
	return loadmat("../data/Indian_pines.mat")["indian_pines"]

def load_cuprite():
	return filter_bands(loadmat("../data/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat")["X"], ((4, 106), (114, 152), (169, 219)))

def load_botswana():
	return loadmat("../data/Botswana.mat")["Botswana"]

def load_pavia():
	return loadmat("../data/Pavia.mat")["pavia"][:1089, :676, :] # Cut spatial dimensions to perfect squares

def load_mauna_kea_raw():
	lines = 2816 # Full file: 2816 lines
	samples = 1047 # Full file: 1047 samples
	channels = 224
	data = np.transpose(np.fromfile("../data/mauna_kea_raw", dtype="float32", count=lines*samples*channels).reshape(lines, channels, samples), axes=(0, 2, 1))
	return data

def load_mauna_kea():
	lines = 2704
	samples = 729
	data = np.fromfile("../data/mauna_kea_preprocessed", dtype="uint16")
	return data.reshape(lines, samples, data.size//(lines*samples))

# End of loaders

def filter_bands(raw_data, ranges_to_keep):
	# ranges_to_keep is a list of tuples (start, end) (inclusive, exclusive)
	bands = 0
	for start, end in ranges_to_keep:
		bands += (end - start)
	data = np.empty((raw_data.shape[0], raw_data.shape[1], bands), dtype=raw_data.dtype)
	i = 0
	for start, end in ranges_to_keep:
		diff = end - start
		data[:, :, i:i + diff] = raw_data[:, :, start:end]
		i += diff
	return data

def plot_intensity(data):
	# Shows the cumulative intensity per pixel using grayscale
	plt.imshow(np.sum(data, axis=2))
	plt.gray()
	plt.show()

def plot_comparison(original, decompressed):
	# Shows the cumulative intensity per pixel using grayscale for two images simultaneously
	fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, subplot_kw=dict(adjustable="box-forced",aspect=original.shape[0]/original.shape[1]))
	axes[0].imshow(np.sum(original, axis=2), cmap="gray")
	axes[1].imshow(np.sum(decompressed, axis=2), cmap="gray")
	axes[2].imshow(np.sqrt(np.sqrt(np.sum((original - decompressed)**2, axis=2))), cmap="gray")
	plt.show()

def rel_error(original, decompressed):
	return custom_norm(original - decompressed)/custom_norm(original)

def print_difference(original, decompressed):
	print("Relative error: %s\n"%rel_error(original, decompressed))

def print_bands():
	
	data = load_cuprite()
	bands = data.shape[2]
	steps = bands
	print("Bands:", bands)
	
	# Determine max value
	max_value = 0
	for i in range(steps):
		start = int(floor(i/steps*bands))
		end = int(floor((i + 1)/steps*bands))
		image = np.sum(data[:, :, start:end], axis=2)
		max_value = max(max_value, np.amax(image))
	
	# Print images
	for i in range(steps):
		start = int(floor(bands/steps*i))
		end = int(floor(bands/steps*(i + 1)))
		image = np.sum(data[:, :, start:end], axis=2)
		imsave("../data/bands/cuprite_bands_%s-%s.png"%(start + 0, end), np.rint(image/max_value*255).astype(int))

def plot_intensities(data):
	
	intensities = np.sum(data, axis=(0, 1))
	plt.plot(range(data.shape[2]), intensities, "b")
	plt.show()
