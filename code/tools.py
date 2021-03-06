import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from imageio import imwrite
from math import floor, sqrt

import st_hosvd

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
	return st_hosvd.custom_norm(data)

# Loaders

def load_indian_pines():
	return loadmat("../data/Indian_pines.mat")["indian_pines"]

def load_indian_pines_cropped():
	return load_indian_pines()[:144, :144, :]

def load_cuprite():
	return filter_bands(loadmat("../data/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat")["X"], ((4, 106), (114, 152), (169, 219)))

def load_cuprite_cropped():
	return load_cuprite()[:484, :576, :] # 22**2, 24**2

def load_cuprite_radiance_cropped():
	lines = 256 # Full file: 256 lines
	samples = 256 # Full file: 256 samples
	channels = 128
	data = np.fromfile("../data/Cuprite_radiance_cropped", dtype="int16").reshape(lines, samples, channels)
	return data

def load_moffet_field_radiance_cropped():
	lines = 256 # Full file: 256 lines
	samples = 256 # Full file: 256 samples
	channels = 128
	data = np.fromfile("../data/Moffet_Field_radiance_cropped", dtype="int16").reshape(lines, samples, channels)
	return data

def load_botswana():
	return loadmat("../data/Botswana.mat")["Botswana"][:, :, 2:]

def load_pavia():
	return loadmat("../data/Pavia.mat")["pavia"][:1089, :676, :] # Cut spatial dimensions to perfect squares

def load_mauna_kea_raw():
	lines = 2816 # Full file: 2816 lines
	samples = 1047 # Full file: 1047 samples
	channels = 224
	data = np.transpose(np.fromfile("../data/mauna_kea_raw", dtype="float32", count=lines*samples*channels).reshape(lines, channels, samples), axes=(0, 2, 1))
	return data

def load_mauna_kea():
	lines = 2704 # 52**2
	samples = 729 # 33**2
	data = np.fromfile("../data/mauna_kea_preprocessed", dtype="uint16")
	return data.reshape(lines, samples, data.size//(lines*samples))

def load_moffett_field_cropped():
	lines = 512
	samples = 512
	channels = 224
	data = np.fromfile("../../tensor-decompositions/data/hyperspectral_images/moffett_field_cropped", dtype="int16").reshape((lines, samples, channels))
	return data

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

def rel_error(original, decompressed, preserve_decompressed=True):
	return st_hosvd.rel_error(original, decompressed, preserve_decompressed=preserve_decompressed)

def print_difference(original, decompressed):
	print("Relative error: %s\n"%rel_error(original, decompressed))

def print_cuprite_bands(steps=None, path="../data/bands/cuprite_bands"):
	
	data = load_cuprite()
	bands = data.shape[2]
	if steps is None:
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
		start = int(round(bands/steps*i))
		end = int(round(bands/steps*(i + 1)))
		image = np.sum(data[:, :, start:end], axis=2)
		imwrite(path + "_%s-%s.png"%(start, end), np.rint(image/max_value*255).astype(int))

def plot_intensities(data):
	
	intensities = np.sum(data, axis=(0, 1))
	plt.plot(range(data.shape[2]), intensities, "b")
	plt.show()

def crop_cuprite_radiance():
	lines = 2776 # Full file: 2776 lines
	samples = 754 # Full file: 754 samples
	channels = 224
	data = np.reshape(np.fromfile("../data/Cuprite_radiance", dtype="int16", count=lines*samples*channels).byteswap(), (lines, samples, channels))
	data = data[:256, :256, :128]
	data.tofile("../data/Cuprite_radiance_cropped")

def crop_moffet_field_radiance():
	lines = 1924 # Full file: 1924 lines
	samples = 753 # Full file: 753 samples
	channels = 224
	data = np.reshape(np.fromfile("../data/Moffet_Field_radiance", dtype="int16", count=lines*samples*channels).byteswap(), (lines, samples, channels))
	data = data[:256, :256, :128]
	data.tofile("../data/Moffet_Field_radiance_cropped")
	
