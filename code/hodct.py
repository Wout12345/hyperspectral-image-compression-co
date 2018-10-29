import numpy as np
import scipy.fftpack
from pylab import imshow, show, gray
import scipy.ndimage
import scipy.misc

def compress(data, relative_target_error, print_progress=False):
	
	"""image = np.sum(scipy.ndimage.imread("../data/lighthouse.png"), axis=2)
	imshow(image)
	gray()
	show()
	dct = scipy.fftpack.dct(image, norm="ortho", axis=0)
	dct = np.delete(dct, np.s_[int(dct.shape[0]*0.8):], axis=0)
	dct = scipy.fftpack.dct(dct, norm="ortho", axis=1)
	dct = np.delete(dct, np.s_[int(dct.shape[1]*0.8):], axis=1)
	approx = scipy.fftpack.idct(scipy.fftpack.idct(dct, norm="ortho", axis=1, n=image.shape[1]), norm="ortho", axis=0, n=image.shape[0])
	imshow(approx)
	show()
	print(np.linalg.norm(approx - image))
	print(np.linalg.norm(image))
	print(np.linalg.norm(approx - image)/np.linalg.norm(image))"""
	
	sizes = data.shape
	compression_sizes = [int(data.shape[0]*0.99), int(data.shape[1]*0.99), int(data.shape[2]*0.13)]
	mode_order = 2, 1, 0
	for mode in mode_order:
		data = scipy.fftpack.dct(data, norm="ortho", axis=mode)
		data = np.delete(data, np.s_[compression_sizes[mode]:], axis=mode)
	
	return sizes, data

def decompress(compressed):
	
	sizes, data = compressed
	mode_order = 0, 1, 2
	for mode in mode_order:
		data = scipy.fftpack.idct(data, norm="ortho", axis=mode, n=sizes[mode])
	
	return data
