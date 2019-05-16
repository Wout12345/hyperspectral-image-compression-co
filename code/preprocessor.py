import numpy as np
from math import sin, cos, pi, floor
from scipy.misc import imsave
from tools import *

def preprocess(input_data, output_filename, angle, start_x, start_y, width, height, ranges_to_keep):
	
	# Loads the hyperspectral image from the filename, rotates by angle (in degrees, positive means counterclockwise),
	# cuts a rectangle starting at (x, y) in the original image with dimensions (width, height)
	# and stores the new hyperspectral image
	# So (x, y) in the original image becomes (0, 0) in the output after rotation and translation
	# input_type, input_shape indicate the type and shape of the input array
	# input_modulus indicates the modulus that should be applied to the input data
	# No scaling is performed and all spectral bands are kept
	# ranges_to_keep indicates channels to be kept
	
	# Output is uint16
	
	# Initialization
	output_data = np.empty((width, height, input_data.shape[2]), dtype="float32")
	
	# Calculate high dynamic range
	cos_val = cos(-angle/180*pi)
	sin_val = sin(-angle/180*pi)
	for output_x in range(width):
		for output_y in range(height):
			
			# Process one spatial pixel (all spectral bands simultaneously)
			
			# Calculate coordinates
			x = start_x + output_x*cos_val - output_y*sin_val
			y = start_y + output_x*sin_val + output_y*cos_val
			x1 = floor(x)
			x2 = x1 + 1
			y1 = floor(y)
			y2 = y1 + 1
			
			output_data[output_x, output_y, :] = input_data[x1, y1, :]*(x2 - x)*(y2 - y) + input_data[x2, y1, :]*(x - x1)*(y2 - y) + input_data[x1, y2, :]*(x2 - x)*(y - y1) + input_data[x2, y2, :]*(x - x1)*(y - y1)
	
	# Calculate bounds outside of ranges without copying data
	min_val = np.amax(output_data)
	max_val = np.amin(output_data)
	for start, end in ranges_to_keep:
		min_val = min(min_val, np.amin(output_data[:, :, start:end]))
		max_val = max(max_val, np.amax(output_data[:, :, start:end]))
	
	# Map values to uint16 and save
	offset = -min_val
	scale = (2**16 - 1)/(max_val - min_val)
	with open(output_filename, "wb") as f:
		for x in range(width):
			f.write(np.rint((filter_bands(output_data[x:x + 1, :, :], ranges_to_keep) + offset)*scale).astype("uint16").tobytes())

def preprocess_mauna_kea():
	preprocess(load_mauna_kea_raw(), "../data/mauna_kea_preprocessed", 4.07, 40, 240, 2704, 729, ((2, 103), (104, 107), (114, 153), (168, 224)))

preprocess_mauna_kea()
