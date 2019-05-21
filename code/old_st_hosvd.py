import numpy as np
from operator import itemgetter
from time import time, clock
from functools import reduce
import math
import matplotlib.pyplot as plt
import bitarray
import zlib

# Original code, real mess from down here

quantization_steps = 256
starting_layer = 10
target_bits_after_point = 0
meta_quantization_step = 2**-target_bits_after_point

def compress_quantize1(data):
	
	# Meta-quantization, per layer do:
	# 1) define set of quantization values using equidistant distribution in the sorted values array
	# 2) round all values in tensor to nearest quantization value
	# 3) store quantization values using constant meta-quantization
	
	factor_matrices, core_tensor = data
	
	# Quantize core tensor
	core_tensor_quantized = []
	for layer in range(starting_layer, max(core_tensor.shape)):
		
		# Quantize one layer
		
		# Collect values
		values = []
		if layer < core_tensor.shape[0]:
			values.extend(list(core_tensor[layer, :layer + 1, :layer + 1].flatten()))
		if layer < core_tensor.shape[1]:
			values.extend(list(core_tensor[:layer, layer, :layer + 1].flatten()))
		if layer < core_tensor.shape[2]:
			values.extend(list(core_tensor[:layer, :layer, layer].flatten()))
		
		# Sort and pick quantization values
		values.sort()
		quantization_values = np.zeros(quantization_steps)
		step_size = (len(values) - 1)/(quantization_steps - 1)
		for step in range(quantization_steps):
			quantization_values[step] = values[max(0, min(len(values), int(round(step*step_size))))]
		
		# Quantize layer in core tensor
		def quantize(x):
			index = np.searchsorted(quantization_values, x)
			if index == quantization_steps - 1:
				return index
			elif abs(x - quantization_values[index]) < abs(x - quantization_values[index + 1]):
				return index
			else:
				return index + 1
		if layer < core_tensor.shape[0]:
			core_tensor[layer, :layer + 1, :layer + 1] = [[quantize(x) for x in subarray] for subarray in core_tensor[layer, :layer + 1, :layer + 1]]
		if layer < core_tensor.shape[1]:
			core_tensor[:layer, layer, :layer + 1] = [[quantize(x) for x in subarray] for subarray in core_tensor[:layer, layer, :layer + 1]]
		if layer < core_tensor.shape[2]:
			core_tensor[:layer, :layer, layer] = [[quantize(x) for x in subarray] for subarray in core_tensor[:layer, :layer, layer]]
		
		# Meta-quantize quantization values
		meta_quantization_min = min(values)
		meta_quantization_bits = int(math.ceil(math.log2(max(values) - min(values) + 2**-target_bits_after_point))) + target_bits_after_point
		quantization_values = np.rint((quantization_values - meta_quantization_min)/meta_quantization_step)
		
		# Store quantization metadata
		core_tensor_quantized.append((meta_quantization_min, meta_quantization_bits, quantization_values))
	
	# Store quantized tensor
	core_tensor_quantized.append(core_tensor)
	
	return factor_matrices, core_tensor_quantized

def decompress_quantize1(data):
	
	factor_matrices, core_tensor_quantized = data
	
	# Dequantize core tensor
	core_tensor = core_tensor_quantized[-1]
	for layer in range(starting_layer, max(core_tensor.shape)):
		
		# Dequantize one layer
		
		# Load metadata
		meta_quantization_min, meta_quantization_bits, quantization_values = core_tensor_quantized[layer - starting_layer]
		
		# Meta-dequantize quantization values
		quantization_values = quantization_values*meta_quantization_step + meta_quantization_min
		
		# Dequantize
		if layer < core_tensor.shape[0]:
			core_tensor[layer, :layer + 1, :layer + 1] = [[quantization_values[int(x)] for x in subarray] for subarray in core_tensor[layer, :layer + 1, :layer + 1]]
		if layer < core_tensor.shape[1]:
			core_tensor[:layer, layer, :layer + 1] = [[quantization_values[int(x)] for x in subarray] for subarray in core_tensor[:layer, layer, :layer + 1]]
		if layer < core_tensor.shape[2]:
			core_tensor[:layer, :layer, layer] = [[quantization_values[int(x)] for x in subarray] for subarray in core_tensor[:layer, :layer, layer]]
	
	return factor_matrices, core_tensor

def print_compression_rate_quantize1(original, compressed):
	
	original_size = reduce((lambda x, y: x * y), original.shape)*original.itemsize
	
	factor_matrices_quantized, core_tensor_quantized = compressed
	core_tensor = core_tensor_quantized[-1]
	factor_matrices_size = sum(map(lambda x : x.itemsize*x.size, factor_matrices_quantized))
	unquantized_section = core_tensor.itemsize*min(core_tensor.shape[0], starting_layer)*min(core_tensor.shape[1], starting_layer)*min(core_tensor.shape[2], starting_layer)
	core_tensor_size = unquantized_section + int(math.ceil((core_tensor.size - unquantized_section)*math.log2(quantization_steps)/8))
	meta_quantization_size = 0
	for layer in range(starting_layer, max(core_tensor.shape)):
		meta_quantization_min, meta_quantization_bits, quantization_values = core_tensor_quantized[layer - starting_layer]
		meta_quantization_size += 8 + 4 + quantization_steps*meta_quantization_bits/8
	meta_quantization_size = int(math.ceil(meta_quantization_size))
	core_tensor_size += meta_quantization_size
	compressed_size = factor_matrices_size + core_tensor_size
	
	print("Method: Tucker + Quantization of core tensor with custom-picked quantization levels per layer and meta-quantization")
	print("Original size:", original_size)
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor:", core_tensor_size)
	print("Compressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

factor_matrix_bits = 11
factor_matrix_quantization_step = 2**-(factor_matrix_bits - 1)

def compress_quantize2(data):
	
	# Quantize each layer separately, constant amount of bits used, scale varies per layer
	
	# 1 byte per value, layer by layer
	
	factor_matrices_original, core_tensor_original = data
	
	# Quantize factor matrices
	factor_matrices = []
	for factor_matrix in factor_matrices_original:
		factor_matrices.append(np.clip((np.rint(factor_matrix/factor_matrix_quantization_step)).astype(int), -2**(factor_matrix_bits - 1), 2**(factor_matrix_bits - 1) - 1))
	
	# Quantize core tensor
	core_tensor_quantized = []
	core_tensor = np.copy(core_tensor_original)
	for layer in range(1, max(core_tensor.shape)):
		
		# Quantize one layer
		
		# Collect values
		values = []
		if layer < core_tensor.shape[0]:
			values.extend(list(core_tensor[layer, :layer + 1, :layer + 1].flatten()))
		if layer < core_tensor.shape[1]:
			values.extend(list(core_tensor[:layer, layer, :layer + 1].flatten()))
		if layer < core_tensor.shape[2]:
			values.extend(list(core_tensor[:layer, :layer, layer].flatten()))
		
		# Store quantization metadata
		min_value = min(values)
		step_size = (max(values) - min_value)/255
		core_tensor_quantized.append((min_value, step_size))
		if step_size < 0.000000001:
			step_size = 1
		
		# Quantize layer in core tensor
		if layer < core_tensor.shape[0]:
			core_tensor[layer, :layer + 1, :layer + 1] = np.rint((core_tensor[layer, :layer + 1, :layer + 1] - min_value)/step_size)
		if layer < core_tensor.shape[1]:
			core_tensor[:layer, layer, :layer + 1] = np.rint((core_tensor[:layer, layer, :layer + 1] - min_value)/step_size)
		if layer < core_tensor.shape[2]:
			core_tensor[:layer, :layer, layer] = np.rint((core_tensor[:layer, :layer, layer] - min_value)/step_size)
	
	# Store quantized tensor
	core_tensor_quantized.append(core_tensor)
	
	return factor_matrices, core_tensor_quantized

def decompress_quantize2(data):
	
	factor_matrices_original, core_tensor_quantized = data
	
	# Dequantize factor matrices
	factor_matrices = []
	for factor_matrix in factor_matrices_original:
		factor_matrices.append(factor_matrix*factor_matrix_quantization_step)
	
	# Dequantize core tensor
	core_tensor = np.copy(core_tensor_quantized[-1])
	for layer in range(1, max(core_tensor.shape)):
		
		# Dequantize one layer
		
		# Load metadata
		min_value, step_size = core_tensor_quantized[layer - 1]
		
		# Dequantize
		if layer < core_tensor.shape[0]:
			core_tensor[layer, :layer + 1, :layer + 1] = np.rint(core_tensor[layer, :layer + 1, :layer + 1]*step_size + min_value)
		if layer < core_tensor.shape[1]:
			core_tensor[:layer, layer, :layer + 1] = np.rint(core_tensor[:layer, layer, :layer + 1]*step_size + min_value)
		if layer < core_tensor.shape[2]:
			core_tensor[:layer, :layer, layer] = np.rint(core_tensor[:layer, :layer, layer]*step_size + min_value)
	
	return factor_matrices, core_tensor

use_graycode = True

def print_compression_rate_quantize2(original, compressed):
	
	original_size = reduce((lambda x, y: x * y), original.shape)*original.itemsize
	factor_matrices, core_tensor_quantized = compressed
	compressed_size, factor_matrices_size_no_zlib, factor_matrices_size, meta_quantization_size, core_tensor_size_no_zlib, core_tensor_size = get_compress_quantize2_size(compressed, full_output=True)
	
	print("Method: Tucker + Quantizing factor matrices with constant precision + Quantizing core tensor with 8 bits/variable quantization step per layer + zlib")
	print("Using graycode:", use_graycode)
	print("Original size:", original_size)
	print("Factor matrices (no zlib):", factor_matrices_size_no_zlib)
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor (no zlib):", core_tensor_size_no_zlib)
	print("Core tensor:", core_tensor_size)
	print("Compressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

orthogonality_reconstruction_steps = 0
orthogonality_reconstruction_margin = 10
def get_compress_quantize2_size(compressed, full_output=False):
	
	factor_matrices_quantized, core_tensor_quantized = compressed
	core_tensor = core_tensor_quantized[-1]
	factor_matrices_size = int(math.ceil(factor_matrix_bits*sum(map(lambda x : x.size, factor_matrices_quantized))/8))
	
	# Convert to bitstring and compress with zlib
	factor_matrices_bits = bitarray.bitarray()
	if use_graycode:
		graycode = calculate_gray_code(factor_matrix_bits)
		code = {x - 2**(factor_matrix_bits - 1): bitarray.bitarray(graycode[x]) for x in range(2**factor_matrix_bits)}
	else:
		code = {x - 2**(factor_matrix_bits - 1): bitarray.bitarray(format(x, "0%sb"%factor_matrix_bits)) for x in range(2**factor_matrix_bits)} # Storing with shifted notation
	for factor_matrix in factor_matrices_quantized:
		for step in range(0, orthogonality_reconstruction_steps + 1):
			
			start_col = int(round(step/(orthogonality_reconstruction_steps + 1)*(factor_matrix.shape[1] - orthogonality_reconstruction_margin))) + orthogonality_reconstruction_margin
			end_col = int(round((step + 1)/(orthogonality_reconstruction_steps + 1)*(factor_matrix.shape[1] - orthogonality_reconstruction_margin))) + orthogonality_reconstruction_margin
			end_row = factor_matrix.shape[0] - start_col + orthogonality_reconstruction_margin
			if end_col == start_col:
				continue
			factor_matrices_bits.encode(code, factor_matrix[:end_row, start_col:end_col].flatten())
			
	factor_matrices_size = len(zlib.compress(factor_matrices_bits.tobytes()))
	
	# Convert to bitstring and compress with zlib
	core_tensor_size = core_tensor.itemsize
	core_tensor_bits = bitarray.bitarray()
	if use_graycode:
		graycode = calculate_gray_code(8)
		code = {x: bitarray.bitarray(graycode[x]) for x in range(256)}
	else:
		code = {x: bitarray.bitarray(format(x, "08b")) for x in range(256)}
	for i in range(1, max(core_tensor.shape)):
		if i < core_tensor.shape[0]:
			core_tensor_bits.encode(code, core_tensor[i, :i + 1, :i + 1].astype(int).flatten())
		if i < core_tensor.shape[1]:
			core_tensor_bits.encode(code, core_tensor[:i, i, :i + 1].astype(int).flatten())
		if i < core_tensor.shape[2]:
			core_tensor_bits.encode(code, core_tensor[:i, :i, i].astype(int).flatten())
	core_tensor_size += len(zlib.compress(core_tensor_bits.tobytes()))
	
	meta_quantization_size = 2*8*(max(core_tensor.shape) - 1) # 64-bit floats are used to store metadata
	core_tensor_size += meta_quantization_size
	compressed_size = factor_matrices_size + core_tensor_size
	
	if full_output:
		return compressed_size, len(factor_matrices_bits.tobytes()), factor_matrices_size, meta_quantization_size, core_tensor.itemsize + len(core_tensor_bits.tobytes()) + meta_quantization_size, core_tensor_size
	else:
		return compressed_size

quantization_step_size = 150

def compress_quantize3(data):
	
	# Quantize each layer separately, constant scale used, variable amount of bits per layer 
	
	# Variable amount of bits per layer, each layer uses same quantization step
	
	factor_matrices, core_tensor = data
	
	# Quantize core tensor
	core_tensor_quantized = []
	for layer in range(1, max(core_tensor.shape)):
		
		# Quantize one layer
		
		# Collect values
		values = []
		if layer < core_tensor.shape[0]:
			values.extend(list(core_tensor[layer, :layer + 1, :layer + 1].flatten()))
		if layer < core_tensor.shape[1]:
			values.extend(list(core_tensor[:layer, layer, :layer + 1].flatten()))
		if layer < core_tensor.shape[2]:
			values.extend(list(core_tensor[:layer, :layer, layer].flatten()))
		
		# Store quantization metadata
		min_value = round(min(values)/quantization_step_size)
		max_value = round(max(values)/quantization_step_size)
		bits_used = math.ceil(math.log2(max(abs(min_value), max_value + 1))) + 1
		core_tensor_quantized.append(bits_used)
		
		# Quantize layer in core tensor
		if layer < core_tensor.shape[0]:
			core_tensor[layer, :layer + 1, :layer + 1] = np.rint(core_tensor[layer, :layer + 1, :layer + 1]/quantization_step_size)
		if layer < core_tensor.shape[1]:
			core_tensor[:layer, layer, :layer + 1] = np.rint(core_tensor[:layer, layer, :layer + 1]/quantization_step_size)
		if layer < core_tensor.shape[2]:
			core_tensor[:layer, :layer, layer] = np.rint(core_tensor[:layer, :layer, layer]/quantization_step_size)
	
	# Store quantized tensor
	core_tensor_quantized.append(core_tensor)
	
	return factor_matrices, core_tensor_quantized

def decompress_quantize3(data):
	
	factor_matrices, core_tensor_quantized = data
	
	# Dequantize core tensor
	core_tensor = core_tensor_quantized[-1]
	for layer in range(1, max(core_tensor.shape)):
		
		# Dequantize one layer
		
		# Load metadata
		bits_used = core_tensor_quantized[layer - 1]
		
		# Dequantize
		if layer < core_tensor.shape[0]:
			core_tensor[layer, :layer + 1, :layer + 1] = np.rint(core_tensor[layer, :layer + 1, :layer + 1]*quantization_step_size)
		if layer < core_tensor.shape[1]:
			core_tensor[:layer, layer, :layer + 1] = np.rint(core_tensor[:layer, layer, :layer + 1]*quantization_step_size)
		if layer < core_tensor.shape[2]:
			core_tensor[:layer, :layer, layer] = np.rint(core_tensor[:layer, :layer, layer]*quantization_step_size)
	
	return factor_matrices, core_tensor

def print_compression_rate_quantize3(original, compressed):
	
	original_size = reduce((lambda x, y: x * y), original.shape)*original.itemsize
	
	factor_matrices_quantized, core_tensor_quantized = compressed
	core_tensor = core_tensor_quantized[-1]
	factor_matrices_size = sum(map(lambda x : x.itemsize*x.size, factor_matrices_quantized))
	core_tensor_size = core_tensor.itemsize
	
	# Convert to bitstring and compress with zlib
	core_tensor_bits = bitarray.bitarray()
	use_graycode = True
	for i in range(1, max(core_tensor.shape)):
		bits_used = core_tensor_quantized[i - 1]
		if use_graycode:
			graycode = calculate_gray_code(bits_used)
			code = {x - 2**(bits_used - 1): bitarray.bitarray(graycode[x]) for x in range(2**bits_used)}
		else:
			code = {x - 2**(bits_used - 1): bitarray.bitarray(format(x, "0%sb"%bits_used)) for x in range(2**bits_used)} # Storing with shifted notation
		if i < core_tensor.shape[0]:
			core_tensor_bits.encode(code, core_tensor[i, :i + 1, :i + 1].astype(int).flatten())
		if i < core_tensor.shape[1]:
			core_tensor_bits.encode(code, core_tensor[:i, i, :i + 1].astype(int).flatten())
		if i < core_tensor.shape[2]:
			core_tensor_bits.encode(code, core_tensor[:i, :i, i].astype(int).flatten())
	core_tensor_size += len(zlib.compress(core_tensor_bits.tobytes()))
	
	meta_quantization_size = 8*(max(core_tensor.shape) - 1) # 8 for 64-bit integer for storing bits per value per layer
	core_tensor_size += meta_quantization_size
	compressed_size = factor_matrices_size + core_tensor_size
	
	print("Method: Tucker + Quantizing core tensor with variable bits/quantization step %s per layer + zlib"%quantization_step_size)
	print("Using graycode:", use_graycode)
	print("Original size:", original_size)
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor (no zlib):", core_tensor.itemsize + len(core_tensor_bits.tobytes()) + meta_quantization_size)
	print("Core tensor:", core_tensor_size)
	print("Compressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

def plot_core_tensor_magnitudes(compressed):
	
	# Plots the RMS's of the 3D slices of the core tensor
	
	factor_matrices, core_tensor = compressed
	length = max(core_tensor.shape)
	rms = np.zeros(length)
	sos = np.zeros(length)
	values = []
	depths = []
	values_per_layer = []
	
	for i in range(length):
		
		initial_length = len(values)
		
		if i < core_tensor.shape[0]:
			core_tensor_slice = core_tensor[i, :i + 1, :i + 1]
			rms[i] += np.sum(core_tensor_slice*core_tensor_slice)
			values.extend(list(core_tensor_slice.flatten()))
			depths.extend([i]*core_tensor_slice.size)
		if i > 0:
			if i < core_tensor.shape[1]:
				core_tensor_slice = core_tensor[:i, i, :i + 1]
				rms[i] += np.sum(core_tensor_slice*core_tensor_slice)
				values.extend(list(core_tensor_slice.flatten()))
				depths.extend([i]*core_tensor_slice.size)
			if i < core_tensor.shape[2]:
				core_tensor_slice = core_tensor[:i, :i, i]
				rms[i] += np.sum(core_tensor_slice*core_tensor_slice)
				values.extend(list(core_tensor_slice.flatten()))
				depths.extend([i]*core_tensor_slice.size)
		
		sos[i] = rms[i]
		rms[i] = math.sqrt(rms[i]/(len(values) - initial_length))
		values_per_layer.append(values[initial_length:])
	
"""for i in range(20):
		plt.hist(values_per_layer[i], bins=min(len(values_per_layer[i]), 25))
		plt.title("Layer %s"%i)
		plt.show()
	
	plt.plot(range(1, length + 1), sos)
	plt.plot(range(1, length + 1), rms)
	plt.yscale("log")
	plt.show()
	
	plt.plot(depths[:1000000], values[:1000000], "bo")
	plt.show"""
	
		
