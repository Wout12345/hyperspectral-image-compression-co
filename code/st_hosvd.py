import numpy as np
from operator import itemgetter
from time import time, clock
from functools import reduce
import math
import matplotlib.pyplot as plt

def compress_tucker(data, relative_target_error, rank=None, print_progress=False):
	
	# This function calculates the ST-HOSVD of the given 3D tensor (see https://epubs.siam.org/doi/abs/10.1137/110836067) using the mode order: 2, 0, 1
	# data should be a numpy array
	# relative_target_error is the desired Frobenius norm of the difference of the input data and the decompressed version of the compressed data divided by the Frobenius norm of the input data
	# if rank is None, the compression rank is determined using the relative target error, else it should be a tuple describing the shape of the output core tensor
	# returns ([U_1, ..., U_n], S), meaning the factor matrices and the core tensor
	
	# Start time measurements
	if print_progress:
		total_cpu_start = clock()
		total_cpu_time_svd = 0
	
	# Initialization
	core_tensor = data
	mode_order = 2, 0, 1
	factor_matrices = [None]*core_tensor.ndim
	current_sizes = list(data.shape)
	sq_abs_target_error = (relative_target_error*np.linalg.norm(data))**2
	sq_error_so_far = 0
	
	# Process modes
	for mode_index in range(len(mode_order)):
		
		mode = mode_order[mode_index]
		
		if print_progress:
			print("Processing mode %s"%mode)
			real_start = time()
			cpu_start = clock()
		
		# Transpose modes if necessary to bring current mode to front (unless current mode is at front of back already)
		if mode != 0 and mode != data.ndim - 1:
			transposition_order = list(range(data.ndim))
			transposition_order[mode] = 0
			transposition_order[0] = mode
			core_tensor = np.transpose(core_tensor, transposition_order)
		
		# Unfold tensor and calculate SVD
		if mode == data.ndim - 1:
			# Mode is already in back, convert to matrix of row vectors
			uncompressed_matrix = np.reshape(core_tensor, (-1, current_sizes[mode]))
		else:
			# Mode is in front (possibly due to transposition)
			uncompressed_matrix = np.reshape(core_tensor, (current_sizes[mode], -1))
		if print_progress:
			cpu_start = clock()
		if mode == data.ndim - 1:
			# We used row vectors instead of column vectors, so convert SVD to corresponding format
			V, S, Uh = np.linalg.svd(uncompressed_matrix, full_matrices=False)
			U = np.transpose(Uh)
			Vh = np.transpose(V)
		else:
			U, S, Vh = np.linalg.svd(uncompressed_matrix, full_matrices=False)
		if print_progress:
			total_cpu_time_svd += clock() - cpu_start
		
		# Determine compression rank
		if rank is None:
			# Using relative target error
			sq_mode_target_error = (sq_abs_target_error - sq_error_so_far)/(data.ndim - mode_index)
			sq_mode_error_so_far = 0
			truncation_rank = S.shape[0]
			for i in range(S.shape[0] - 1, -1, -1):
				new_error = S[i]**2
				if sq_mode_error_so_far + new_error > sq_mode_target_error:
					# Target error was excdeeded, truncate at previous rank
					truncation_rank = i + 1
					break
				else:
					# Target error was not exceeded, add error and continue
					sq_mode_error_so_far += new_error
			sq_error_so_far += sq_mode_error_so_far
		else:
			truncation_rank = rank[mode]
		
		# Apply compression and fold back into tensor
		factor_matrices[mode] = U[:, :truncation_rank]
		if mode == data.ndim - 1:
			compressed_matrix = np.matmul(np.transpose(Vh[:truncation_rank, :]), np.diag(S[:truncation_rank]))
			current_sizes[mode] = truncation_rank
		else:
			compressed_matrix = np.matmul(np.diag(S[:truncation_rank]), Vh[:truncation_rank, :])
			if mode != 0:
				# Will transpose later, so truncated mode is actually in front at the moment
				current_sizes[mode] = current_sizes[0]
				current_sizes[0] = truncation_rank
			else:
				current_sizes[mode] = truncation_rank
		core_tensor = np.reshape(compressed_matrix, current_sizes)
		
		if mode != 0 and mode != data.ndim - 1:
			# Transpose back to original order
			core_tensor = np.transpose(core_tensor, transposition_order)
			current_sizes[0], current_sizes[mode] = current_sizes[mode], current_sizes[0]
		
		if print_progress:
			print("Finished mode")
			print("Real time:", time() - real_start)
			print("CPU time:", clock() - cpu_start)
			real_start = time()
			cpu_start = clock()
	
	if print_progress:
		print("")
		print("Finished compression")
		total_cpu_time = clock() - total_cpu_start
		print("Total CPU time spent:", total_cpu_time)
		print("Total CPU time spent on SVD:", total_cpu_time_svd)
		print("Ratio:", total_cpu_time_svd/total_cpu_time)
		print("")
	
	return factor_matrices, core_tensor

def decompress_tucker(compressed):
	
	# This function converts the given Tucker decomposition to the full tensor
	# compressed is a tuple ([U_1, ..., U_n], S), meaning the factor matrices and the core tensor
	# returns the full tensor
	
	factor_matrices, core_tensor = compressed
	
	# Mode order is mathematically irrelevant, but may affect processing time (and maybe precision) significantly
	data = core_tensor
	current_sizes = list(data.shape)
	for mode in range(len(factor_matrices)):
		
		# Transpose modes if necessary
		if mode != 0 and mode != data.ndim - 1:
			transposition_order = list(range(data.ndim))
			transposition_order[mode] = 0
			transposition_order[0] = mode
			data = np.transpose(data, transposition_order)
		
		# Unfold tensor and transform the vectors
		factor_matrix = factor_matrices[mode]
		if mode == data.ndim - 1:
			# Mode is already in back, convert to matrix of row vectors
			compressed_matrix = np.reshape(data, (-1, current_sizes[mode]))
			decompressed_matrix = np.matmul(compressed_matrix, np.transpose(factor_matrix))
		else:
			# Mode is in front (possibly due to transposition)
			compressed_matrix = np.reshape(data, (current_sizes[mode], -1))
			decompressed_matrix = np.matmul(factor_matrix, compressed_matrix)
		
		# Fold back into tensor
		if mode == data.ndim - 1:
			current_sizes[mode] = factor_matrix.shape[0]
		else:
			if mode != 0:
				# Will transpose later, so truncated mode is actually in front at the moment
				current_sizes[mode] = current_sizes[0]
				current_sizes[0] = factor_matrix.shape[0]
			else:
				current_sizes[mode] = factor_matrix.shape[0]
		data = np.reshape(decompressed_matrix, current_sizes)
		
		if mode != 0 and mode != data.ndim - 1:
			# Transpose back to original order
			data = np.transpose(data, transposition_order)
			current_sizes[0], current_sizes[mode] = current_sizes[mode], current_sizes[0]
	
	return data

def load_tucker(path):
	
	arrays = np.load(path)["arr_0"]
	return list(arrays[:-1]), arrays[-1]

def save_tucker(compressed, path):
	
	factor_matrices, core_tensor = compressed
	arrays = factor_matrices
	arrays.append(core_tensor)
	np.savez(path, arrays)

def print_compression_rate_tucker(original, compressed):
	original_size = original.ndim + reduce((lambda x, y: x * y), original.shape)
	factor_matrices, core_tensor = compressed
	compressed_size = 2*original.ndim + sum([factor_matrix.shape[0]*factor_matrix.shape[1] for factor_matrix in factor_matrices]) + reduce((lambda x, y: x * y), core_tensor.shape)
	print("Sizes in number of floats:")
	print("Original shape:", original.shape, "\toriginal size:", original_size)
	print("Compressed shape:", core_tensor.shape, "\tcompressed size:", compressed_size)
	print("Compression ratio (bytes):", 4*compressed_size/original_size) # Times 4 because the Tucker decomposition is stored using 64-bit floats, the original data using 16-bit ints

target_bits_after_point = 0
meta_quantization_step = 2**-target_bits_after_point
quantization_steps = 256
starting_layer = 10

def compress_quantize(data):
	
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

def decompress_quantize(data):
	
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

def print_compression_rate_quantize(original, compressed):
	
	original_size = reduce((lambda x, y: x * y), original.shape)*2
	
	factor_matrices_quantized, core_tensor_quantized = compressed
	core_tensor = core_tensor_quantized[-1]
	factor_matrices_size = sum(map(lambda x : x.size, factor_matrices_quantized))*8
	unquantized_section = min(core_tensor.shape[0], starting_layer)*min(core_tensor.shape[1], starting_layer)*min(core_tensor.shape[2], starting_layer)
	core_tensor_size = unquantized_section*8 + int(math.ceil((core_tensor.size - unquantized_section)*math.log2(quantization_steps)/8))
	meta_quantization_size = 0
	for layer in range(starting_layer, max(core_tensor.shape)):
		meta_quantization_min, meta_quantization_bits, quantization_values = core_tensor_quantized[layer - starting_layer]
		meta_quantization_size += 8 + 4 + quantization_steps*meta_quantization_bits/8
	meta_quantization_size = int(math.ceil(meta_quantization_size))
	core_tensor_size += meta_quantization_size
	compressed_size = factor_matrices_size + core_tensor_size
	
	print("Sizes in bytes:")
	print("Original size:", original_size)
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor (including meta quantization):", core_tensor_size)
	print("Compressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

factor_matrix_quantization_bits = 11
factor_matrix_quantization_step = 2**-(factor_matrix_quantization_bits - 1)

def compress_quantize2(data):
	
	# 1 byte per value, layer by layer
	
	factor_matrices, core_tensor = data
	
	# Quantize factor matrices
	for i in range(len(factor_matrices)):
		factor_matrices[i] = np.rint(factor_matrices[i]/factor_matrix_quantization_step)
	
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
		min_value = min(values)
		step_size = (max(values) - min_value)/255
		core_tensor_quantized.append((min_value, step_size))
		
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
	
	factor_matrices, core_tensor_quantized = data
	
	# Dequantize factor matrices
	for i in range(len(factor_matrices)):
		factor_matrices[i] = factor_matrices[i]*factor_matrix_quantization_step
	
	# Dequantize core tensor
	core_tensor = core_tensor_quantized[-1]
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

def print_compression_rate_quantize2(original, compressed):
	
	original_size = reduce((lambda x, y: x * y), original.shape)*2
	
	factor_matrices_quantized, core_tensor_quantized = compressed
	core_tensor = core_tensor_quantized[-1]
	factor_matrices_size = int(math.ceil(sum(map(lambda x : x.size, factor_matrices_quantized))*factor_matrix_quantization_bits/8))
	core_tensor_size = 8 + core_tensor.size*1
	meta_quantization_size = 2*8*(max(core_tensor.shape) - 1)
	core_tensor_size += meta_quantization_size
	compressed_size = factor_matrices_size + core_tensor_size
	
	print("Sizes in bytes:")
	print("Original size:", original_size)
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor (including meta quantization):", core_tensor_size)
	print("Compressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

quantization_step_size = 150

def compress_quantize3(data):
	
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
		min_value = int(round(min(values)/quantization_step_size))
		max_value = int(round(max(values)/quantization_step_size))
		bits_used = int(math.ceil(math.log2(max(abs(min_value), max_value + 1)))) + 1
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
	
	original_size = reduce((lambda x, y: x * y), original.shape)*2
	
	factor_matrices_quantized, core_tensor_quantized = compressed
	core_tensor = core_tensor_quantized[-1]
	factor_matrices_size = sum(map(lambda x : x.size, factor_matrices_quantized))*8
	core_tensor_size = 8*8 # In bits
	for i in range(1, max(core_tensor.shape)):
		core_tensor_size += core_tensor_quantized[i - 1]*(core_tensor[:i + 1, :i + 1, :i + 1].size - core_tensor[:i, :i, :i].size)
	core_tensor_size = int(math.ceil(core_tensor_size/8)) # To bytes
	meta_quantization_size = 8*(max(core_tensor.shape) - 1)
	core_tensor_size += meta_quantization_size
	compressed_size = factor_matrices_size + core_tensor_size
	
	print("Sizes in bytes:")
	print("Original size:", original_size)
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor (including meta quantization):", core_tensor_size)
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
	
		
