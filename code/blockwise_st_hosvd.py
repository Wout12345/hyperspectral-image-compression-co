import numpy as np
from operator import itemgetter
from time import time, clock
from functools import reduce
import math
import matplotlib.pyplot as plt
import bitarray
import zlib

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
	block_size = 32
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
		vectors_amount = core_tensor.size//current_sizes[mode]
		indices_size = current_sizes[mode]*20
		if indices_size < vectors_amount:
			indices = np.random.randint(vectors_amount, size=indices_size)
		else:
			indices = range(vectors_amount)
		if mode == data.ndim - 1:
			# Mode is already in back, convert to matrix of row vectors
			uncompressed_matrix = np.reshape(core_tensor, (-1, current_sizes[mode]))
			sample_matrix = uncompressed_matrix[indices]
		else:
			# Mode is in front (possibly due to transposition)
			uncompressed_matrix = np.reshape(core_tensor, (current_sizes[mode], -1))
			sample_matrix = uncompressed_matrix[:, indices]
		if print_progress:
			cpu_start = clock()
		if mode == data.ndim - 1:
			# We used row vectors instead of column vectors, so convert SVD to corresponding format
			_, S, Uh = np.linalg.svd(sample_matrix, full_matrices=False)
			U = np.transpose(Uh)
		else:
			#gram_matrix = uncompressed_matrix @ uncompressed_matrix.T
			#S, U = np.linalg.eig(gram_matrix)
			U, S, Vh = np.linalg.svd(sample_matrix, full_matrices=False)
		if print_progress:
			cpu_time_svd = clock() - cpu_start
			print("CPU time spent on SVD:", cpu_time_svd)
			total_cpu_time_svd += cpu_time_svd
		
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
			compressed_matrix = uncompressed_matrix @ factor_matrices[mode]
			current_sizes[mode] = truncation_rank
		else:
			compressed_matrix = factor_matrices[mode].T @ uncompressed_matrix
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
	
	# Cast to lower-precision float
	data_type = np.dtype("float32")
	core_tensor = core_tensor.astype(data_type)
	for i, factor_matrix in enumerate(factor_matrices):
		factor_matrices[i] = factor_matrix.astype(data_type)
	
	return factor_matrices, core_tensor

factor_matrix_reconstruction_margin = 10
factor_matrix_reconstruction_steps = 1

def decompress_tucker(compressed):
	
	# This function converts the given Tucker decomposition to the full tensor
	# compressed is a tuple ([U_1, ..., U_n], S), meaning the factor matrices and the core tensor
	# returns the full tensor
	
	factor_matrices, core_tensor = compressed
	
	# Cast to higher-precision float
	data_type = np.dtype("float64")
	core_tensor = core_tensor.astype(data_type)
	for i, factor_matrix in enumerate(factor_matrices):
		factor_matrices[i] = factor_matrix.astype(data_type)
	
	# Decompress factor matrices by reconstructing orthogonal components
	for factor_matrix in factor_matrices:
		
		#print("Decompressing factor matrix")
		cpu_start = clock()
		
		rows, cols = factor_matrix.shape
		error = 0
		exact = np.copy(factor_matrix)
		for step in range(1, factor_matrix_reconstruction_steps + 1):
			
			start_col = int(round(step/(factor_matrix_reconstruction_steps + 1)*(factor_matrix.shape[1] - factor_matrix_reconstruction_margin))) + factor_matrix_reconstruction_margin
			end_col = int(round((step + 1)/(factor_matrix_reconstruction_steps + 1)*(factor_matrix.shape[1] - factor_matrix_reconstruction_margin))) + factor_matrix_reconstruction_margin
			start_row = factor_matrix.shape[0] - start_col + factor_matrix_reconstruction_margin
			
			if end_col == start_col or start_row >= factor_matrix.shape[0]:
				continue
			
			factor_matrix[start_row:, start_col:end_col], residuals, rank, s = np.linalg.lstsq(factor_matrix[start_row:, :start_col].T, -factor_matrix[:start_row, :start_col].T @ factor_matrix[:start_row, start_col:end_col])
			
			# Orthogonalize full column
			for _ in range(2):
				factor_matrix[:, start_col:end_col] = factor_matrix[:, start_col:end_col] - factor_matrix[:, :start_col] @ factor_matrix[:, :start_col].T @ factor_matrix[:, start_col:end_col]
			
			"""for i in range(1, min(cols, 1000)):
				# For each column, calculate last i values with i equations describing the orthogonal relationships in between this column and the i previous ones
				#factor_matrix[-i:, i] = np.linalg.solve(np.transpose(factor_matrix[-i:, :i]), -np.dot(np.transpose(factor_matrix[:-i, :i]), factor_matrix[:-i, i]))
				if i > factor_matrix_reconstruction_margin:
					j = -i + factor_matrix_reconstruction_margin
					factor_matrix[j:, i], residuals, rank, s = np.linalg.lstsq(np.transpose(factor_matrix[j:, :i]), -np.dot(np.transpose(factor_matrix[:j, :i]), factor_matrix[:j, i]))
				# Normalize
				#factor_matrix[-i:, i] = factor_matrix[-i:, i]*math.sqrt((1 - np.sum(factor_matrix[:-i, i]**2))/np.sum(factor_matrix[-i:, i]**2))"""
		
		#print("Frobenius norm of error in matrix:", math.sqrt(np.sum((factor_matrix - exact)**2)))
		
		#print("Done! Took %s seconds"%(clock() - cpu_start))
	
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

def print_compression_rate_tucker(original, compressed):
	original_size = original.dtype.itemsize*reduce((lambda x, y: x * y), original.shape)
	factor_matrices, core_tensor = compressed
	
	# Calculate factor matrix size
	factor_matrices_size = 0
	for factor_matrix in factor_matrices:
		
		factor_matrix_size = factor_matrix.size
		for step in range(1, factor_matrix_reconstruction_steps + 1):
			
			start_col = int(round(step/(factor_matrix_reconstruction_steps + 1)*(factor_matrix.shape[1] - factor_matrix_reconstruction_margin))) + factor_matrix_reconstruction_margin
			end_col = int(round((step + 1)/(factor_matrix_reconstruction_steps + 1)*(factor_matrix.shape[1] - factor_matrix_reconstruction_margin))) + factor_matrix_reconstruction_margin
			start_row = factor_matrix.shape[0] - start_col + factor_matrix_reconstruction_margin
			
			if end_col == start_col or start_row >= factor_matrix.shape[0]:
				continue
			
			factor_matrix_size -= (end_col - start_col)*(factor_matrix.shape[0] - start_row)
			
		factor_matrices_size += factor_matrix_size*factor_matrix.itemsize
	
	compressed_size = factor_matrices_size + core_tensor.dtype.itemsize*reduce((lambda x, y: x * y), core_tensor.shape)
	print("Method: Tucker")
	print("Data type:", core_tensor.dtype)
	print("Original shape:", original.shape, "\toriginal size:", original_size)
	print("Compressed shape:", core_tensor.shape, "\tcompressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

def load_tucker(path):
	
	arrays = np.load(path)["arr_0"]
	return list(arrays[:-1]), arrays[-1]

def save_tucker(compressed, path):
	
	factor_matrices, core_tensor = compressed
	arrays = factor_matrices
	arrays.append(core_tensor)
	np.savez(path, arrays)

factor_matrix_bits = 11
factor_matrix_quantization_step = 2**-(factor_matrix_bits - 1)

def compress_quantize2(data):
	
	# 1 byte per value, layer by layer
	
	factor_matrices, core_tensor = data
	
	# Quantize factor matrices
	for i, factor_matrix in enumerate(factor_matrices):
		factor_matrices[i] = np.clip((np.rint(factor_matrix/factor_matrix_quantization_step)).astype(int), -2**(factor_matrix_bits - 1), 2**(factor_matrix_bits - 1) - 1)
	
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
	for i, factor_matrix in enumerate(factor_matrices):
		factor_matrices[i] = factor_matrix*factor_matrix_quantization_step
	
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

# From https://stackoverflow.com/questions/38738835/generating-gray-codes
def calculate_gray_code(n):
	
	def gray_code_recurse(g, n):
		k = len(g)
		if n <= 0 :
			return
		else:
			for i in range(k - 1, -1, -1):
				char = "1" + g[i]
				g.append(char)
			for i in range(k - 1, -1, -1):
				g[i] = "0" + g[i]

			gray_code_recurse(g, n - 1)

	g = ["0", "1"]
	gray_code_recurse(g, n - 1)
	return g

def print_compression_rate_quantize2(original, compressed):
	
	use_graycode = True
	
	original_size = reduce((lambda x, y: x * y), original.shape)*original.itemsize
	
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
		for step in range(0, factor_matrix_reconstruction_steps + 1):
			
			start_col = int(round(step/(factor_matrix_reconstruction_steps + 1)*(factor_matrix.shape[1] - factor_matrix_reconstruction_margin))) + factor_matrix_reconstruction_margin
			end_col = int(round((step + 1)/(factor_matrix_reconstruction_steps + 1)*(factor_matrix.shape[1] - factor_matrix_reconstruction_margin))) + factor_matrix_reconstruction_margin
			end_row = factor_matrix.shape[0] - start_col + factor_matrix_reconstruction_margin
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
	
	print("Method: Tucker + Quantizing factor matrices with constant precision + Quantizing core tensor with 8 bits/variable quantization step per layer + zlib")
	print("Using graycode:", use_graycode)
	print("Original size:", original_size)
	print("Factor matrices (no zlib):", len(factor_matrices_bits.tobytes()))
	print("Factor matrices:", factor_matrices_size)
	print("Meta quantization size:", meta_quantization_size)
	print("Core tensor (no zlib):", core_tensor.itemsize + len(core_tensor_bits.tobytes()) + meta_quantization_size)
	print("Core tensor:", core_tensor_size)
	print("Compressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)
