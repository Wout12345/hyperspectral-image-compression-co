import numpy as np
from operator import itemgetter
from time import time, clock
from functools import reduce
import math
import matplotlib.pyplot as plt
import bitarray
import zlib

epsilon = 1e-9

def compress_tensor_trains(data, relative_target_error, rank=None, print_progress=False):
	
	# This function calculates the TT using the ST-HOSVD of the given 3D tensor (see https://epubs.siam.org/doi/abs/10.1137/110836067) using the mode order: 4, 0, 1, 2, 3 after splitting the spatial dimensions
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
	original_shape = core_tensor.shape
	mode_order = 4, 0, 1, 2, 3 # What about last mode?
	factor_matrices = [None]*(len(mode_order) - 1)
	sq_abs_target_error = (relative_target_error*np.linalg.norm(data))**2
	sq_error_so_far = 0
	
	# Reshape
	new_shape = [0, 0, 0, 0, original_shape[2]]
	new_shape[0] = int(round(math.sqrt(original_shape[0])))
	new_shape[1] = new_shape[0]
	new_shape[2] = int(round(math.sqrt(original_shape[1])))
	new_shape[3] = new_shape[2]
	original_shape = new_shape
	core_tensor = np.reshape(core_tensor, original_shape)
		
	# Transpose modes
	core_tensor = np.transpose(core_tensor, mode_order)
	current_sizes = list(core_tensor.shape)
	
	# Process modes
	for mode_index in range(len(mode_order) - 1):
		
		mode = mode_order[mode_index]
		
		if print_progress:
			print("Processing mode %s"%mode)
			real_start = time()
			cpu_start = clock()
		
		# Sample tensor and calculate SVD
		current_vector_size = core_tensor.shape[0]
		vectors_amount = core_tensor.size//current_vector_size
		indices_size = current_vector_size*25
		if indices_size < vectors_amount:
			indices = np.random.randint(vectors_amount, size=indices_size)
			indices.sort()
		else:
			indices = range(vectors_amount)
		# Mode is in front
		uncompressed_matrix = np.reshape(core_tensor, (core_tensor.shape[0], -1))
		sample_matrix = uncompressed_matrix[:, indices]
		
		if print_progress:
			cpu_start = clock()
		U, S, Vh = np.linalg.svd(sample_matrix, full_matrices=False)
		if print_progress:
			cpu_time_svd = clock() - cpu_start
			print("CPU time spent on SVD:", cpu_time_svd)
			total_cpu_time_svd += cpu_time_svd
		
		# Determine compression rank
		if rank is None:
			# Using relative target error
			sq_mode_target_error = (sq_abs_target_error - sq_error_so_far)/(len(mode_order) - 1 - mode_index)
			sq_mode_error_so_far = 0
			truncation_rank = S.shape[0]
			for i in range(S.shape[0] - 1, -1, -1):
				new_error = max(1, vectors_amount/indices_size)*S[i]**2
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
		factor_matrices[mode_index] = U[:, :truncation_rank]
		# Merge first two modes
		compressed_matrix = factor_matrices[mode_index].T @ uncompressed_matrix
		core_tensor = np.reshape(compressed_matrix, [truncation_rank*core_tensor.shape[1],] + list(core_tensor.shape[2:]))
		
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
	
	return factor_matrices, core_tensor, original_shape

def decompress_tensor_trains(compressed):
	
	# This function converts the given TT decomposition to the full tensor
	# compressed is a tuple ([U_1, ..., U_n], S, original_shape), meaning the factor matrices and the core tensor
	# returns the full tensor
	
	factor_matrices_original, core_tensor_original, original_shape = compressed
	
	# Cast to higher-precision float
	data_type = np.dtype("float32")
	core_tensor = core_tensor_original.astype(data_type)
	factor_matrices = []
	for factor_matrix in factor_matrices_original:
		factor_matrices.append(factor_matrix.astype(data_type))
	
	# Iterate over modes
	mode_order = 4, 0, 1, 2, 3
	transposed_original_shape = [original_shape[mode] for mode in mode_order]
	data = core_tensor
	current_sizes = list(data.shape)
	for mode_index in range(len(factor_matrices)):
		# Unfold tensor and transform the vectors
		factor_matrix = factor_matrices[-mode_index - 1]
		compressed_matrix = np.reshape(data, (data.shape[0]//transposed_original_shape[-mode_index - 1], -1))
		data = factor_matrix @ compressed_matrix
	
	# Reshape into transformed transposed original shape
	data = np.reshape(data, transposed_original_shape)
	
	# Transpose back into original order
	inverse_order = []
	for i in range(len(mode_order)):
		inverse_order.append(mode_order.index(i))
	data = np.transpose(data, inverse_order)
	
	# Merge spatial dimensions again
	data = np.reshape(data, (original_shape[0]*original_shape[1], original_shape[2]*original_shape[3], original_shape[4]))
	
	return data

def print_compression_rate_tensor_trains(original, compressed):
	original_size = original.dtype.itemsize*reduce((lambda x, y: x * y), original.shape)
	factor_matrices, core_tensor, original_shape = compressed
	
	# Calculate factor matrix size
	factor_matrices_size = 0
	for factor_matrix in factor_matrices:
		factor_matrix_size = factor_matrix.size
		factor_matrices_size += factor_matrix_size*factor_matrix.itemsize
	
	compressed_size = factor_matrices_size + core_tensor.dtype.itemsize*reduce((lambda x, y: x * y), core_tensor.shape)
	print("Method: Tensor trains")
	print("Data type:", core_tensor.dtype)
	print("Original shape:", original.shape, "\toriginal size:", original_size)
	print("Compressed shape:", core_tensor.shape, "\tcompressed size:", compressed_size)
	print("Factor matrices:")
	for factor_matrix in factor_matrices:
		print(factor_matrix.shape)
	print("Compression ratio:", compressed_size/original_size)
