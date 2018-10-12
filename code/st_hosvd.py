import numpy as np
from operator import itemgetter

def compress(data, relative_target_error, print_progress=False):
	
	# This function calculates the ST-HOSVD of the given 3D tensor (see https://epubs.siam.org/doi/abs/10.1137/110836067) using the mode order: 2, 0, 1
	# data should be a numpy array
	# relative_target_error is the desired Frobenius norm of the difference of the input data and the decompressed version of the compressed data divided by the Frobenius norm of the input data
	# returns ([U_1, ..., U_n], S), meaning the factor matrices and the core tensor
	
	# Initialization
	core_tensor = data
	mode_order = 2, 0, 1
	factor_matrices = [None]*core_tensor.ndim
	"""mode_order = []
	factor_matrices = []
	for size in data.shape:
		factor_matrices.append(np.identity(size))"""
	current_sizes = list(data.shape)
	sq_abs_target_error = (relative_target_error*np.linalg.norm(data))**2
	sq_error_so_far = 0
	
	# Process modes
	for mode_index in range(len(mode_order)):
		
		mode = mode_order[mode_index]
		
		if print_progress:
			print("Processing mode %s"%mode)
		
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
			V, S, Uh = np.linalg.svd(uncompressed_matrix, full_matrices=False)
			U = np.transpose(Uh)
			Vh = np.transpose(V)
		else:
			# Mode is in front (possibly due to transposition)
			uncompressed_matrix = np.reshape(core_tensor, (current_sizes[mode], -1))
			U, S, Vh = np.linalg.svd(uncompressed_matrix, full_matrices=False)
		
		# Determine compression rank
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
	
	return factor_matrices, core_tensor

def decompress(compressed):
	
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
