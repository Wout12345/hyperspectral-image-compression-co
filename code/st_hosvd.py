import numpy as np
import scipy.linalg
import scipy.linalg.lapack as lapack
from operator import itemgetter
from time import time, clock
from functools import reduce
import math
import matplotlib.pyplot as plt
from bitarray import bitarray
import zlib
from copy import deepcopy
from heapq import heappush, heappop, heapify

from tools import *

# Constants
epsilon = 1e-9

# General helper functions

def product(iterable):
	return reduce(lambda x, y: x*y, iterable, 1)

def memory_size(A):
	return A.size*A.itemsize

# Phase 1: ST-HOSVD

# ST-HOSVD helper functions

def custom_eigh(A):
	# Returns eigenvalue decomposition with the result sorted based on the eigenvalues and the sqrt of the eigenvalues instead of the eigenvalues
	# Matrix must be symmetric
	lambdas, X = np.linalg.eigh(A)
	if not np.all(lambdas > -epsilon) and np.amin(lambdas)/np.amax(lambdas) < -1e-6:
		print("Not all eigenvalues are larger than -%s! Largest eigenvalue: %s, smallest eigenvalue: %s"%(epsilon, np.amax(lambdas), np.amin(lambdas)))
	lambdas = lambdas.clip(0, None)
	return np.sqrt(lambdas[::-1]), X[:, ::-1]

def custom_lanczos(A, bound, truncation_rank=None, transpose=False):
	
	# Computes the square roots of biggest eigenvalues and corresponding eigenvectors of A*A^T (or A^T*A if transpose) until the sum of the eigenvalues is >= bound
	
	# Initialization
	vectors_dimension = A.shape[0] if not transpose else A.shape[1]
	Q = np.empty([vectors_dimension, vectors_dimension])
	Q[:, 0] = 0
	Q[0, 0] = 1
	alphas = np.empty(vectors_dimension)
	betas = np.empty(vectors_dimension) # Index 0 is not needed but kept for simplicity
	
	# First iteration, corner case
	w = A @ (A.T @ Q[:, 0]) if not transpose else A.T @ (A @ Q[:, 0])
	alphas[0] = np.dot(w, Q[:, 0])
	w = w - alphas[0]*Q[:, 0]
	
	# Iterations
	i = 1
	alpha_sum = alphas[0]
	while (truncation_rank is None and i < vectors_dimension and alpha_sum < bound) or i < truncation_rank:
		
		betas[i] = np.linalg.norm(w)
		Q[:, i] = w/betas[i] # betas[i] should be non-zero for input of full rank
		w = A @ (A.T @ Q[:, i]) if not transpose else A.T @ (A @ Q[:, i])
		alphas[i] = np.dot(w, Q[:, i])
		alpha_sum += alphas[i]
		w = w - alphas[i]*Q[:, i] - betas[i]*Q[:, i - 1]
		
		# Re-orthogonalize
		w = w - Q[:, :i] @ (Q[:, :i].T @ w)
		
		i += 1
	
	# Calculate eigenvalue decomposition and transform
	lambdas, X = scipy.linalg.eigh_tridiagonal(alphas[:i], betas[1:i])
	transformed_X = Q[:, :i] @ X
	
	return np.sqrt(lambdas[::-1]), transformed_X[:, ::-1]

# End of ST-HOSVD helper functions

def compress_tucker(data, relative_target_error, extra_output=False, print_progress=False, mode_order=None, output_type="float32", compression_rank=None, randomized_svd=False, sample_ratio=0.1, samples_per_dimension=5, sort_indices=False, use_pure_gramian=True, use_qr_gramian=False, use_lanczos_gramian=False, store_rel_estimated_S_errors=False, test_all_truncation_ranks=False, calculate_explicit_errors=False, test_truncation_rank_limit=None):
	
	# This function calculates the ST-HOSVD of the given 3D tensor (see https://epubs.siam.org/doi/abs/10.1137/110836067)
	# data should be a numpy array and will not be changed by this call
	# relative_target_error is the desired Frobenius norm of the difference of the input data and the decompressed version of the compressed data divided by the Frobenius norm of the input data
	# extra_output turns the output into a dictionary containing the keys: "compressed", "original_size", "compressed_size", "total_cpu_time", "cpu_times", "cpu_times_svd", "population_sizes", "sample_sizes", 
	# if compression_rank is None, the compression rank is determined using the relative target error, else it should be a tuple describing the shape of the output core tensor
	# ...
	
	# compressed result is a dictionary of the form:
	#	- method: for now just "st-hosvd"
	#	- mode_order: given mode order
	#	- original_shape: relevant for reshaping
	#	- factor_matrices: list of factor matrices in order of mode handling (so each time a mode is processed a factor matrix is appended)
	#	- core_tensor: core tensor in output_type
	#	- S: list of arrays of (approximate) singular values, added in same order as factor matrices
	
	# extra_output fields:
	#	- compressed
	#	- original_size
	#	- compressed_size
	#	- total_cpu_time
	#	- cpu_times
	#	- cpu_times_svd
	#	- population_sizes
	#	- sample_sizes
	#	- rel_estimated_S_errors: only if store_rel_estimated_S_errors
	#	- truncation_rank_errors: only if test_all_truncation_ranks, dictionary mapping modes to list of errors, else {}
	
	
	# Start time measurements
	total_cpu_start = clock()
	
	# Initialization
	core_tensor = data.astype("float32")
	if mode_order is None:
		mode_order = [core_tensor.ndim - 1,] # Spectral dimension
		mode_order.extend(np.flip(np.argsort(core_tensor.shape[:-1])).tolist()) # Spatial dimensions, from large to small
	factor_matrices = []
	data_norm = custom_norm(data)
	sq_abs_target_error = (relative_target_error*data_norm)**2
	sq_error_so_far = 0
	modes = core_tensor.ndim
	output = {
		"total_cpu_time": 0,
		"cpu_times": [],
		"cpu_times_svd": [],
		"population_sizes": [],
		"sample_sizes": [],
		"rel_estimated_S_errors": [],
		"truncation_rank_errors": {}
	}
	compressed = {
		"method": "st-hosvd",
		"original_shape": data.shape,
		"mode_order": mode_order,
		"S": [] # Stored but not counted towards storage because this is only needed for further phases of the compression
	}
	
	# Process modes
	for mode_index, mode in enumerate(mode_order):
		
		cpu_start = clock()
		if print_progress:
			print("Processing mode %s"%mode)
		
		# Calculate population and sample size
		population_size = core_tensor.size//core_tensor.shape[mode]
		output["population_sizes"].append(population_size)
		sample_size = round(min(population_size, max(core_tensor.shape[mode]*samples_per_dimension, population_size*sample_ratio)))
		
		if core_tensor.shape[mode] >= core_tensor.size//core_tensor.shape[mode]:
			raise Exception("Less vectors than dimensions for SVD!")
		
		# Transpose modes if necessary to bring current mode to front (unless current mode is at front of back already)
		# transposition_order is also its own inverse order since just two elements are swapped
		transposition_order = list(range(modes))
		if mode != modes - 1:
			transposition_order[mode] = 0
			transposition_order[0] = mode
		core_tensor = np.transpose(core_tensor, transposition_order)
		
		# Take sample vectors from tensor
		use_sample = randomized_svd and sample_size < population_size
		if use_sample:
			sample_indices = np.random.choice(population_size, size=sample_size, replace=False)
			if sort_indices:
				sample_indices.sort()
			output["sample_sizes"].append(sample_size)
		else:
			output["sample_sizes"].append(population_size)
		transposed_ranks = list(core_tensor.shape)
		if mode == modes - 1:
			# Mode is already in back, convert to matrix of row vectors
			core_tensor = np.reshape(core_tensor, (-1, core_tensor.shape[-1]))
			sample_matrix = core_tensor[sample_indices] if use_sample else core_tensor
		else:
			# Mode is in front (possibly due to transposition)
			core_tensor = np.reshape(core_tensor, (core_tensor.shape[0], -1))
			sample_matrix = core_tensor[:, sample_indices] if use_sample else core_tensor
		
		# Calculate SVD of sample vectors, we only need U and S (V is useful too but can only be calculated without random sampling)
		# Pure Gramian: Calculate eigenvalue decomposition of A*A^T
		# QR Gramian: Calculate QR-decompositiion of A^T, calculate eigenvalue decomposition of R^T*R
		# Lanczos Gramian: Use Lanczos algorithm to calculate the truncated eigendecomposition of A*A^T
		cpu_start_svd = clock()
		sq_mode_target_error = (sq_abs_target_error - sq_error_so_far)/(modes - mode_index)
		if mode == modes - 1:
			# We used row vectors instead of column vectors, so convert SVD to corresponding format
			if use_pure_gramian:
				S, U = custom_eigh(sample_matrix.T @ sample_matrix)
			elif use_qr_gramian:
				R = np.linalg.qr(sample_matrix, mode="r")
				S, U = custom_eigh(R.T @ R)
			elif use_lanczos_gramian:
				sq_mode_target_norm = data_norm**2 - sq_error_so_far - sq_mode_target_error
				S, U = custom_lanczos(sample_matrix, sq_mode_target_norm, transpose=True, truncation_rank=None if compression_rank is None else compression_rank[mode])
				sq_abs_norm_so_far = np.sum(np.square(S))
				truncation_rank = S.size
			else:
				V, S, Uh = np.linalg.svd(sample_matrix, full_matrices=False)
				U = Uh.T
		else:
			# Using column vectors
			if use_pure_gramian:
				S, U = custom_eigh(sample_matrix @ sample_matrix.T)
			elif use_qr_gramian:
				R = np.linalg.qr(sample_matrix.T, mode="r")
				S, U = custom_eigh(R.T @ R)
			elif use_lanczos_gramian:
				sq_mode_target_norm = data_norm**2 - sq_error_so_far - sq_mode_target_error
				S, U = custom_lanczos(sample_matrix, sq_mode_target_norm, truncation_rank=None if compression_rank is None else compression_rank[mode])
				sq_abs_norm_so_far = np.sum(np.square(S))
				truncation_rank = S.size
			else:
				U, S, Vh = np.linalg.svd(sample_matrix, full_matrices=False)
		if use_sample:
			S = math.sqrt(max(1, population_size/sample_size))*S
		compressed["S"].append(S)
		
		output["cpu_times_svd"].append(clock() - cpu_start_svd)
		if print_progress:
			print("CPU time spent on SVD:", output["cpu_times_svd"][-1])
		
		# Estimate actual S using sample S
		S_calculation_time = clock()
		if store_rel_estimated_S_errors:
			_, exact_S, _ = np.linalg.svd(core_tensor, full_matrices=False)
			output["rel_estimated_S_errors"].append(np.linalg.norm(S - exact_S)/np.linalg.norm(exact_S))
		S_calculation_time = clock() - S_calculation_time
		cpu_start += S_calculation_time
		total_cpu_start += S_calculation_time
		
		# Test all truncation ranks if requested
		if test_all_truncation_ranks:
			output["truncation_rank_errors"][mode] = []
			current_norm = custom_norm(core_tensor)
			orthogonality_factor = np.linalg.norm(U.T @ U - np.eye(S.shape[0]))/math.sqrt(S.shape[0])
			max_truncation_rank = min(S.shape[0], test_truncation_rank_limit)
			if not calculate_explicit_errors and orthogonality_factor < 1e-3:
				# Factor matrix is considered orthogonal enough
				# Should work mathematically, also tested for low ranks to give approximately the same results
				if mode == modes - 1:
					sq_norms = np.sum(np.square(core_tensor @ U), axis=0)
				else:
					sq_norms = np.sum(np.square(U.T.copy() @ core_tensor), axis=1)
				accumulated_sq_norm = current_norm**2
				for truncation_rank2 in range(1, max_truncation_rank + 1):
					accumulated_sq_norm -= sq_norms[truncation_rank2 - 1]
					# accumulated_sq_norm should be positive mathematically, but may be a bit negative because of numerics, unorthogonality in the factor matrix, ...
					# Difference didn't seem significant enough to look into further
					output["truncation_rank_errors"][mode].append(math.sqrt(max(0, accumulated_sq_norm))/current_norm)
			else:
				# Factor matrix is not very orthogonal, brute-force each truncation rank
				for truncation_rank2 in range(1, max_truncation_rank + 1):
					factor_matrix = U[:, :truncation_rank2]
					if mode == modes - 1:
						decompressed_tensor = core_tensor @ (factor_matrix @ factor_matrix.T)
					else:
						decompressed_tensor = (factor_matrix @ factor_matrix.T) @ core_tensor
					output["truncation_rank_errors"][mode].append(custom_norm(decompressed_tensor - core_tensor)/current_norm)
		
		# Determine compression rank
		if not use_lanczos_gramian:
			if compression_rank is None:
				# Using relative target error
				sq_mode_error_so_far = 0
				truncation_rank = S.shape[0]
				for i in range(S.shape[0] - 1, -1, -1):
					new_error = S[i]**2 # We can use the singular values of the sample but need to scale to account for the full population
					if sq_mode_error_so_far + new_error > sq_mode_target_error:
						# Target error was excdeeded, truncate at previous rank
						truncation_rank = i + 1
						break
					else:
						# Target error was not exceeded, add error and continue
						sq_mode_error_so_far += new_error
				sq_error_so_far += sq_mode_error_so_far
			else:
				truncation_rank = compression_rank[mode]
		
		# Apply compression and fold back into tensor
		# Use V or Vh if possible
		factor_matrix = U[:, :truncation_rank]
		factor_matrices.append(factor_matrix)
		no_V = use_sample or use_pure_gramian or use_qr_gramian or use_lanczos_gramian
		if mode == modes - 1:
			if no_V:
				core_tensor = core_tensor @ factor_matrix
			else:
				core_tensor = V[:, :truncation_rank]*S[:truncation_rank]
			transposed_ranks[-1] = truncation_rank
		else:
			if no_V:
				core_tensor = factor_matrix.T.copy() @ core_tensor
			else:
				core_tensor = S[:truncation_rank, None]*Vh[:truncation_rank, :]
			transposed_ranks[0] = truncation_rank
		core_tensor = np.reshape(core_tensor, transposed_ranks)
		# Transpose back to original order
		core_tensor = np.transpose(core_tensor, transposition_order)
		
		output["cpu_times"].append(clock() - cpu_start)
		if print_progress:
			print("Finished mode, CPU time:", output["cpu_times"][-1])
	
	# Cast to lower-precision float
	core_tensor = core_tensor.astype(np.dtype(output_type))
	for i, factor_matrix in enumerate(factor_matrices):
		factor_matrices[i] = factor_matrix.astype(np.dtype(output_type))
	
	compressed["core_tensor"] = core_tensor
	compressed["factor_matrices"] = factor_matrices
	
	# Print timings
	output["total_cpu_time"] = clock() - total_cpu_start
	if print_progress:
		print("")
		print("Finished compression")
		print("Total CPU time spent:", output["total_cpu_time"])
		print("Total CPU time spent on SVD:", sum(output["cpu_times_svd"]))
		print("Ratio:", sum(output["cpu_times_svd"])/output["total_cpu_time"])
		print("")
	
	# Return output
	if extra_output:
		output["compressed"] = compressed
		output["compressed_size"] = get_compress_tucker_size(compressed)
		output["original_size"] = memory_size(data)
		return output
	else:
		return compressed

def decompress_tucker(compressed):
	
	# This function converts the given Tucker decomposition to the full tensor
	# This call does not change compressed
	# returns the full tensor
	
	# Cast to higher-precision float
	core_tensor = compressed["core_tensor"].astype(np.dtype("float32"))
	factor_matrices = []
	for factor_matrix in compressed["factor_matrices"]:
		factor_matrices.append(factor_matrix.astype(np.dtype("float32")))
	
	# Mode order is mathematically irrelevant, but may affect processing time (and maybe precision) significantly
	modes = core_tensor.ndim
	for mode_index, mode in reversed(list(enumerate(compressed["mode_order"]))):
		
		# Transpose modes if necessary
		if mode != 0 and mode != modes - 1:
			transposition_order = list(range(core_tensor.ndim))
			transposition_order[mode] = 0
			transposition_order[0] = mode
			core_tensor = np.transpose(core_tensor, transposition_order)
		
		# Unfold tensor and transform the vectors
		transposed_ranks = list(core_tensor.shape)
		factor_matrix = factor_matrices[mode_index]
		if mode == core_tensor.ndim - 1:
			# Mode is already in back, convert to matrix of row vectors
			core_tensor = np.reshape(core_tensor, (-1, core_tensor.shape[-1]))
			core_tensor = core_tensor @ factor_matrix.T
		else:
			# Mode is in front (possibly due to transposition)
			core_tensor = np.reshape(core_tensor, (core_tensor.shape[0], -1))
			core_tensor = factor_matrix @ core_tensor
		
		# Fold back into tensor
		if mode == modes - 1:
			transposed_ranks[-1] = factor_matrix.shape[0]
		else:
			transposed_ranks[0] = factor_matrix.shape[0]
		core_tensor = np.reshape(core_tensor, transposed_ranks)
		
		# Transpose back to original order
		if mode != 0 and mode != modes - 1:
			core_tensor = np.transpose(core_tensor, transposition_order)
	
	# Reshape if necessary
	if core_tensor.shape != compressed["original_shape"]:
		core_tensor = np.reshape(core_tensor, original_shape)
	
	return core_tensor

def get_compress_tucker_size(compressed):
	
	# Calculate factor matrix size
	factor_matrices_size = 0
	for factor_matrix in compressed["factor_matrices"]:
		factor_matrices_size += memory_size(factor_matrix)
	
	compressed_size = factor_matrices_size + memory_size(compressed["core_tensor"])
	
	return compressed_size

def get_compression_factor_tucker(original, compressed):
	return memory_size(original)/get_compress_tucker_size(compressed)

def print_compression_rate_tucker(original, compressed):
	
	original_size = memory_size(original)
	compressed_size = get_compress_tucker_size(compressed)
	
	print("Method: Tucker")
	print("Data type:", compressed["core_tensor"].dtype)
	print("Original shape:", original.shape, "\toriginal size:", original_size)
	print("Compressed shape:", compressed["core_tensor"].shape, "\tcompressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

# Phase 2: Orthogonality compression

def compress_orthogonality(data, copy=False, method="systems", quantize=False, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0):
	
	# Removes values from the factor matrices which can be theoretically reconstructed
	
	# data: output from phase 1
	# copy: whether or not the data from phase 1 should be copied
	# method: "systems" (custom code) or "householder" (using QR from LAPACK, assuming float32 is used)
	# orthogonality_reconstruction_steps (only for "systems"): amount of steps used in the truncation/reconstruction process, setting to 0 disables truncation
	# orthogonality_reconstruction_margin (only for "systems"): size of margin used in the truncation/reconstruction process
	# quantize: quantize output to uint16, used for testing the effect of different orthogonality compression parameters. quantization is performed after orthogonality compression
	
	# output is dictionary with keys:
	# - st_hosvd: dictionary containing ST-HOSVD output, apart from factor matrices
	# - method: method
	# - quantize: quantize
	# - factor_matrices: list of the new factor matrices, each a dictionary with keys:
	#	- data:
	#		- if "systems": a concatenation of all truncated columns into a 1D array (so column-major)
	#		- if "householder": a concatenation of the columns of the lower triangle of the H-matrix (so H[1:, 0], H[2:, 1], ...)
	#	- shape: shape of the original factor matrix
	#	- tau (only if "householder"): tau-vector from QR-call
	#	- blocks (only for "systems"): list of tuples of blocks stored in data, each tuple consists of (col2, truncation_row), the exclusive ending indices for columns and rows respectively
	#	- offset (only if quantize=True): offset applied to factor matrix values
	#	- scale (only if quantize=True): scale applied to factor matrix values after offset
	# - orthogonality_reconstruction_steps (only for "systems"): orthogonality_reconstruction_steps
	# - orthogonality_reconstruction_margin (only for "systems"): orthogonality_reconstruction_margin
	
	# a factor matrix is a distionary with keys:
	
	# Initialization
	compressed = {
		"st_hosvd": {},
		"factor_matrices": [],
		"method": method,
		"quantize": quantize
	}
	if method == "systems":
		compressed["orthogonality_reconstruction_steps"] = orthogonality_reconstruction_steps
		compressed["orthogonality_reconstruction_margin"] = orthogonality_reconstruction_margin
	for key in data:
		if key != "factor_matrices":
			if copy:
				compressed["st_hosvd"][key] = deepcopy(data[key])
			else:
				compressed["st_hosvd"][key] = data[key]
	
	# Construct flattened factor matrices
	margin = orthogonality_reconstruction_margin + 1
	for factor_matrix in data["factor_matrices"]:
		
		factor_matrix_dict = {
			"shape": factor_matrix.shape
		}
		rows = factor_matrix.shape[0]
		cols = factor_matrix.shape[1]
		
		if method == "systems":
			
			# Use linear systems
			
			# Initialization
			reconstruction_width = max(0, factor_matrix.shape[1] - margin)
			blocks_amount = min(reconstruction_width, orthogonality_reconstruction_steps + 1) + 1
			if blocks_amount > 1:
				block_size = reconstruction_width/(blocks_amount - 1)
			
			# Calculate truncated matrix size
			col2 = min(cols, margin)
			size = rows*col2
			for step in range(1, blocks_amount):
				col1 = col2
				col2 = min(cols, round(step*block_size) + margin)
				if col2 == col1:
					raise Exception("Empty block!")
				truncation_row = min(rows, rows - col1 + orthogonality_reconstruction_margin)
				size += truncation_row*(col2 - col1)
			
			# Construct truncated matrix
			truncated_factor_matrix = np.empty(size, dtype="float32")
			col2 = min(cols, margin)
			index = 0
			size = rows*col2
			truncated_factor_matrix[index:index + size] = factor_matrix[:, :col2].flatten()
			index += size
			blocks_list = [(col2, rows)]
			for step in range(1, blocks_amount):
				col1 = col2
				col2 = min(cols, round(step*block_size) + margin)
				truncation_row = min(rows, rows - col1 + orthogonality_reconstruction_margin)
				size = truncation_row*(col2 - col1)
				truncated_factor_matrix[index:index + size] = factor_matrix[:truncation_row, col1:col2].flatten()
				index += size
				blocks_list.append((col2, truncation_row))
			
			factor_matrix_dict["blocks"] = blocks_list
		
		elif method == "householder":
			
			# Use Householder reflections with LAPACK
			(h, tau), _ = scipy.linalg.qr(factor_matrix, mode="raw")
			truncated_factor_matrix = h.T[np.triu_indices(cols, 1, rows)]
			tau[tau < 0.5] = 0.5 # Set 0 values to 0.5 so sign will be clear
			factor_matrix_dict["tau"] = tau*np.sign(np.diag(h))
			
		# Quantize matrix if necessary
		if quantize:
			offset = -np.amin(truncated_factor_matrix)
			scale = (2**16 - 1)/(np.amax(truncated_factor_matrix) + offset)
			truncated_factor_matrix = np.rint((truncated_factor_matrix + offset)*scale).astype("uint16")
		
		# Finish factor matrix dictionary
		factor_matrix_dict["data"] = truncated_factor_matrix
		if quantize:
			factor_matrix_dict["offset"] = offset
			factor_matrix_dict["scale"] = scale
		compressed["factor_matrices"].append(factor_matrix_dict)
	
	# Return output
	return compressed

def decompress_orthogonality(data, copy=False, renormalize=False):
	
	# Reconstructs truncated values from the factor matrices
	
	# data: output from phase 2 or reverse phase 3
	# copy: whether or not the data from phase 1 should be copied
	# renormalize (only for "systems"): whether or not to explicitly renormalize columns
	
	# Initialize decompression
	if copy:
		decompressed = deepcopy(data["st_hosvd"])
	else:
		decompressed = data["st_hosvd"]
	decompressed["factor_matrices"] = []
	
	# Reconstruct factor matrices
	for factor_matrix in data["factor_matrices"]:
		
		rows = factor_matrix["shape"][0]
		cols = factor_matrix["shape"][1]
		
		# Dequantize if necessary
		# Needs to happen before orthogonality compression so matrix is actually (approximately) orthogonal
		data_matrix = factor_matrix["data"] if not data["quantize"] else factor_matrix["data"]/factor_matrix["scale"] - factor_matrix["offset"]
		
		if data["method"] == "systems":
			
			# Load known matrix blocks
			full_matrix = np.empty((rows, cols), dtype="float32")
			index = 0
			col1 = 0
			for col2, truncation_row in factor_matrix["blocks"]:
				size = (col2 - col1)*truncation_row
				full_matrix[:truncation_row, col1:col2] = data_matrix[index:index + size].reshape((truncation_row, col2 - col1))
				index += size
				col1 = col2
			
			# Reconstruct unknown matrix blocks
			col1 = 0
			for col2, truncation_row in factor_matrix["blocks"]:
				
				if truncation_row < rows:
					
					# Actually reconstruct values
					full_matrix[truncation_row:, col1:col2] = np.linalg.lstsq(full_matrix[truncation_row:, :col1].T, -full_matrix[:truncation_row, :col1].T @ full_matrix[:truncation_row, col1:col2], rcond=None)[0]
					
					# Optional renormalization
					if renormalize:
						# Renormalize all new columns entirely
						norms = np.linalg.norm(full_matrix[:, col1:col2], axis=0)
						np.putmask(norms, norms < epsilon, 1)
						full_matrix[:, col1:col2] = full_matrix[:, col1:col2]/norms
						
				col1 = col2
		
		elif data["method"] == "householder":
			
			# Use Householder reflections with LAPACK
			h = np.empty((cols, rows), dtype="float32")
			h[np.triu_indices(cols, 1, rows)] = data_matrix
			h = h.T
			signs = np.sign(factor_matrix["tau"])
			tau = np.abs(factor_matrix["tau"])
			tau[tau < 0.75] = 0
			full_matrix = lapack.sorgqr(h, tau)[0]*signs # Correct signs using signs stored in tau
		
		# Append factor matrix
		decompressed["factor_matrices"].append(full_matrix)
	
	return decompressed

# Phase 3: Quantization

def get_smallest_int_type(bits):
	sizes = (8, 16, 32, 64)
	for size in sizes:
		if size >= bits:
			return "uint%s"%size

def quantize_and_encode(data, data_bits, quantization_bits, encoding_method, copy=False, endian="little", min_val=None, max_val=None):
	
	# Quantize the given array and encode the resulting bits into a bitarray
	
	# encoding_method: "default", "graycode", "huffman"
	# If bitstring is None, a new bitstring is created
	
	# Returns bitarray, offset, scale(, tree if huffman encoding else None) with bitarray the given bitarray extended with the quantized data
	
	# Quantize
	offset = np.float32(-(np.amin(data) if min_val is None else min_val))
	scale = np.float32((2**quantization_bits - 1)/((np.amax(data) if max_val is None else max_val) + offset))
	# Perform operations in-place if allowed to
	if copy:
		corrected_data = (data + offset)*scale
	else:
		corrected_data = data.__iadd__(offset).__imul__(scale)
	corrected_data = np.rint(corrected_data, out=np.empty(corrected_data.shape)).astype(get_smallest_int_type(quantization_bits))
	corrected_data = np.reshape(corrected_data, (-1,))
	
	# Encode into bitstring
	if encoding_method == "huffman":
		code, tree = generate_code(corrected_data, quantization_bits, encoding_method, endian=endian)
		data_bits.encode(code, corrected_data)
		return offset, scale, tree
	else:
		code = generate_code(corrected_data, quantization_bits, encoding_method, endian=endian)
		data_bits.encode(code, corrected_data)
		return offset, scale, None

def decode_and_dequantize(data_bits, start, end, quantization_bits, encoding_method, count, scale, offset, endian="little", huffman_tree=None):
	
	# Decode the given bitarray in the range [start:end] using the encoding_method and possibly a Huffman code stored in compressed tree format
	
	# Decode
	if encoding_method == "huffman":
		iterator = data_bits[start:end].iterdecodetree(huffman_tree_to_bitarray(huffman_tree), quantization_bits) # Constructing tree for variable length prefix codes is pretty slow
	else:
		code = generate_code(None, quantization_bits, encoding_method, endian=endian, as_list=True)
		iterator = data_bits[start:end].iterdecodeconstant(code)
	data = np.fromiter(iterator, dtype="float32", count=count)
	
	# Dequantize
	(data.__imul__(1/scale) if abs(scale) > epsilon else data).__iadd__(-offset)
	return data

# Cache codes for faster reuse later
code_cache = {}
def generate_code(data, bits_per_symbol, encoding_method, as_list=False, endian="little"):
	
	# Generate code for data with the given amount of bits_per_symbol (apart from Huffman coding)
	# data is only necessary for Huffman coding
	# as_list returns code as a list instead of dict, only supported for constant-length codes ("default" and "graycode"), with symbols in list at index given by code
	
	# Calculate encoding code
	if encoding_method in ("default", "graycode"):
		graycode = (encoding_method == "graycode")
		key = (bits_per_symbol, as_list, graycode)
		if key in code_cache:
			return code_cache[key]
		else:
			if graycode:
				if as_list:
					code = [None,]*(2**bits_per_symbol)
					for symbol, code_value in enumerate(calculate_gray_code(bits_per_symbol)):
						code[code_value] = symbol
				else:
					code = {symbol: bitarray(code_value, endian) for symbol, code_value in enumerate(calculate_gray_code(bits_per_symbol))}
			else:
				if as_list:
					code = list(range(2**bits_per_symbol))
				else:
					code = {symbol: bitarray(format(symbol, "0" + str(bits_per_symbol) + "b"), endian=endian) for symbol in range(2**bits_per_symbol)}
			code_cache[key] = code
			return code
	elif encoding_method == "huffman":
		return huffman_code(data, bits_per_symbol, endian=endian)

def clear_code_cache():
	code_cache.clear()

def calculate_gray_code(bits):
	
	# Calculate graycode for n bits
	# From https://stackoverflow.com/questions/38738835/generating-gray-codes
	
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
	gray_code_recurse(g, bits - 1)
	return g

# Huffman code based on https://github.com/soxofaan/dahuffman/blob/master/dahuffman/huffmancodec.py
# Code compression was based on https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
# Probably quite inefficient in Python, so try to not use large alphabets

def huffman_code(data, quantization_bits, endian="little"):
	
	"""
	Build Huffman code table from symbol sequence
	:param data: sequence of symbols (numpy array of ints)
	:return: code, tree (in compressed tree format, which is a dictionary {endian: endian, data: tree_bytes} with tree_bytes compressed by zlib)
	"""
	
	# Heap consists of tuples: (frequency, [list of tuples: (symbol, (bitsize, value))])
	unique, counts = np.unique(data, return_counts=True)
	heap = [(f, [(s, (0, 0))], bitarray("1" + format(s, "0%sb"%quantization_bits), endian=endian)) for s, f in zip(unique, counts)]
	
	# Use heapq approach to build the encodings of the huffman tree leaves.
	heapify(heap)
	while len(heap) > 1:
		# Pop the 2 smallest items from heap
		a = heappop(heap)
		b = heappop(heap)
		# Merge nodes (update codes for each symbol appropriately)
		merged = (
			a[0] + b[0], # Sum of frequencies
			[(s, (n + 1, v)) for (s, (n, v)) in a[1]] # List of leaf nodes for this tree
			+ [(s, (n + 1, (1 << n) + v)) for (s, (n, v)) in b[1]],
			bitarray("0", endian=endian) + a[2] + b[2] # Compressed tree
		)
		heappush(heap, merged)
	
	# Code table is dictionary mapping symbol to (bitsize, value)
	root = heappop(heap)
	code = {}
	for key, (depth, value) in root[1]:
		code[key] = bitarray(format(value, "0%sb"%depth), endian=endian)
	
	return code, {"data": zlib.compress(root[2].tobytes()), "endian": endian} # We store some extra bits but these will be ignored by the parser

# End of Huffman code

def huffman_tree_to_bitarray(tree):
	
	# Decompresses tree format
	tree_bits = bitarray(endian=tree["endian"])
	tree_bits.frombytes(zlib.decompress(tree["data"]))
	return tree_bits

def huffman_tree_to_code(tree, quantization_bits, endian="little"):
	
	# Reconstructs Huffman code from compressed tree format
	# Based on https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
	# 1 bit means leaf, 0 bit means non-leaf node
	
	# Reconstruct bits
	tree_bits = huffman_tree_to_bitarray(tree)
	
	# Reconstruct code
	def huffman_tree_to_code_rec(tree_bits, code, current_prefix, index):
		# Interpret node
		# Returns new index
		if tree_bits[index]:
			# Leaf
			index += 1
			code[int(tree_bits[index:index + quantization_bits].to01(), 2)] = current_prefix.copy()
			return index + quantization_bits
		else:
			# Non-leaf node
			index += 1
			current_prefix.append(False)
			index = huffman_tree_to_code_rec(tree_bits, code, current_prefix, index)
			current_prefix[-1] = True
			index = huffman_tree_to_code_rec(tree_bits, code, current_prefix, index)
			del current_prefix[-1]
			return index
	
	code = {}
	huffman_tree_to_code_rec(tree_bits, code, bitarray(endian=endian), 0)
	
	return code

# Tensor layer code

def get_chunk_shape(tensor, current_dim, start, end):
	return [min(tensor.shape[dim], start) if dim < current_dim else (max(0, min(tensor.shape[dim], end) - start) if dim == current_dim else min(tensor.shape[dim], end)) for dim in range(tensor.ndim)]

def get_chunk_indices(tensor, current_dim, start, end):
	return tuple([slice(start)]*current_dim + [slice(start, end)] + [slice(end)]*(tensor.ndim - current_dim - 1))

def get_layers_size(tensor, start, end):
	return product([min(tensor.shape[dim], end) for dim in range(tensor.ndim)]) - product([min(tensor.shape[dim], start) for dim in range(tensor.ndim)])

def extract_layers(tensor, start, end):
	
	# Extract layers from range [start:end] from tensor in a consistent order
	
	# Initialization
	size = get_layers_size(tensor, start, end)
	out = np.empty(size, tensor.dtype)
	
	# Iterate over dimensions
	index = 0
	for dim in range(tensor.ndim):
		# Iterate over all indices with:
		# - index_i with i < dim: index_i < start
		# - index_i with i == dim: start <= index_i < end
		# - index_i with i > dim: index_i < end
		chunk_size = product(get_chunk_shape(tensor, dim, start, end))
		if chunk_size > 0:
			values = tensor[get_chunk_indices(tensor, dim, start, end)].flat
			out[index:index + chunk_size] = values
			index += chunk_size
	
	return out

def insert_layers(tensor, start, end, data):
	
	# Insert layers into range [start:end] in tensor in a consistent order
	
	# Iterate over dimensions
	index = 0
	for dim in range(tensor.ndim):
		# Iterate over all indices with:
		# - index_i with i < dim: index_i < start
		# - index_i with i == dim: start <= index_i < end
		# - index_i with i > dim: index_i < end
		shape = get_chunk_shape(tensor, dim, start, end)
		chunk_size = product(shape)
		if chunk_size > 0:
			tensor[get_chunk_indices(tensor, dim, start, end)] = np.reshape(data[index:index + chunk_size], shape)
			index += chunk_size

# End of tensor layer code

def compress_quantize(data, copy=False, endian="little", encoding_method="default", use_zlib=True, core_tensor_method="constant", core_tensor_parameter=16, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="constant", factor_matrix_parameter=16):
	
	# Quantize core tensor and factor matrices and apply lossless bitstring compression
	
	# Input:
	
	# data: compression dictionary as returned by phase 2 (orthogonality compression) with method "householder"
	# copy: whether or not the input data should be copied
	# endian: endianness of encoded bits
	# encoding_method: "default" (encode values in default binary form), "graycode" (using Gray-codes), "huffman" (using Huffman coding)
	
	# core_tensor_method:
	#	- "constant": constant quantization step for full tensor, no distinguishing in between layers
	#	- "layered-constant-bits": distinguish in between columns, use a constant amount of bits, quantize each layer separately
	#	- "layered-constant-step": distinguish in between columns, use a variable amount of bits per layer based on a target quantization step
	# core_tensor_parameter: at most 64
	#	- "constant": amount of bits that should be used per value
	#	- "layered-constant-bits": amount of bits that should be used per value
	#	- "layered-constant-step": (largest quantization step that may be used per layer)/(largest absolute value in core tensor), hopefully this relative definition causes the effect of this parameter to be roughly the same for different datasets, amount of bits per layer will be capped at 64
	# core_tensor_unquantized_rel_norm: the fraction of elements of the core tensor that should remain unquantized and stored as float32, the algorithm will select i as the smallest value so that the relative norm of core_tensor[:i, :i, ..., :i] meets this bound
	
	# factor_matrix_method:
	#	- "constant": constant quantization step for full tensor, no distinguishing in between columns
	#	- "layered": distinguish in between columns, decrease number of bits used based on column size and relevant S from ST-HOSVD phase 
	# factor_matrix_parameter: at most 64
	#	- "constant": amount of bits that should be used per value
	#	- "layered": amount of bits that should be used per value in the first column, other amounts are derived from here
	
	# Output: dictionary with keys:
	# - orthogonality_compression: contains compression as given by data, with st_hosvd->core_tensor, st_hosvd->S and factor_matrices removed
	# - data_bytes: bitarray after conversion to bytes and compression to zlib, this bitarray contains all concatenated quantized object data, starting with the core tensor
	# - endian: endian
	# - encoding_method: encoding_method
	# - use_zlib: use_zlib
	# - core_tensor: dictionary with the following keys:
	#	- unquantized_data: unquantized first section of core tensor
	# 	- method: core_tensor_method
	#	- shape: shape of core tensor
	#	- bits (only if "constant" or "layered-constant-bits"): core_tensor_parameter
	#	if "constant":
	#		- end: ending index (exclusive) of core tensor bits in data
	#		- offset: offset applied to data for quantization
	#		- scale: scale factor applied to data for quantization
	#		- tree (only if "huffman"): compressed Huffman tree
	#	- layers (only if "layered-constant-bits" or "layered-constant-step"): a list of dictionaries containing the following keys:
	#		- start: starting index (inclusive) of core tensor layer bits in data
	#		- end: ending index (exclusive) of core tensor layer bits in data
	#		- offset: offset applied to data for quantization
	#		- scale: scale factor applied to data for quantization
	#		- bits (only if "layered-constant-step"): amount of bits used per value on this layer
	#		- tree (only if "huffman"): compressed Huffman tree
	# - factor_matrix_method: factor_matrix_method
	# - factor_matrix_parameter: core_tensor_parameter
	# - factor_matrices: list of dictionaries with the following keys:
	#	- shape: shape of factor matrix
	#	- tau: tau-vector from orthogonality compression, stored in float32
	#	if "constant":
	#		- tree (only if "huffman"): compressed Huffman tree
	#		- start: starting index (inclusive) of factor matrix bits in data
	#		- end: ending index (exclusive) of factor matrix bits in data
	#		- offset: offset applied to data for quantization
	#		- scale: scale factor applied to data for quantization
	
	# Initalization
	clear_code_cache() # For more fair timing
	# Construct basic compression dictionary and add values from previous steps
	compressed = {
		"orthogonality_compression": {"st_hosvd": {}},
		"endian": endian,
		"encoding_method": encoding_method,
		"use_zlib": use_zlib,
		"factor_matrices": [],
		"factor_matrix_method": factor_matrix_method,
		"factor_matrix_parameter": factor_matrix_parameter
	}
	# Copy orthogonality compression data apart from factor matrices and ST-HOSVD data
	for key in data:
		if key not in ("st_hosvd", "factor_matrices"):
			if copy:
				compressed["orthogonality_compression"][key] = deepcopy(data[key])
			else:
				compressed["orthogonality_compression"][key] = data[key]
	# Copy ST-HOSVD data apart from core tensor
	for key in data["st_hosvd"]:
		if key not in ("core_tensor", "S"):
			if copy:
				compressed["orthogonality_compression"]["st_hosvd"][key] = deepcopy(data["st_hosvd"][key])
			else:
				compressed["orthogonality_compression"]["st_hosvd"][key] = data["st_hosvd"][key]
	data_bits = bitarray(endian=endian)
	
	# Core tensor
	
	core_tensor = data["st_hosvd"]["core_tensor"]
	core_tensor_dict = {
		"method": core_tensor_method,
		"shape": core_tensor.shape
	}
	
	# Choose starting layer for quantization
	total_norm = np.linalg.norm(core_tensor)
	start_layer = 0
	while start_layer < max(core_tensor.shape):
		if np.linalg.norm(core_tensor[(slice(start_layer),)*core_tensor.ndim])/total_norm >= core_tensor_unquantized_rel_norm:
			break
		start_layer += 1
	end_layer = max(core_tensor.shape)
	core_tensor_dict["unquantized_data"] = core_tensor[(slice(start_layer),)*core_tensor.ndim]
	
	# Handle quantization method
	if core_tensor_method == "constant":
		
		# Constant quantization, apply same quantization to entire core tensor
		core_tensor_dict["bits"] = core_tensor_parameter
		core_tensor_dict["offset"], core_tensor_dict["scale"], tree = quantize_and_encode(extract_layers(core_tensor, start_layer, end_layer), data_bits, core_tensor_parameter, encoding_method, copy=copy, endian=endian)
		core_tensor_dict["end"] = data_bits.length()
		if encoding_method == "huffman":
			core_tensor_dict["tree"] = tree
	
	elif core_tensor_method in ("layered-constant-bits", "layered-constant-step"):
		
		# Use layers
		core_tensor_dict["layers"] = []
		if core_tensor_method == "layered-constant-bits":
			core_tensor_dict["bits"] = core_tensor_parameter
		if core_tensor_method == "layered-constant-step":
			max_quantization_step = core_tensor_parameter*np.amax(np.abs(core_tensor))
		for layer_index in range(start_layer, end_layer):
			
			layer_dict = {
				"start": data_bits.length()
			}
			
			# Extract layer
			layer = extract_layers(core_tensor, layer_index, layer_index + 1)
			
			# Determine bits used
			min_val = np.amin(layer)
			max_val = np.amax(layer)
			if core_tensor_method == "layered-constant-bits":
				bits_per_symbol = core_tensor_parameter
			elif core_tensor_method == "layered-constant-step":
				# Number of bits is chosen so that quantization step <= max_quantization_step
				bits_per_symbol = min(64, max(1, math.ceil(math.log2((max_val - min_val)/max_quantization_step + 1))))
				layer_dict["bits"] = bits_per_symbol
			
			# Quantize and encode layer
			layer_dict["offset"], layer_dict["scale"], tree = quantize_and_encode(layer, data_bits, bits_per_symbol, encoding_method, copy=copy, endian=endian, min_val=min_val, max_val=max_val)
			layer_dict["end"] = data_bits.length()
			if encoding_method == "huffman":
				layer_dict["tree"] = tree
			
			core_tensor_dict["layers"].append(layer_dict)
	
	compressed["core_tensor"] = core_tensor_dict
	
	# End of core tensor
	
	# Factor matrices
	
	for factor_matrix in data["factor_matrices"]:
		
		factor_matrix_dict = {
			"shape": factor_matrix["shape"],
			"tau": factor_matrix["tau"] if not copy else deepcopy(factor_matrix["tau"])
		}
		
		# Handle quantization method
		if factor_matrix_method == "constant":
			
			# Constant quantization, apply same quantization to entire factor matrix
			factor_matrix_dict["start"] = data_bits.length()
			factor_matrix_dict["offset"], factor_matrix_dict["scale"], tree = quantize_and_encode(factor_matrix["data"], data_bits, factor_matrix_parameter, encoding_method, copy=copy, endian=endian)
			factor_matrix_dict["end"] = data_bits.length()
			if encoding_method == "huffman":
				factor_matrix_dict["tree"] = tree
		
		compressed["factor_matrices"].append(factor_matrix_dict)
	
	# End of factor matrices
	
	# Apply zlib to full data and return output
	compressed["data_bytes"] = zlib.compress(data_bits.tobytes()) if use_zlib else data_bits.tobytes()
	return compressed

def decompress_quantize(data, copy=False):
	
	# Decompress bitstrings and dequantize core tensor and factor matrices
	
	# Input:
	# data: compression dictionary as returned by phase 3
	# copy: whether or not the input data should be copied
	
	# Initialization
	clear_code_cache() # For more fair timing
	if copy:
		decompressed = deepcopy(data["orthogonality_compression"])
	else:
		decompressed = data["orthogonality_compression"]
	decompressed["factor_matrices"] = []
	endian = data["endian"]
	data_bits = bitarray(endian=endian)
	data_bits.frombytes(zlib.decompress(data["data_bytes"]) if data["use_zlib"] else data["data_bytes"])
	encoding_method = data["encoding_method"]
	
	# Core tensor
	
	core_tensor_dict = data["core_tensor"]
	core_tensor = np.empty(core_tensor_dict["shape"], "float32")
	
	# Load unquantized part
	start_layer = max(core_tensor_dict["unquantized_data"].shape)
	core_tensor[(slice(start_layer),)*core_tensor.ndim] = core_tensor_dict["unquantized_data"]
	end_layer = max(core_tensor.shape)
	
	# Load quantized part
	if core_tensor_dict["method"] == "constant":
		# Constant quantization, apply same quantization to entire core tensor
		insert_layers(core_tensor, start_layer, end_layer, decode_and_dequantize(data_bits, 0, core_tensor_dict["end"], core_tensor_dict["bits"], encoding_method, get_layers_size(core_tensor, start_layer, end_layer), core_tensor_dict["scale"], core_tensor_dict["offset"], endian=endian, huffman_tree=core_tensor_dict["tree"] if encoding_method == "huffman" else None))
	elif core_tensor_dict["method"] in ("layered-constant-bits", "layered-constant-step"):
		
		# Use layers
		for layer_index in range(start_layer, end_layer):
			start_time = clock()
			
			layer = core_tensor_dict["layers"][layer_index - start_layer]
			
			# Determine bits used
			if core_tensor_dict["method"] == "layered-constant-bits":
				bits_per_symbol = core_tensor_dict["bits"]
			elif core_tensor_dict["method"] == "layered-constant-step":
				bits_per_symbol = layer["bits"]
			
			# Insert layer
			insert_layers(core_tensor, layer_index, layer_index + 1, decode_and_dequantize(data_bits, layer["start"], layer["end"], bits_per_symbol, encoding_method, get_layers_size(core_tensor, layer_index, layer_index + 1), layer["scale"], layer["offset"], endian=endian, huffman_tree=layer["tree"] if encoding_method == "huffman" else None))
			
			print("Finished layer %s"%layer_index, clock() - start_time)
	
	decompressed["st_hosvd"]["core_tensor"] = core_tensor
	
	# End of core tensor
	
	# Factor matrices
	
	for factor_matrix in data["factor_matrices"]:
		
		factor_matrix_dict = {
			"shape": factor_matrix["shape"],
			"tau": factor_matrix["tau"] if not copy else deepcopy(factor_matrix["tau"])
		}
		
		if data["factor_matrix_method"] == "constant":
			
			# Constant quantization, apply same dequantization to entire factor matrix
			rows, cols = factor_matrix["shape"]
			factor_matrix_size = (rows - cols - 1)*cols + cols*(cols + 1)//2 # After orthogonality compression
			factor_matrix_dict["data"] = decode_and_dequantize(data_bits, factor_matrix["start"], factor_matrix["end"], data["factor_matrix_parameter"], encoding_method, factor_matrix_size, factor_matrix["scale"], factor_matrix["offset"], endian=endian, huffman_tree=factor_matrix["tree"] if encoding_method == "huffman" else None)
		
		decompressed["factor_matrices"].append(factor_matrix_dict)
	
	# End of factor matrices
	
	return decompressed

def get_compress_quantize_size(compressed, print_intermediate_values=False):
	
	# Get final compressed object size
	# We will consider the following objects:
	# - data_bytes: contains compressed and quantized values from core tensor and factor matrices
	# - core_tensor: metadata for core tensor
	#	if "constant":
	#		- tree (only if encoding_method == "huffman"): compressed Huffman tree, should hopefully be small but still may be noticeable
	#			- data
	#	if "layered-constant-bits" or "layered-constant-step":
	#		- start: 64-bit integer (32-bit would be fine if data_bits doesn't exceed 4 Gbit, which should hopefully be the case after the first compression phases anyway), we ignore end because this could be encoded more efficiently by sharing bounds across objects
	#		- offset: 32-bit float
	#		- scale: 32-bit float
	#		- bits (only if "layered-constant-step"): unsigned 8-bit integer
	#		- tree (only if encoding_method == "huffman"): compressed Huffman tree, should hopefully be small but still may be noticeable
	#			- data
	# - factor_matrices: metadata for factor matrices
	#	- tau: tau-vector from orthogonality compression, not quantized and stored in float32 for simplicity
	#	if "constant":
	#		- tree (only if encoding_method == "huffman"): compressed Huffman tree, should hopefully be small but still may be noticeable
	#			- data
	
	# General data
	compressed_size = len(compressed["data_bytes"])
	if print_intermediate_values:
		print("Only data bytes:", compressed_size)
	
	# Extra core tensor data
	compressed_size += memory_size(compressed["core_tensor"]["unquantized_data"])
	if compressed["core_tensor"]["method"] == "constant" and compressed["encoding_method"] == "huffman":
		
		if print_intermediate_values:
			print("Core tensor Huffman tree:", len(compressed["core_tensor"]["tree"]["data"]))
		compressed_size += len(compressed["core_tensor"]["tree"]["data"])
	
	elif compressed["core_tensor"]["method"] in ("layered-constant-bits", "layered-constant-step"):
		
		# Layers data
		layers_data = (8 + 4 + 4 + int(compressed["core_tensor"]["method"] == "layered-constant-step"))*len(compressed["core_tensor"]["layers"])
		if print_intermediate_values:
			print("Core tensor layer data:", layers_data)
		compressed_size += layers_data
		
		# Trees in layers
		if compressed["encoding_method"] == "huffman":
			for i, layer in enumerate(compressed["core_tensor"]["layers"]):
				tree_data = len(layer["tree"]["data"])
				if print_intermediate_values:
					print("Core tensor Huffman tree for layer %s:"%i, tree_data)
				compressed_size += tree_data
	
	# Extra factor matrix data
	for factor_matrix in compressed["factor_matrices"]:
		compressed_size += memory_size(factor_matrix["tau"])
		if compressed["factor_matrix_method"] == "constant" and compressed["encoding_method"] == "huffman":
			tree_data = len(factor_matrix["tree"]["data"])
			if print_intermediate_values:
				print("Factor matrix Huffman tree:", tree_data)
			compressed_size += tree_data
	
	# Total
	if print_intermediate_values:
		print("Total bytes:", compressed_size)
	return compressed_size

def get_compression_quantize_tucker(original, compressed):
	return memory_size(original)/get_compress_quantize_size(compressed)
