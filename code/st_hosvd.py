import numpy as np
import scipy.linalg
import scipy.linalg.lapack as lapack
from operator import itemgetter
from time import time, clock
from functools import reduce
import math
import matplotlib.pyplot as plt
import bitarray
import zlib
from copy import deepcopy

from tools import *

# Constants
epsilon = 1e-9

# Helper functions

def product(iterable):
	return reduce(lambda x, y: x*y, iterable, 1)

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

# Phase 1: ST-HOSVD

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
		output["original_size"] = data.dtype.itemsize*data.size
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
		
		factor_matrix_size = factor_matrix.size
		for step in range(1, orthogonality_reconstruction_steps + 1):
			
			start_col = int(round(step/(orthogonality_reconstruction_steps + 1)*(factor_matrix.shape[1] - orthogonality_reconstruction_margin))) + orthogonality_reconstruction_margin
			end_col = int(round((step + 1)/(orthogonality_reconstruction_steps + 1)*(factor_matrix.shape[1] - orthogonality_reconstruction_margin))) + orthogonality_reconstruction_margin
			start_row = factor_matrix.shape[0] - start_col + orthogonality_reconstruction_margin
			
			if end_col == start_col or start_row >= factor_matrix.shape[0]:
				continue
			
			factor_matrix_size -= (end_col - start_col)*(factor_matrix.shape[0] - start_row)
			
		factor_matrices_size += factor_matrix_size*factor_matrix.itemsize
	
	compressed_size = factor_matrices_size + compressed["core_tensor"].dtype.itemsize*compressed["core_tensor"].size
	
	return compressed_size

def get_compression_factor_tucker(original, compressed):
	return (original.dtype.itemsize*original.size)/get_compress_tucker_size(compressed)

def print_compression_rate_tucker(original, compressed):
	
	original_size = original.dtype.itemsize*original.size
	compressed_size = get_compress_tucker_size(compressed)
	
	print("Method: Tucker")
	print("Data type:", compressed["core_tensor"].dtype)
	print("Original shape:", original.shape, "\toriginal size:", original_size)
	print("Compressed shape:", compressed["core_tensor"].shape, "\tcompressed size:", compressed_size)
	print("Compression ratio:", compressed_size/original_size)

# Phase 2: Orthgonality compression

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
	# - factor_matrices: the new factor matrices, stored in a dictionary stored 
	# - orthogonality_reconstruction_steps (only for "systems"): orthogonality_reconstruction_steps
	# - orthogonality_reconstruction_margin (only for "systems"): orthogonality_reconstruction_margin
	
	# a factor matrix is a distionary with keys:
	# - data:
	#	- "systems": a concatenation of all truncated columns into a 1D array (so column-major)
	#	- "householder": a concatenation of the columns of the lower triangle of the H-matrix (so H[1:, 0], H[2:, 1], ...)
	# - shape: shape of the original factor matrix
	# - tau (only if "householder"): tau-vector from QR-call
	# - blocks (only for "systems"): list of tuples of blocks stored in data, each tuple consists of (col2, truncation_row), the exclusive ending indices for columns and rows respectively
	# - offset (only if quantize=True): offset applied to factor matrix values
	# - scale (only if quantize=True): scale applied to factor matrix values after offset
	
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

def decompress_orthogonality(data, copy=False, correct_signs=False, renormalize=False):
	
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

# Original code, real mess from down here

quantization_steps = 256
starting_layer = 10
target_bits_after_point = 0
meta_quantization_step = 2**-target_bits_after_point

def compress_quantize1(data):
	
	# 
	
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
	
		
