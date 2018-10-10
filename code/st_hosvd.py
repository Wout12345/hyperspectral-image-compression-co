import numpy as np
from operator import itemgetter
from numpy import linalg

def compress(data, relative_target_error=0.05):
	
	# This function calculates the ST-HOSVD of the given 3D tensor (see https://epubs.siam.org/doi/abs/10.1137/110836067) using the mode order: 2, 0, 1
	# data should be a numpy array
	# relative_target_error is the desired Frobenius norm of the difference of the input data and the decompressed version of the compressed data divided by the Frobenius norm of the input data
	# returns ((U_1, ..., U_n), S), meaning the factor matrices and the core tensor
	
	factor_matrices = [None]*data.ndim
	S = data
	sizes = data.shape
	for mode in 2, 0, 1:
		
		# Process one mode
		
		if mode == 0:
			# Mode is already in front, no transposition required, convert to matrix of column vectors
		elif mode == data.ndim:
			# Mode is already in back, no transposition required, convert to matrix of row vectors
		else:
			# Transposition required, set current mode to back, convert to matrix of row vectors
			transposition_order = list(range(data.ndim))
			transposition_order[mode] = data.ndim - 1
			transposition_order[data.ndim - 1] = mode
			transposed = np.transpose(S, transposition_order)
		
		matrix = 

def decompress((factor_matrices, core_tensor)):
	
	# This function converts the given Tucker decomposition to the full tensor
	# returns the full tensor
