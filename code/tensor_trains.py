from copy import deepcopy

import st_hosvd

def calculate_parameters(quality):
	# Parameter selection functions
	st_hosvd_rel_error = max(0.001, 0.9867464784343342*quality + 0.0011985623020501247)
	factor_matrix_parameter = max(9, int(round(-132.50545904417126*quality + 13.048260547290035)))
	return st_hosvd_rel_error, factor_matrix_parameter

def compress(data, quality=0.025, adaptive=False):
	
	# Compress using automatic parameter selection
	# Different from Tucker because of different parameter selection
	
	st_hosvd_rel_error, factor_matrix_parameter = calculate_parameters(quality)
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, st_hosvd_rel_error, reshape=True, method="tensor_trains"))
	
	if adaptive:
		
		# Initialization
		current_compression = st_hosvd.compress_quantize(deepcopy(compressed1), factor_matrix_parameter=factor_matrix_parameter)
		current_error = st_hosvd.rel_error(data, st_hosvd.decompress_tucker( st_hosvd.decompress_orthogonality( st_hosvd.decompress_quantize(current_compression) )) )
		current_size = st_hosvd.get_compress_quantize_size(current_compression)
		
		# Iterate as long as error stays below quality
		while True:
			
			alt_compression = st_hosvd.compress_quantize(deepcopy(compressed1), factor_matrix_parameter=factor_matrix_parameter - 1)
			alt_error = st_hosvd.rel_error(data, st_hosvd.decompress_tucker( st_hosvd.decompress_orthogonality( st_hosvd.decompress_quantize(alt_compression) )) )
			alt_size = st_hosvd.get_compress_quantize_size(alt_compression)
			
			if alt_error < quality and alt_size < current_size:
				# Continue
				factor_matrix_parameter -= 1
				current_compression = alt_compression
				current_error = alt_error
				current_size = alt_size
			else:
				break
		
		return current_compression
			
	else:
		
		# Non-adaptive, use initial factor matrix parameter value
		return st_hosvd.compress_quantize(compressed1, factor_matrix_parameter=factor_matrix_parameter)

def decompress(data):
	# Same as with Tucker decomposition, individual phases handle differences with tensor trains
	return st_hosvd.decompress(data)
	
