import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from copy import deepcopy
import json
import os

from tools import *
import st_hosvd

def sweep_parameters(dataset_name = "Pavia_Centre"):
	
	# Sweeps parameter space by brute-forcing combinations of parameters from sets of legal values
	
	# Constants
	reset = False
	measurements_path = "../measurements/parameters_measurements_%s.json"%dataset_name
	measurements_temp_path = measurements_path + ".tmp"
	if dataset_name == "Cuprite":
		data = load_cuprite()
	elif dataset_name == "Pavia_Centre":
		data = load_pavia()
	elif dataset_name == "Botswana":
		data = load_botswana()
	elif dataset_name == "Indian_Pines":
		data = load_indian_pines()
	else:
		raise Exception("Invalid dataset name!")
	
	# Reset measurements if necessary
	if reset or not os.path.isfile(measurements_path):
		measurements = {}
		with open(measurements_path, "w") as f:
			json.dump(measurements, f)
	else:
		with open(measurements_path, "r") as f:
			measurements = json.load(f)
	
	# Iterate through parameter space
	# Parameter iterables are defined in order of increasing error (decreasing compression factor)
	
	# Cuprite
	st_hosvd_relative_errors = [val/400 for val in range(2, 21)]
	core_tensor_parameters = list(range(16, 0, -1))
	factor_matrix_parameters = list(range(16, 0, -1))
	
	max_relative_error = 0.05 # Don't bother performing experiments that will surely give us a relative error beyond this bound
	for i, st_hosvd_relative_error in enumerate(st_hosvd_relative_errors):
		
		# Perform first compression step already
		compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, st_hosvd_relative_error))
		
		for j, core_tensor_parameter in enumerate(core_tensor_parameters):
			
			for k, factor_matrix_parameter in enumerate(factor_matrix_parameters):
				
				# Test one parameter combination
				
				# Only perform experiment if measurement hasn't been made yet
				key = str((st_hosvd_relative_error, core_tensor_parameter, factor_matrix_parameter))
				if not key in measurements:
					
					# Only consider measuring if it won't exceed the error bound
					indices = [i, j, k]
					break_flag = False
					for index in range(3):
						if (indices[index] > 0):
							indices[index] -= 1
							lower_key = str((st_hosvd_relative_errors[indices[0]], core_tensor_parameters[indices[1]], factor_matrix_parameters[indices[2]]))
							indices[index] += 1
							if lower_key in measurements and (measurements[lower_key] == "Error too large" or measurements[lower_key][0] >= max_relative_error):
								# Set this measurement to None, indicating that the error is already too large
								measurements[key] = "Error too large" # As string since we're storing this as JSON
								with open(measurements_temp_path, "w") as f:
									json.dump(measurements, f)
								os.rename(measurements_temp_path, measurements_path)
								break_flag = True
					if break_flag:
						continue
					
					# Perform actual measurements
					print("Measuring key %s"%key)
					compressed2 = st_hosvd.compress_quantize(compressed1, copy=True, endian="big", encoding_method="adaptive", allow_approximate_huffman=False, use_zlib=True, core_tensor_method="layered-constant-step", core_tensor_parameter=core_tensor_parameter, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", factor_matrix_parameter=factor_matrix_parameter, factor_matrix_columns_per_block=1, bits_amount_selection="norm-based")
					measurements[key] = (rel_error(data, st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.decompress_quantize(compressed2)))), st_hosvd.get_compression_factor_quantize(data, compressed2))
					with open(measurements_temp_path, "w") as f:
						json.dump(measurements, f)
					os.rename(measurements_temp_path, measurements_path)

#sweep_parameters("Indian_Pines")
sweep_parameters("Cuprite")
