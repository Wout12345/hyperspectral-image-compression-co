import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import scipy.stats as stats
from time import time, clock
import gc
from copy import deepcopy
import json
import os
import math

from tools import *
import st_hosvd
import tensor_trains
import other_compression

# Constants
default_experiments_amount = 10

# Hoofdstuk 1: Inleiding

def save_cuprite_bands():
	print_cuprite_bands(6, "../tekst/images/cuprite_bands")

# Hoofdstuk 3: Methodologie

def save_image(data, path):
	image = np.sum(data, axis=2)
	imsave(path, np.rint((image - np.amin(image))/(np.amax(image) - np.amin(image))*255).astype(int))
	
def save_indian_pines_image():
	save_image(load_indian_pines(), "../tekst/images/indian_pines_sum.png")
	
def save_indian_pines_cropped_image():
	save_image(load_indian_pines_cropped(), "../tekst/images/indian_pines_cropped_sum.png")

def save_cuprite_image():
	save_image(load_cuprite(), "../tekst/images/cuprite_sum.png")

def save_cuprite_cropped_image():
	save_image(load_cuprite_cropped(), "../tekst/images/cuprite_cropped_sum.png")

def save_pavia_image():
	save_image(load_pavia(), "../tekst/images/pavia_sum.png")

def save_mauna_kea_raw_image():
	save_image(load_mauna_kea_raw(), "../tekst/images/mauna_kea_raw_sum.png")

def save_mauna_kea_image():
	save_image(load_mauna_kea(), "../tekst/images/mauna_kea_sum.png")

# Hoofdstuk 4: De Tuckerdecompositie

# Sectie 4.1.1

def mode_order():
	
	data = load_cuprite()
	mode_orders = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
	output = ""
	for mode_order in mode_orders:
		compressed = st_hosvd.compress_tucker(data, 0.025, mode_order=mode_order)
		output += "%s & %s \\\\ \\hline\n"%(mode_order, compressed["core_tensor"].shape)
	
	with open("../tekst/data/modevolgorde.tex", "w") as f:
		f.write(output)

# Sectie 4.1.2: Versnellen van de SVD

def gram_matrix():
	
	amount = default_experiments_amount
	data = load_cuprite()
	lines = []
	
	for name in ("Standaard", "Gram-matrix", "Gram-matrix met QR"):
		total_time = 0
		for _ in range(amount):
			print(name)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, compression_rank=(138, 192, 4), use_pure_gramian=(name == "Gram-matrix"), use_qr_gramian=(name == "Gram-matrix met QR"))
			total_time += output["total_cpu_time"]
		error = rel_error(data, st_hosvd.decompress_tucker(output["compressed"]))
		lines.append("{} & {:18.16f} & {:.1f} \\\\ \\hline\n".format(name, error, total_time/amount))
	
	with open("../tekst/data/gram-matrix.tex", "w") as f:
		f.write(lines[0])
		f.write(lines[1])
	
	with open("../tekst/data/gram-matrix-qr.tex", "w") as f:
		f.write(lines[1])
		f.write(lines[2])

def lanczos():
	
	amount = default_experiments_amount
	data = load_cuprite()
	lines = []
	
	for name in ("Gram-matrix", "Lanczos"):
		total_time = 0
		for _ in range(amount):
			print(name)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=(name == "Gram-matrix"), use_lanczos_gramian=(name == "Lanczos"))
			total_time += output["total_cpu_time"]
		error = rel_error(data, st_hosvd.decompress_tucker(output["compressed"]))
		shape = output["compressed"]["core_tensor"].shape
		compression_factor = st_hosvd.get_compression_factor_tucker(data, output["compressed"])
		lines.append("{} & {:.10f} & {:.1f} & {} & {:.1f} \\\\ \\hline\n".format(name, error, total_time/amount, shape, compression_factor))
	
	with open("../tekst/data/gram-matrix-lanczos.tex", "w") as f:
		f.write(lines[0])
		f.write(lines[1])

def lanczos_rank_comparison():
	
	data = load_cuprite()
	test_truncation_rank_limit = 15
	
	for name, rank in (("Gram-matrix", None), ("Lanczos", 6), ("Lanczos", 10), ("Lanczos", 15)):
		output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, print_progress=True, use_pure_gramian=(name == "Gram-matrix"), use_lanczos_gramian=(name == "Lanczos"), test_all_truncation_ranks=True, calculate_explicit_errors=True, test_truncation_rank_limit=test_truncation_rank_limit, compression_rank=None if rank is None else (163, 261, rank))
		errors = total_rank = output["truncation_rank_errors"][2]
		plt.plot(range(1, len(errors) + 1), errors, label=name if rank is None else name + " (%s)"%rank)
	
	plt.xlabel("Compressierang")
	plt.ylabel("Relatieve fout")
	plt.legend()
	plt.xlim(0, test_truncation_rank_limit)
	plt.savefig("../tekst/images/lanczos_rank_comparison.png")
	plt.close()

def randomized_svd_cuprite_test():
	
	amount = 1
	data = load_cuprite()
	lines = []
	modes = 3
	
	# Measurements
	rel_estimated_S_errors = np.empty((2, modes, amount))
	mode_times = np.empty((2, modes, amount))
	rel_errors = np.empty((2, amount))
	total_times = np.empty((2, amount))
	compression_rank_default = []
	population_sizes_default = []
	
	for method_index, (randomized_svd, samples_per_dimension) in enumerate([[False, 1000], [True, 5]]):
		for i in range(amount):
			print(randomized_svd)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=True, randomized_svd=randomized_svd, sample_ratio=0, samples_per_dimension=samples_per_dimension, store_rel_estimated_S_errors=True)
			if not randomized_svd:
				print(output["compressed"]["core_tensor"].shape)
				compression_rank_default = output["compressed"]["core_tensor"].shape
				population_sizes_default = output["population_sizes"]
			else:
				print(output["compressed"]["core_tensor"].shape, output["population_sizes"], output["sample_sizes"])
			
			for mode_index in range(modes):
				rel_estimated_S_errors[method_index][mode_index][i] = output["rel_estimated_S_errors"][mode_index]
				mode_times[method_index][mode_index][i] = output["cpu_times"][mode_index]
			rel_errors[method_index][i] = rel_error(data, st_hosvd.decompress_tucker(output["compressed"]))
			total_times[method_index][i] = output["total_cpu_time"]
	
	for mode_index, mode in enumerate(output["compressed"]["mode_order"]):
		lines.append("Mode & \\multicolumn{2}{|c|}{%s} \\\\ \\hline"%mode)
		lines.append("Originele rang & \\multicolumn{2}{|c|}{%s} \\\\ \\hline"%data.shape[mode])
		lines.append("Compressierang & %s & %s \\\\ \\hline"%(compression_rank_default[mode], output["compressed"]["core_tensor"].shape[mode]))
		lines.append("Populatiegrootte & %s & %s \\\\ \\hline"%(population_sizes_default[mode_index], output["population_sizes"][mode_index]))
		lines.append("Steekproefgrootte & N/A & %s \\\\ \\hline"%output["sample_sizes"][mode_index])
		lines.append("Relatieve fout $\\Sigma$ & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(rel_estimated_S_errors[0][mode_index]), np.mean(rel_estimated_S_errors[1][mode_index])))
		lines.append("Tijd voor mode (s) & {:.2f} & {:.2f}".format(np.mean(mode_times[0][mode_index]), np.mean(mode_times[1][mode_index])) + "\\\\ \\hhline{|=|=|=|}")
	
	lines.append("Relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(rel_errors[0]), np.mean(rel_errors[1])))
	lines.append("Totale tijd (s) & {:.2f} & {:.2f} \\\\ \\hline".format(np.mean(total_times[0]), np.mean(total_times[1])))
	
	with open("../tekst/data/randomized-svd-cuprite-test.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

def randomized_svd_cuprite_average():
	
	amount = default_experiments_amount
	data = load_cuprite()
	lines = []
	modes = 3
	
	# Measurements
	rel_errors = np.empty((2, amount))
	compression_factors = np.empty((2, amount))
	total_times = np.empty((2, amount))
	
	for method_index, (randomized_svd, samples_per_dimension) in enumerate([[False, 1000], [True, 5]]):
		for i in range(amount):
			print(randomized_svd)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=True, randomized_svd=randomized_svd, sample_ratio=0, samples_per_dimension=samples_per_dimension)
			
			rel_errors[method_index][i] = rel_error(data, st_hosvd.decompress_tucker(output["compressed"]))
			compression_factors[method_index][i] = st_hosvd.get_compression_factor_tucker(data, output["compressed"])
			total_times[method_index][i] = output["total_cpu_time"]
	
	lines.append("Min. relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.amin(rel_errors[0]), np.amin(rel_errors[1])))
	lines.append("Gem. relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(rel_errors[0]), np.mean(rel_errors[1])))
	lines.append("Max. relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.amax(rel_errors[0]), np.amax(rel_errors[1])))
	lines.append("Min. compressiefactor & {:.8f} & {:.8f} \\\\ \\hline".format(np.amin(compression_factors[0]), np.amin(compression_factors[1])))
	lines.append("Gem. compressiefactor & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(compression_factors[0]), np.mean(compression_factors[1])))
	lines.append("Max. compressiefactor & {:.8f} & {:.8f} \\\\ \\hline".format(np.amax(compression_factors[0]), np.amax(compression_factors[1])))
	lines.append("Totale tijd (s) & {:.2f} & {:.2f} \\\\ \\hline".format(np.mean(total_times[0]), np.mean(total_times[1])))
	
	with open("../tekst/data/randomized-svd-cuprite-average.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

def randomized_svd_pavia_ratios():
	
	# Plots errors and times for various sample ratios
	
	amount = 3
	sample_ratios = [0.8, 0.9, 1]
	measurements = []
	
	# Calculate errors
	data = load_pavia()
	for sample_ratio in sample_ratios:
		
		errors = np.zeros(amount)
		times = np.zeros(amount)
		for i in range(amount):
			print("Testing sample ratio", sample_ratio)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=True, randomized_svd=True, sample_ratio=sample_ratio, samples_per_dimension=0)
			errors[i] = rel_error(data, st_hosvd.decompress_tucker(output["compressed"]))
		measurements.append((sample_ratio, np.mean(errors),  np.mean(errors) - np.amin(errors), np.amax(errors) - np.mean(errors)))
	
	# Plot errors
	plt.errorbar([x[0] for x in measurements], [x[1] for x in measurements], yerr=[[x[2] for x in measurements], [x[3] for x in measurements]], capsize=5)
	print("measurements =", measurements)
	
	plt.xlabel("Steekproefratio")
	plt.ylabel("Relatieve fout")
	plt.savefig("../tekst/images/randomized_svd_pavia_ratios.png")
	plt.close()

def randomized_svd_pavia_test():
	
	amount = 1
	data = load_pavia()
	lines = []
	modes = 3
	
	# Measurements
	rel_estimated_S_errors = np.empty((2, modes, amount))
	mode_times = np.empty((2, modes, amount))
	rel_errors = np.empty((2, amount))
	total_times = np.empty((2, amount))
	compression_rank_default = []
	population_sizes_default = []
	
	for method_index, randomized_svd in enumerate([False, True]):
		for i in range(amount):
			print(randomized_svd)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=True, randomized_svd=randomized_svd, sample_ratio=0.2, samples_per_dimension=1000, store_rel_estimated_S_errors=True)
			if not randomized_svd:
				print(output["compressed"]["core_tensor"].shape)
				compression_rank_default = output["compressed"]["core_tensor"].shape
				population_sizes_default = output["population_sizes"]
			else:
				print(output["compressed"]["core_tensor"].shape, output["population_sizes"], output["sample_sizes"])
			
			for mode_index in range(modes):
				rel_estimated_S_errors[method_index][mode_index][i] = output["rel_estimated_S_errors"][mode_index]
				mode_times[method_index][mode_index][i] = output["cpu_times"][mode_index]
			rel_errors[method_index][i] = rel_error(data, st_hosvd.decompress_tucker(output["compressed"]))
			total_times[method_index][i] = output["total_cpu_time"]
	
	for mode_index, mode in enumerate(output["compressed"]["mode_order"]):
		lines.append("Mode & \\multicolumn{2}{|c|}{%s} \\\\ \\hline"%mode)
		lines.append("Originele rang & \\multicolumn{2}{|c|}{%s} \\\\ \\hline"%data.shape[mode])
		lines.append("Compressierang & %s & %s \\\\ \\hline"%(compression_rank_default[mode], output["compressed"]["core_tensor"].shape[mode]))
		lines.append("Populatiegrootte & %s & %s \\\\ \\hline"%(population_sizes_default[mode_index], output["population_sizes"][mode_index]))
		lines.append("Steekproefgrootte & N/A & %s \\\\ \\hline"%output["sample_sizes"][mode_index])
		lines.append("Relatieve fout $\\Sigma$ & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(rel_estimated_S_errors[0][mode_index]), np.mean(rel_estimated_S_errors[1][mode_index])))
		lines.append("Tijd voor mode (s) & {:.2f} & {:.2f}".format(np.mean(mode_times[0][mode_index]), np.mean(mode_times[1][mode_index])) + "\\\\ \\hhline{|=|=|=|}")
	
	lines.append("Relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(rel_errors[0]), np.mean(rel_errors[1])))
	lines.append("Totale tijd (s) & {:.2f} & {:.2f} \\\\ \\hline".format(np.mean(total_times[0]), np.mean(total_times[1])))
	
	with open("../tekst/data/randomized-svd-pavia-test.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

def randomized_svd_mauna_kea_factors():
	
	# Plots errors for various sample factors
	
	amount = 3
	sample_factors = []
	measurements = [(5, 0.02717880737646068, 0.00012861920327634016, 0.00015362455127352487), (10, 0.02502612674184154, 0.00024805838640050773, 0.00023499619384074327), (20, 0.024994331423124776, 9.843555069857096e-05, 9.79481495378097e-05)]
	
	# Calculate errors
	data = load_mauna_kea()
	norm = custom_norm(data)
	for sample_factor in sample_factors:
		
		errors = np.zeros(amount)
		for i in range(amount):
			print("Testing sample factor", sample_factor)
			errors[i] = custom_norm(st_hosvd.decompress_tucker(st_hosvd.compress_tucker(data, 0.025, use_pure_gramian=True, randomized_svd=True, samples_per_dimension=sample_factor, sample_ratio=0)).__isub__(data))/norm
			gc.collect()
		measurements.append((sample_factor, np.mean(errors),  np.mean(errors) - np.amin(errors), np.amax(errors) - np.mean(errors)))
		print(measurements[-1])
	
	# Plot errors
	plt.errorbar([x[0] for x in measurements], [x[1] for x in measurements], yerr=[[x[2] for x in measurements], [x[3] for x in measurements]], capsize=5)
	print("measurements =", measurements)
	
	plt.xlabel("Steekproeffactor")
	plt.ylabel("Relatieve fout")
	plt.savefig("../tekst/images/randomized_svd_mauna_kea_factors.png")
	plt.close()

def randomized_svd_mauna_kea_average():
	
	amount = default_experiments_amount
	data = load_mauna_kea()
	lines = []
	modes = 3
	
	# Measurements
	rel_errors = np.empty((2, amount))
	compression_factors = np.empty((2, amount))
	total_times = np.empty((2, amount))
	
	norm = custom_norm(data)
	for method_index, (randomized_svd, samples_per_dimension) in enumerate([[False, 1000], [True, 20]]):
		for i in range(amount):
			print(randomized_svd)
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=True, randomized_svd=randomized_svd, sample_ratio=0, samples_per_dimension=samples_per_dimension)
			
			rel_errors[method_index][i] = custom_norm(st_hosvd.decompress_tucker(output["compressed"]).__isub__(data))/norm
			gc.collect()
			compression_factors[method_index][i] = st_hosvd.get_compression_factor_tucker(data, output["compressed"])
			total_times[method_index][i] = output["total_cpu_time"]
	
	lines.append("Min. relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.amin(rel_errors[0]), np.amin(rel_errors[1])))
	lines.append("Gem. relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(rel_errors[0]), np.mean(rel_errors[1])))
	lines.append("Max. relatieve fout & {:.8f} & {:.8f} \\\\ \\hline".format(np.amax(rel_errors[0]), np.amax(rel_errors[1])))
	lines.append("Min. compressiefactor & {:.8f} & {:.8f} \\\\ \\hline".format(np.amin(compression_factors[0]), np.amin(compression_factors[1])))
	lines.append("Gem. compressiefactor & {:.8f} & {:.8f} \\\\ \\hline".format(np.mean(compression_factors[0]), np.mean(compression_factors[1])))
	lines.append("Max. compressiefactor & {:.8f} & {:.8f} \\\\ \\hline".format(np.amax(compression_factors[0]), np.amax(compression_factors[1])))
	lines.append("Totale tijd (s) & {:.2f} & {:.2f} \\\\ \\hline".format(np.mean(total_times[0]), np.mean(total_times[1])))
	
	with open("../tekst/data/randomized-svd-mauna-kea-average.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

# Sectie 4.2.1: Methode met stelsels

def orthogonality_compression_basic():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	for name, quantize in (("Geen quantisatie", None), ("Quantisatie", 16)):
		compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=False)
		plt.plot(cols, np.linalg.norm(reference - compressed2["factor_matrices"][factor_matrix_index], axis=0), label=name)
		
	plt.xlabel("Kolomindex")
	plt.ylabel("Relatieve fout")
	plt.legend()
	plt.savefig("../tekst/images/orthogonality_compression_basic.png")
	plt.close()

def orthogonality_compression_basic_matrix():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	for name, quantize in (("Geen quantisatie", None), ("Quantisatie", 16)):
		compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=False)
		plt.plot(cols, [rel_error(reference[:, :i + 1], compressed2["factor_matrices"][factor_matrix_index][:, :i + 1]) for i in cols], label=name)
		
	plt.xlabel("Maximale kolomindex (exclusief)")
	plt.ylabel("Relatieve fout")
	plt.legend()
	plt.savefig("../tekst/images/orthogonality_compression_basic_matrix.png")
	plt.close()

def orthogonality_compression_basic_inverse_norm_kept_values():
	
	# Not used in text
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	for name, quantize in (("Geen quantisatie", None), ("Quantisatie", 16)):
		compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=False)
		plt.plot(cols, 1/np.flip(np.linalg.norm(np.triu(np.flip(compressed2["factor_matrices"][factor_matrix_index], axis=1)), axis=0)), label=name)
		
	plt.xlabel("Kolomindex")
	plt.ylabel("Inverse norm van behouden waarden")
	plt.legend()
	plt.savefig("../tekst/images/orthogonality_compression_basic_inverse_norm_kept_values.png")
	plt.close()

def orthogonality_compression_basic_error():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	reference = data.copy()
	for name, quantize in (("Geen quantisatie", None), ("Quantisatie", 16)):
		decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=False))
		print(name + ":", "{:.4f}".format(rel_error(reference, decompressed)))

def orthogonality_compression_basic_timing():
	
	amount = default_experiments_amount
	
	for name, loader in (("Cuprite", load_cuprite), ("Pavia Centre", load_pavia)):
		data = loader()
		compressed1 = st_hosvd.compress_tucker(data, 0.025)
		total_time = 0
		for i in range(amount):
			print("Testing", name, i)
			start = clock()
			compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=16, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=False)
			total_time += clock() - start
		print(name, "{:.2f}".format(total_time/amount))

# Sectie 4.2.1.1: Hernormalisatie

def orthogonality_compression_norms():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	cols = range(compressed1["factor_matrices"][factor_matrix_index].shape[1])
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=16, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=False)
	plt.plot(cols, np.linalg.norm(compressed2["factor_matrices"][factor_matrix_index], axis=0))
		
	plt.xlabel("Kolomindex")
	plt.ylabel("Norm")
	plt.savefig("../tekst/images/orthogonality_compression_norms1.png")
	plt.ylim(0.975, 1.2)
	plt.savefig("../tekst/images/orthogonality_compression_norms2.png")
	plt.close()

def orthogonality_compression_renormalization():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=16, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=True)
	plt.plot(cols, np.linalg.norm(reference - compressed2["factor_matrices"][factor_matrix_index], axis=0))
	print("Relative error:", "{:.4f}".format(rel_error(data, st_hosvd.decompress_tucker(compressed2))))
		
	plt.xlabel("Kolomindex")
	plt.ylabel("Relatieve fout")
	plt.savefig("../tekst/images/orthogonality_compression_renormalization.png")
	plt.close()

# Sectie 4.2.1.2: Blokken

def orthogonality_compression_blocks():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=16, orthogonality_reconstruction_steps=10, orthogonality_reconstruction_margin=0, method="systems"), renormalize=True)
	plt.plot(cols, np.linalg.norm(reference - compressed2["factor_matrices"][factor_matrix_index], axis=0))
	print("Relative error:", "{:.4f}".format(rel_error(data, st_hosvd.decompress_tucker(compressed2))))
		
	plt.xlabel("Kolomindex")
	plt.ylabel("Relatieve fout")
	plt.savefig("../tekst/images/orthogonality_compression_blocks.png")
	plt.close()

def orthogonality_compression_blocks_timing():
	
	amount = default_experiments_amount
	
	for name, loader in (("Cuprite", load_cuprite), ("Pavia Centre", load_pavia)):
		data = loader()
		compressed1 = st_hosvd.compress_tucker(data, 0.025)
		total_time = 0
		for i in range(amount):
			print("Testing", name, i)
			start = clock()
			compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=16, orthogonality_reconstruction_steps=10, orthogonality_reconstruction_margin=0, method="systems"), renormalize=True)
			total_time += clock() - start
		print(name, "{:.2f}".format(total_time/amount))

# Sectie 4.2.1.3: Marge

def orthogonality_compression_margin():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=16, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=3, method="systems"), renormalize=True)
	plt.plot(cols, np.linalg.norm(reference - compressed2["factor_matrices"][factor_matrix_index], axis=0))
	print("Relative error:", "{:.4f}".format(rel_error(data, st_hosvd.decompress_tucker(compressed2))))
		
	plt.xlabel("Kolomindex")
	plt.ylabel("Relatieve fout")
	plt.savefig("../tekst/images/orthogonality_compression_margin.png")
	plt.close()

# Sectie 4.2.1.5: Samenvatting

def orthogonality_compression_systems_summary():
	
	amount = default_experiments_amount
	measurements = []
	datasets = (("Cuprite", load_cuprite), ("Pavia Centre", load_pavia))
	settings = (("Standaard", False, 500, 0), ("Hernormalisatie (HN)", True, 500, 0), ("HN + 10 blokken", True, 10, 0), ("HN + marge 3", True, 500, 3), ("HN + 10 blokken + marge 3", True, 10, 3))
	
	# Perform experiments
	for name, loader in datasets:
		measurements.append({"times": [], "errors": []})
		data = loader()
		compressed1 = st_hosvd.compress_tucker(data, 0.025)
		for setting_name, renormalize, steps, margin in settings:
			total_time = 0
			for i in range(amount):
				print("Testing", name, "with setting", setting_name, "on run", i)
				copy = deepcopy(compressed1)
				start = clock()
				compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(copy, quantize=16, orthogonality_reconstruction_steps=steps, orthogonality_reconstruction_margin=margin, method="systems"), renormalize=renormalize)
				total_time += clock() - start
				if i == 0:
					# Calculate error
					decompressed = st_hosvd.decompress_tucker(compressed2)
					measurements[-1]["errors"].append(rel_error(data, decompressed))
			measurements[-1]["times"].append(total_time/amount)
			print(measurements)
	
	# Construct lines
	lines = []
	for i, (setting_name, _, _, _) in enumerate(settings):
		lines.append("{} & {:.4f} & {:.4f} & {:.3f} & {:.3f} \\\\ \\hline".format(setting_name, measurements[0]["errors"][i], measurements[1]["errors"][i], measurements[0]["times"][i], measurements[1]["times"][i]))
	
	with open("../tekst/data/orthogonality-compression-systems-summary.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

# Sectie 4.2.3: Methode met Householder-reflecties

def orthogonality_compression_householder_summary():
	
	amount = default_experiments_amount
	measurements = []
	datasets = (("Cuprite", load_cuprite), ("Pavia Centre", load_pavia))
	settings = (("Referentie", False, None), ("Reflectoren", True, None), ("Reflectoren + 16-bit quantisatie", True, 16))
	
	# Perform experiments
	for name, loader in datasets:
		measurements.append({"times": [], "errors": []})
		data = loader()
		compressed1 = st_hosvd.compress_tucker(data, 0.025)
		for setting_name, compress_orthogonality, quantize in settings:
			if compress_orthogonality:
				total_time = 0
				for i in range(amount):
					print("Testing", name, "with setting", setting_name, "on run", i)
					copy = deepcopy(compressed1)
					start = clock()
					compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(copy, quantize=quantize, method="householder"), renormalize=False)
					total_time += clock() - start
					if i == 0:
						# Calculate error
						decompressed = st_hosvd.decompress_tucker(compressed2)
						measurements[-1]["errors"].append(rel_error(data, decompressed))
				measurements[-1]["times"].append(total_time/amount)
				print(measurements)
			else:
				measurements[-1]["times"].append(0)
				measurements[-1]["errors"].append(rel_error(data, st_hosvd.decompress_tucker(compressed1)))
	
	# Construct lines
	lines = []
	for i, (setting_name, compress_orthogonality, _) in enumerate(settings):
		if compress_orthogonality:
			lines.append("{} & {:.8f} & {:.10f} & {:.3f} & {:.3f} \\\\ \\hline".format(setting_name, measurements[0]["errors"][i], measurements[1]["errors"][i], measurements[0]["times"][i], measurements[1]["times"][i]))
		else:
			lines.append("{} & {:.8f} & {:.10f} & - & - \\\\ \\hline".format(setting_name, measurements[0]["errors"][i], measurements[1]["errors"][i]))
	
	with open("../tekst/data/orthogonality-compression-householder-summary.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

def orthogonality_compression_householder_quantisation_bits():
	
	data = load_cuprite()
	reference = deepcopy(data)
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	quantization_bits_amounts = range(5, 9)
	
	for name, renormalize in (("Geen hernormalisatie", False), ("Hernormalisatie", True)):
		rel_errors = []
		for quantization_bits in quantization_bits_amounts:
			compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=quantization_bits, method="householder"), renormalize=renormalize)
			rel_errors.append(rel_error(reference, st_hosvd.decompress_tucker(compressed2)))
		plt.plot(quantization_bits_amounts, rel_errors, label=name)
		
	plt.xlabel("Quantisatiebits")
	plt.ylabel("Relatieve fout")
	plt.legend()
	plt.savefig("../tekst/images/orthogonality_compression_householder_quantisation_bits.png")
	plt.close()

# Sectie 4.3.1: Quantisatie kerntensor

def core_tensor_values_distribution():
	
	data = load_cuprite()
	compressed = st_hosvd.compress_tucker(data, 0.025)
	core_tensor = compressed["core_tensor"]
	mins = []
	means = []
	maxs = []
	layers = range(max(core_tensor.shape))
	
	for layer in layers:
		abs_values = np.abs(st_hosvd.extract_layers(core_tensor, layer, layer + 1))
		mins.append(np.amin(abs_values))
		means.append(np.mean(abs_values))
		maxs.append(np.amax(abs_values))
	
	plt.plot(layers, maxs, label="Maximum")
	plt.plot(layers, means, label="Gemiddeld")
	plt.plot(layers, mins, label="Minimum")
	plt.xlabel("Laag")
	plt.ylabel("Absolute waarde")
	plt.yscale("log")
	plt.legend()
	plt.savefig("../tekst/images/core_tensor_values_distribution.png")
	plt.close()

def core_tensor_unquantized_portion():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	quantization_bits_amounts = range(10, 17)
	
	for unquantized_layers in range(4):
		core_tensor = compressed1["st_hosvd"]["core_tensor"]
		core_tensor_unquantized_rel_norm = (np.linalg.norm(core_tensor[(slice(max(0, unquantized_layers - 1)),)*core_tensor.ndim]) + np.linalg.norm(core_tensor[(slice(unquantized_layers),)*core_tensor.ndim]))/(2*np.linalg.norm(core_tensor))
		rel_errors = []
		compression_factors = []
		for quantization_bits in quantization_bits_amounts:
			print("Testing %s bits with %s unquantized layers"%(quantization_bits, unquantized_layers))
			compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian="little", encoding_method="default", use_zlib=False, core_tensor_method="constant", core_tensor_parameter=quantization_bits, core_tensor_unquantized_rel_norm=core_tensor_unquantized_rel_norm, factor_matrix_method="constant", factor_matrix_parameter=16)
			rel_errors.append(rel_error(data, st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.decompress_quantize(compressed2)))))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed2))
			if unquantized_layers < 2 and (unquantized_layers == 0 or quantization_bits < 14):
				plt.annotate(str(quantization_bits), (rel_errors[-1], compression_factors[-1]))
		plt.plot(rel_errors, compression_factors, label="%s ongequantiseerde lagen"%unquantized_layers)
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.legend()
	plt.savefig("../tekst/images/core_tensor_unquantized_portion.png")
	plt.close()

def core_tensor_quantization_comparison():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	
	for name, method, quantization_bits_amounts, label_offset_x, label_offset_y in (("Globaal", "constant", range(8, 13), 0, 0), ("Gelaagd (constant)", "layered-constant-bits", range(4, 13), 0, -3), ("Gelaagd (variabel)", "layered-constant-step", range(7, 17), -0.0003, 0)):
		
		rel_errors = []
		compression_factors = []
		for quantization_bits in quantization_bits_amounts:
			print("Testing method %s with %s bits"%(name, quantization_bits))
			compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian="little", encoding_method="default", use_zlib=False, core_tensor_method=method, core_tensor_parameter=quantization_bits, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="constant", factor_matrix_parameter=16)
			rel_errors.append(rel_error(data, st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.decompress_quantize(compressed2)))))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed2))
			plt.annotate(str(quantization_bits), (rel_errors[-1] + label_offset_x, compression_factors[-1] + label_offset_y))
		plt.plot(rel_errors, compression_factors, label=name)
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.legend()
	plt.savefig("../tekst/images/core_tensor_quantization_comparison.png")
	plt.close()

# Sectie 4.3.2: Quantisatie factormatrices

def factor_matrix_quantization_block_cols():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	quantization_bits_amounts = range(7, 11)
	
	for block_cols in (1, 2, 3, 4):
		rel_errors = []
		compression_factors = []
		for quantization_bits in quantization_bits_amounts:
			print("Testing block cols %s with %s bits"%(block_cols, quantization_bits))
			compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian="little", encoding_method="default", use_zlib=False, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", factor_matrix_parameter=quantization_bits, factor_matrix_columns_per_block=block_cols, bits_amount_selection="norm-based")
			rel_errors.append(rel_error(data, st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.decompress_quantize(compressed2)))))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed2))
			if quantization_bits < 9 or (quantization_bits == 9 and block_cols < 4) or (quantization_bits == 10 and block_cols < 2):
				plt.annotate(str(quantization_bits), (rel_errors[-1], compression_factors[-1]))
		plt.plot(rel_errors, compression_factors, label="%s reflectoren/blok"%block_cols if block_cols != 1 else "1 reflector/blok")
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.legend()
	plt.savefig("../tekst/images/factor_matrix_quantization_block_cols.png")
	plt.close()

def factor_matrix_quantization_comparison():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	
	for name, method, bits_amount_selection, quantization_bits_amounts, annotate_bound, label_x_offset, label_y_offset in (("Globaal", "constant", None, range(8, 12), 12, 0, 0), ("Gelaagd (constant)", "layered", "constant", range(4, 9), 12, 0.0002, 0), ("Gelaagd (norm)", "layered", "norm-based", range(5, 16), 16, -0.0004, 0), ("Gelaagd (norm + dimensie)", "layered", "norm-height-based", range(5, 12), 12, 0, 0)):
		
		rel_errors = []
		compression_factors = []
		for quantization_bits in quantization_bits_amounts:
			print("Testing method %s with %s bits"%(name, quantization_bits))
			compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian="little", encoding_method="default", use_zlib=False, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method=method, factor_matrix_parameter=quantization_bits, factor_matrix_columns_per_block=1, bits_amount_selection=bits_amount_selection)
			rel_errors.append(rel_error(data, st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.decompress_quantize(compressed2)))))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed2))
			if quantization_bits < annotate_bound:
				plt.annotate(str(quantization_bits), (rel_errors[-1] + label_x_offset, compression_factors[-1] + label_y_offset))
		plt.plot(rel_errors, compression_factors, label=name)
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.legend()
	plt.savefig("../tekst/images/factor_matrix_quantization_comparison.png")
	plt.close()

# Sectie 4.4: Encodering en lossless compressie

def distribution_quantized_values():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	compressed2 = st_hosvd.compress_orthogonality(compressed1, method="householder")
	
	st_hosvd.plot_save_path = "../tekst/images/distribution_quantized_values_layer_"
	st_hosvd.plot_counter = 30
	compressed3 = st_hosvd.compress_quantize(compressed2, endian="little", encoding_method="huffman", use_zlib=True, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", factor_matrix_parameter=10, factor_matrix_columns_per_block=1, bits_amount_selection="norm-based", plot_frequencies=range(30, 34))

def encoding_comparison1():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	
	lines = []
	for encoding_method_name, encoding_method in (("Standaard", "default"), ("Gray-code", "graycode"), ("Huffman", "huffman"), ("Adaptief", "adaptive")):
		compression_sizes = []
		for use_zlib in (False, True):
			for endian in ("little", "big"):
				print("Testing encoding method %s, use_zlib %s, endian %s"%(encoding_method_name, use_zlib, endian))
				compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian=endian, encoding_method=encoding_method, allow_approximate_huffman=True, use_zlib=use_zlib, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", factor_matrix_parameter=10, factor_matrix_columns_per_block=1, bits_amount_selection="norm-based")
				compression_sizes.append(st_hosvd.get_compress_quantize_size(compressed2))
	
		# Construct line
		lines.append("%s & "%encoding_method_name + " & ".join([str(size) for size in compression_sizes]) + " \\\\ \\hline")
	
	with open("../tekst/data/encoding-comparison1.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

def encoding_comparison2():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	
	lines = []
	for encoding_method_name, encoding_method in (("Standaard", "default"), ("Gray-code", "graycode"), ("Huffman", "huffman"), ("Adaptief", "adaptive")):
		compression_sizes = []
		print("Testing encoding method %s"%encoding_method_name)
		compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian="big", encoding_method=encoding_method, allow_approximate_huffman=True, use_zlib=True, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", factor_matrix_parameter=10, factor_matrix_columns_per_block=1, bits_amount_selection="norm-based")
		data_size = st_hosvd.get_compress_quantize_size(compressed2, count_trees=False)
		total_size = st_hosvd.get_compress_quantize_size(compressed2, count_trees=True)
		trees_size = total_size - data_size
		
		# Construct line
		lines.append("{} & {} & {} & {} \\\\ \\hline".format(encoding_method_name, data_size, trees_size, total_size))
	
	with open("../tekst/data/encoding-comparison2.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

def encoding_timing():
	
	amount = default_experiments_amount
	data = load_cuprite()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025))
	settings = (("Standaard", "default", False), ("Gray-code", "graycode", False), ("Huffman", "huffman", False), ("Adaptief (met BHC)", "adaptive", True), ("Adaptief (zonder BHC)", "adaptive", False))
	
	compression_sizes = []
	times = []
	for encoding_method_name, encoding_method, allow_approximate_huffman in settings:
		total_time = 0
		for i in range(amount):
			print("Testing encoding method %s %s"%(encoding_method_name, i))
			copy = deepcopy(compressed1)
			start_time = clock()
			compressed2 = st_hosvd.compress_quantize(copy, endian="big", encoding_method=encoding_method, allow_approximate_huffman=allow_approximate_huffman, use_zlib=True, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", factor_matrix_parameter=10, factor_matrix_columns_per_block=1, bits_amount_selection="norm-based")
			total_time += clock() - start_time
		compression_sizes.append(st_hosvd.get_compress_quantize_size(compressed2))
		times.append(total_time/amount)
	
	# Construct lines
	lines = []
	for i, (encoding_method_name, _, _) in enumerate(settings):
		lines.append("{} & {:.4f} & {:.2f} \\\\ \\hline".format(encoding_method_name, compression_sizes[i]/min(compression_sizes), times[i]))
	
	with open("../tekst/data/encoding-timing.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

# Sectie 4.5: Afstellen van de parameters
# Own experiments

def test_parameter_functions(dataset_name, adaptive, method="tucker"):
	
	# Initialization
	if dataset_name == "Cuprite":
		data = load_cuprite_cropped() if method == "tensor_trains" else load_cuprite()
	elif dataset_name == "Indian_Pines":
		data = load_indian_pines_cropped() if method == "tensor_trains" else load_indian_pines()
	else:
		raise Exception("Invalid dataset name!")
	measurements_path = "../measurements/test_parameter_functions_%s_%s_%s.json"%(dataset_name, adaptive, method)
	measurements_temp_path = measurements_path + ".tmp"
	measurements = []
	
	# Test quality values
	for quality_index in range(5, 51):
		quality = quality_index/1000
		print("Testing quality", quality)
		if method == "tucker":
			compressed = st_hosvd.compress(data, quality=quality, adaptive=adaptive)
		elif method == "tensor_trains":
			compressed = tensor_trains.compress(data, quality=quality, adaptive=adaptive)
		measurements.append((rel_error(data, st_hosvd.decompress(compressed)), st_hosvd.get_compression_factor_quantize(data, compressed)))
		with open(measurements_temp_path, "w") as f:
			json.dump(measurements, f)
		os.rename(measurements_temp_path, measurements_path)

def plot_sweep_results(dataset_name="Cuprite", annotate=False, plot_test_functions=False):
	
	# Load measurements
	measurements_path = "../measurements/parameters_measurements_%s.json"%dataset_name
	with open(measurements_path, "r") as f:
		measurements = json.load(f)
	measurements = {key: value for key, value in measurements.items() if not type(value) is str}
	
	# Plot all measurements
	points = measurements.values()
	errors = [point[0] for point in points]
	factors = [point[1] for point in points]
	plt.scatter(errors, factors, c="C0")
	
	# Plot best curve
	# Filter all points covered by another one
	mask = {key: True for key in measurements}
	for key1 in measurements:
		if mask[key1]:
			error1, factor1 = measurements[key1]
			for key2 in measurements:
				if key2 != key1 and mask[key2]:
					# Point is different and still marked as in
					error2, factor2 = measurements[key2]
					if error2 > error1 and factor2 < factor1:
						# Point isn't useful, filter out
						mask[key2] = False
	
	# Plot filtered points with annotations
	filtered_errors = []
	filtered_factors = []
	filtered_keys = []
	for key in measurements:
		if mask[key]:
			error, factor = measurements[key]
			filtered_errors.append(error)
			filtered_factors.append(factor)
			filtered_keys.append(key)
	filtered_errors, filtered_factors_and_keys = zip(*sorted(zip(filtered_errors, zip(filtered_factors, filtered_keys))))
	filtered_factors, filtered_keys = zip(*filtered_factors_and_keys)
	plt.plot(filtered_errors, filtered_factors, "C0")
	
	if annotate:
		last_st_hosvd_relative_error = None
		last_core_tensor_parameter = None
		for i, key in enumerate(filtered_keys):
			words = key[1:-1].split(", ")
			st_hosvd_relative_error = float(words[0])
			core_tensor_parameter = int(words[1])
			if st_hosvd_relative_error != last_st_hosvd_relative_error or core_tensor_parameter != last_core_tensor_parameter:
				last_st_hosvd_relative_error = st_hosvd_relative_error
				last_core_tensor_parameter = core_tensor_parameter
				plt.annotate(key, (filtered_errors[i], filtered_factors[i]))
	
	# Plot test functions measurements
	if plot_test_functions:
		test_functions_measurements_path = "../measurements/test_parameter_functions_%s.json"%dataset_name
		with open(test_functions_measurements_path, "r") as f:
			test_functions_measurements = json.load(f)
		plt.plot([point[0] for point in test_functions_measurements], [point[1] for point in test_functions_measurements], color="C1")
	
	# Finish plot
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.show()
	plt.close()
	
	# Apply linear regression to first parameter and logarithms of second/third parameter
	
	# Plot parameter values per error and try to approximate
	names = ["1000*(relatieve doelfout ST-HOSVD)", "Bits-parameter kerntensor", "Bits-parameter factormatrices"]
	for i in range(3):
		values = [float(key[1:-1].split(", ")[i])*(1000 if i == 0 else 1) for key in filtered_keys]
		plt.plot(filtered_errors, values, label=names[i], color="C%s"%i, linestyle="-")
		# Approximation
		if i == 0:
			indices = np.array(range(len(filtered_errors)))
		elif i == 1:
			indices = np.array(filtered_errors) < 0.016
		elif i == 2:
			indices = np.array(range(len(filtered_errors)))
		errors_for_regression = np.array(filtered_errors)[indices]
		values_for_regression = np.array(values)[indices]
		if i == 0:
			slope, intercept, _, _, _ = stats.linregress(errors_for_regression, values_for_regression)
			approx = slope*np.array(filtered_errors) + intercept
			print(slope, intercept)
		elif i == 1:
			slope, intercept, _, _, _ = stats.linregress(errors_for_regression, values_for_regression)
			print(slope, intercept)
			approx = [round(slope*error + intercept) if error < 0.016 else 10 for error in filtered_errors]
		elif i == 2:
			slope, intercept, _, _, _ = stats.linregress(errors_for_regression, values_for_regression)
			slope = -228.571428571
			intercept = 13 - slope*0.005
			print(slope, intercept)
			approx = [round(slope*error + intercept) if error < 0.04 else 5 for error in filtered_errors]
		plt.plot(filtered_errors, approx, color="C%s"%i, linestyle=":")
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Parameterwaarde")
	plt.legend()
	plt.show()
	plt.close()

# Helper functions

def filter_points(dataset_name, tensor_trains=False):
	
	# Load measurements
	method = "tensor_trains" if tensor_trains else "tucker"
	measurements_path = "../measurements/parameters_measurements_%s_%s.json"%(method, dataset_name)
	with open(measurements_path, "r") as f:
		measurements = json.load(f)
	measurements = {key: value for key, value in measurements.items() if not type(value) is str}
	
	# Calculate filter
	mask = {key: True for key in measurements}
	for key1 in measurements:
		if mask[key1]:
			error1, factor1 = measurements[key1]
			for key2 in measurements:
				if key2 != key1 and mask[key2]:
					# Point is different and still marked as in
					error2, factor2 = measurements[key2]
					if error2 > error1 and factor2 < factor1:
						# Point isn't useful, filter out
						mask[key2] = False
	
	# Extract points to lists
	filtered_errors = []
	filtered_factors = []
	filtered_keys = []
	for key in measurements:
		if mask[key]:
			error, factor = measurements[key]
			filtered_errors.append(error)
			filtered_factors.append(factor)
			filtered_keys.append(key)
	
	# Sort based on errors
	filtered_errors, filtered_factors_and_keys = zip(*sorted(zip(filtered_errors, zip(filtered_factors, filtered_keys))))
	filtered_factors, filtered_keys = zip(*filtered_factors_and_keys)
	
	return filtered_errors, filtered_factors, filtered_keys

# Experiments for text

def plot_all_sweep_points_cuprite():
	
	# Load measurements
	dataset_name = "Cuprite"
	measurements_path = "../measurements/parameters_measurements_tucker_%s.json"%dataset_name
	with open(measurements_path, "r") as f:
		measurements = json.load(f)
	measurements = {key: value for key, value in measurements.items() if not type(value) is str}
	
	# Plot all measurements
	points = measurements.values()
	errors = [point[0] for point in points]
	factors = [point[1] for point in points]
	plt.scatter(errors, factors)
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.savefig("../tekst/images/all_sweep_points_cuprite.png")
	plt.close()

def plot_filtered_sweep_points_cuprite():
	
	errors, factors, _ = filter_points("Cuprite")
	plt.plot(errors, factors)
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.savefig("../tekst/images/filtered_sweep_points_cuprite.png")
	plt.close()

def plot_filtered_sweep_points_parameters():
	
	cuprite_errors, _, cuprite_keys = filter_points("Cuprite")
	indian_pines_errors, _, indian_pines_keys = filter_points("Indian_Pines")
	
	for i, parameter_name in enumerate(("RDS", "BPK", "BPF")):
		
		# Plot values
		for name, errors, keys in (("Cuprite", cuprite_errors, cuprite_keys), ("Indian Pines", indian_pines_errors, indian_pines_keys)):
			values = [float(key[1:-1].split(", ")[i]) for key in keys]
			plt.plot(errors, values, label=name)
		
		# Plot selection function
		qualities = np.linspace(0.005, 0.05, num=46)
		plt.plot(qualities, [st_hosvd.calculate_parameters(quality)[i] for quality in qualities], label="Selectiefunctie")
			
		plt.xlabel("Relatieve fout")
		plt.ylabel(parameter_name)
		plt.legend()
		plt.savefig("../tekst/images/filtered_sweep_points_%s.png"%parameter_name)
		plt.close()

def plot_parameter_functions_results(include_adaptive=True, method="tucker"):
	
	for dataset_name in ("Indian_Pines", "Cuprite"):
		
		# Sample optima
		errors, factors, _ = filter_points(dataset_name, tensor_trains=(method=="tensor_trains"))
		plt.plot(errors, factors, label="Steekproefoptima")
		
		# Selection function results
		settings = (("Niet-adaptief", False), ("Adaptief", True)) if include_adaptive else (("Met selectiefuncties", False),)
		for label, adaptive in settings:
			
			test_functions_measurements_path = "../measurements/test_parameter_functions_%s_%s_%s.json"%(dataset_name, adaptive, method)
			with open(test_functions_measurements_path, "r") as f:
				test_functions_measurements = json.load(f)
			plt.plot([point[0] for point in test_functions_measurements], [point[1] for point in test_functions_measurements], label=label)
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/parameter_functions_results%s_%s_%s.png"%("_including_adaptive" if include_adaptive else "", dataset_name, method))
		plt.close()

def parameter_selection_results_big_datasets():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=9)
	
	for dataset_name, loader in (("Mauna_Kea", load_mauna_kea), ("Pavia_Centre", load_pavia), ):
		
		data = loader()
		for name, adaptive in (("Adaptief", True), ("Niet-adaptief", False), ):
			rel_errors = []
			compression_factors = []
			for rel_target_error in rel_target_errors:
				print("Testing dataset %s with method %s with target error %s"%(dataset_name, name, rel_target_error))
				compressed = st_hosvd.compress(data, rel_target_error, adaptive=adaptive)
				rel_errors.append(rel_error(data, st_hosvd.decompress(compressed)))
				compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
			plt.plot(rel_errors, compression_factors, label=name)
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/parameter_selection_results_%s.png"%dataset_name)
		plt.close()

def plot_adaptive_timings():
	
	amount = default_experiments_amount
	qualities = np.linspace(0.01, 0.05, num=5)
	
	for dataset_name, loader in (("Cuprite", load_cuprite), ("Indian_Pines", load_indian_pines)):
		
		# Load dataset
		data = loader()
		
		# Test both non-adaptive and adaptive
		settings = (("Niet-adaptief", False), ("Adaptief", True))
		for label, adaptive in settings:
			times = []
			for quality in qualities:
				total_time = 0
				for i in range(amount):
					print("Testing dataset %s adaptive %s quality %s"%(dataset_name, adaptive, quality))
					start_time = clock()
					compressed = st_hosvd.compress(data, quality=quality, adaptive=adaptive)
					total_time += clock() - start_time
				times.append(total_time/amount)
			plt.plot(qualities, times, label=label)
			
		plt.xlabel("Kwaliteitsparameter")
		plt.ylabel("Compressietijd (s)")
		plt.legend()
		plt.savefig("../tekst/images/adaptive_timings_%s.png"%dataset_name)
		plt.close()

# Hoofdstuk 5: Compressie na hervorming

# Sectie 5.2: Tucker na hervorming

def reshaped_tucker_st_hosvd_results():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=5)
	
	for dataset_name, loader in (("Indian_Pines", load_indian_pines_cropped), ("Cuprite", load_cuprite_cropped), ("Pavia_Centre", load_pavia), ("Mauna_Kea", load_mauna_kea)):
		
		data = loader()
		for name, reshape in (("Tucker (zonder hervorming)", False), ("Tucker (met hervorming)", True)):
			rel_errors = []
			compression_factors = []
			for rel_target_error in rel_target_errors:
				print("Testing dataset %s with method %s with target error %s"%(dataset_name, name, rel_target_error))
				compressed = st_hosvd.compress_tucker(data, rel_target_error, reshape=reshape)
				rel_errors.append(st_hosvd.rel_error(data, st_hosvd.decompress_tucker(compressed)))
				compression_factors.append(st_hosvd.get_compression_factor_tucker(data, compressed))
			plt.plot(rel_errors, compression_factors, label=name)
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/reshaped_tucker_st_hosvd_results_%s.png"%dataset_name)
		plt.close()

# Sectie 5.3: Tensor trains

def tensor_trains_st_hosvd_results():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=5)
	
	for dataset_name, loader in (("Indian_Pines", load_indian_pines_cropped), ("Cuprite", load_cuprite_cropped), ("Pavia_Centre", load_pavia), ("Mauna_Kea", load_mauna_kea),):
		
		data = loader()
		for name, method, reshape in (("Tucker", "tucker", False), ("Tensor trains", "tensor_trains", True)):
			rel_errors = []
			compression_factors = []
			for rel_target_error in rel_target_errors:
				print("Testing dataset %s with method %s with target error %s"%(dataset_name, name, rel_target_error))
				compressed = st_hosvd.compress_tucker(data, rel_target_error, method=method, reshape=reshape)
				rel_errors.append(st_hosvd.rel_error(data, st_hosvd.decompress_tucker(compressed)))
				compression_factors.append(st_hosvd.get_compression_factor_tucker(data, compressed))
			plt.plot(rel_errors, compression_factors, label=name)
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/tensor_trains_st_hosvd_results_%s.png"%dataset_name)
		plt.close()

def tensor_trains_core_tensor_size():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=5)
	
	for dataset_name, loader in (("Indian_Pines", load_indian_pines_cropped), ("Cuprite", load_cuprite_cropped), ("Pavia_Centre", load_pavia), ("Mauna_Kea", load_mauna_kea),):
		
		data = loader()
		rel_errors = []
		size_ratios_core_tensor = []
		for rel_target_error in rel_target_errors:
			print("Testing dataset %s with target error %s"%(dataset_name, rel_target_error))
			compressed = st_hosvd.compress_tucker(data, rel_target_error, method="tensor_trains", reshape=True)
			rel_errors.append(st_hosvd.rel_error(data, st_hosvd.decompress_tucker(compressed)))
			size_ratios_core_tensor.append(st_hosvd.memory_size(compressed["core_tensor"])/(st_hosvd.memory_size(compressed["core_tensor"]) + sum(map(st_hosvd.memory_size, compressed["factor_matrices"]))))
		plt.plot(rel_errors, size_ratios_core_tensor, label=dataset_name.replace("_", " "))
	
	plt.xlabel("Relatieve fout")
	plt.ylabel("(Grootte kerntensor)/(Totale grootte compressie)")
	plt.legend()
	plt.savefig("../tekst/images/tensor_trains_core_tensor_size.png")
	plt.close()

def factor_matrix_quantization_comparison_tensor_trains():
	
	data = load_cuprite_cropped()
	compressed1 = st_hosvd.compress_orthogonality(st_hosvd.compress_tucker(data, 0.025, reshape=True, method="tensor_trains"))
	
	for name, bits_amount_selection, quantization_bits_amounts, annotate_bound, label_x_offset, label_y_offset in (("Gelaagd (norm)", "norm-based", range(5, 16), 16, -0.0004, 0), ("Gelaagd (norm + rang)", "norm-rank-based", range(8, 16), 32, 0, 0)):
		
		rel_errors = []
		compression_factors = []
		for quantization_bits in quantization_bits_amounts:
			print("Testing method %s with %s bits"%(name, quantization_bits))
			compressed2 = st_hosvd.compress_quantize(deepcopy(compressed1), endian="little", encoding_method="default", use_zlib=False, factor_matrix_method="layered", factor_matrix_parameter=quantization_bits, factor_matrix_columns_per_block=1, bits_amount_selection=bits_amount_selection)
			rel_errors.append(rel_error(data, st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.decompress_quantize(compressed2)))))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed2))
			if quantization_bits < annotate_bound:
				plt.annotate(str(quantization_bits), (rel_errors[-1] + label_x_offset, compression_factors[-1] + label_y_offset))
		plt.plot(rel_errors, compression_factors, label=name)
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.legend()
	plt.savefig("../tekst/images/factor_matrix_quantization_comparison_tensor_trains.png")
	plt.close()

def plot_filtered_sweep_points_tensor_trains():
	
	for dataset_name in ("Cuprite", "Indian_Pines"):
		
		errors, factors, _ = filter_points(dataset_name, tensor_trains=True)
		plt.plot(errors, factors)
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.show()
		plt.close()

def plot_filtered_sweep_points_parameters_tensor_trains():
	
	cuprite_errors, _, cuprite_keys = filter_points("Cuprite", tensor_trains=True)
	indian_pines_errors, _, indian_pines_keys = filter_points("Indian_Pines", tensor_trains=True)
	
	for i, parameter_name in enumerate(("RDS", "BPK", "BPF")):
		
		if i == 1:
			continue
		
		# Plot values
		for name, errors, keys in (("Cuprite", cuprite_errors, cuprite_keys), ("Indian Pines", indian_pines_errors, indian_pines_keys)):
			values = [float(key[1:-1].split(", ")[i]) for key in keys]
			plt.plot(errors, values, label=name)
			if (parameter_name == "RDS" and name == "Cuprite") or (parameter_name == "BPF" and name == "Indian Pines"):
				# Print regression parameters
				bound = 0.06 if parameter_name == "RDS" else 0.035
				errors_for_regression = np.array(errors)[np.array(errors) < bound]
				values_for_regression = np.array(values)[np.array(errors) < bound]
				slope, intercept, _, _, _ = stats.linregress(errors_for_regression, values_for_regression)
				print(slope, intercept)
		
		# Plot selection function
		qualities = np.linspace(0.005, 0.05, num=46)
		plt.plot(qualities, [tensor_trains.calculate_parameters(quality)[int(i == 2)] for quality in qualities], label="Selectiefunctie")
			
		plt.xlabel("Relatieve fout")
		plt.ylabel(parameter_name)
		plt.legend()
		plt.savefig("../tekst/images/filtered_sweep_points_tensor_trains_%s.png"%parameter_name)
		plt.close()

def plot_all_sweep_points_indian_pines_tensor_trains():
	
	# Load measurements
	dataset_name = "Indian_Pines"
	measurements_path = "../measurements/parameters_measurements_tensor_trains_%s.json"%dataset_name
	with open(measurements_path, "r") as f:
		measurements = json.load(f)
	measurements = {key: value for key, value in measurements.items() if not type(value) is str}
	
	# Plot all measurements
	points = measurements.values()
	errors = [point[0] for point in points]
	factors = [point[1] for point in points]
	plt.scatter(errors, factors)
	for key in measurements.keys():
		plt.annotate(key, (measurements[key][0], measurements[key][1]))
		
	plt.xlabel("Relatieve fout")
	plt.ylabel("Compressiefactor")
	plt.show()
	plt.close()

def tensor_trains_parameter_selection_results_big_datasets():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=9)
	
	for dataset_name, loader in (("Mauna_Kea", load_mauna_kea), ("Pavia_Centre", load_pavia), ):
		
		data = loader()
		for name, adaptive in (("Adaptief", True), ("Niet-adaptief", False), ):
			rel_errors = []
			compression_factors = []
			for rel_target_error in rel_target_errors:
				print("Testing dataset %s with method %s with target error %s"%(dataset_name, name, rel_target_error))
				compressed = tensor_trains.compress(data, rel_target_error, adaptive=adaptive)
				rel_errors.append(rel_error(data, tensor_trains.decompress(compressed)))
				compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
				print(rel_errors[-1], compression_factors[-1])
			plt.plot(rel_errors, compression_factors, label=name)
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/tensor_trains_parameter_selection_results_%s.png"%dataset_name)
		plt.close()

# Hoofdstuk 6: Resultaten

# Sectie 6.1: Tucker versus tensor trains

def tucker_vs_tensor_trains_results():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=9)
	
	for dataset_name, loader, include_adaptive_tucker in (("Indian_Pines", load_indian_pines_cropped, True), ("Cuprite", load_cuprite_cropped, True), ("Pavia_Centre", load_pavia, False), ("Mauna_Kea", load_mauna_kea, False), ):
		
		data = loader()
		for method, module in (("Tucker", st_hosvd), ("Tensor trains", tensor_trains), ):
			settings = (("niet-adaptief", False), ("adaptief", True), ) if include_adaptive_tucker and method == "Tucker" else (("niet-adaptief", False), )
			for name, adaptive in settings:
				rel_errors = []
				compression_factors = []
				for rel_target_error in rel_target_errors:
					print("Testing dataset %s with method %s with adaptive %s with target error %s"%(dataset_name, method, adaptive, rel_target_error))
					compressed = module.compress(data, rel_target_error, adaptive=adaptive)
					rel_errors.append(rel_error(data, module.decompress(compressed), preserve_decompressed=False))
					compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
				plt.plot(rel_errors, compression_factors, label=method + (" (" + name + ")" if include_adaptive_tucker and method == "Tucker" else ""))
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/tucker_vs_tensor_trains_%s.png"%dataset_name)
		plt.close()

def tucker_vs_tensor_trains_timings():
	
	amount = 3
	
	for dataset_name, loader, include_adaptive_tucker in (("Indian_Pines", load_indian_pines_cropped, True), ("Cuprite", load_cuprite_cropped, True), ("Pavia_Centre", load_pavia, False), ("Mauna_Kea", load_mauna_kea, False), ):
		
		data = loader()
		rel_target_errors = np.linspace(0.01, 0.05, num=3)
		for method, module in (("Tucker", st_hosvd), ("Tensor trains", tensor_trains), ):
			settings = (("niet-adaptief", False), ("adaptief", True), ) if include_adaptive_tucker and method == "Tucker" else (("niet-adaptief", False), )
			for name, adaptive in settings:
				rel_errors = []
				times = []
				for rel_target_error in rel_target_errors:
					print("Testing dataset %s with method %s with adaptive %s with target error %s"%(dataset_name, method, adaptive, rel_target_error))
					total_time = 0
					for i in range(amount):
						print("Running experiment", i)
						start = clock()
						compressed = module.compress(data, rel_target_error, adaptive=adaptive)
						total_time += clock() - start
						print(total_time/(i + 1))
					rel_errors.append(rel_error(data, module.decompress(compressed), preserve_decompressed=False))
					times.append(total_time/amount)
				plt.plot(rel_errors, times, label=method + (" (" + name + ")" if include_adaptive_tucker and method == "Tucker" else ""))
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressietijd (s)")
		plt.legend()
		plt.savefig("../tekst/images/tucker_vs_tensor_trains_times_%s.png"%dataset_name)
		plt.close()

# Sectie 6.2: Vergelijking met algemene lossy compressie

def general_comparison():
	
	rel_target_errors = np.linspace(0.01, 0.05, num=9)
	
	for dataset_name, loader, qualities, crfs_medium, crfs_ultrafast in (("Indian_Pines", load_indian_pines_cropped, range(95, 20, -10), range(10, 31, 5), range(10, 31, 5)), ("Cuprite", load_cuprite_cropped, range(95, 10, -10), range(15, 41, 5), range(15, 36, 5)), ("Pavia_Centre", load_pavia, range(95, 75, -10), range(0, 21, 5), range(0, 26, 5)), ("Mauna_Kea", load_mauna_kea, range(95, 65, -10), range(0, 31, 5), range(0, 31, 5)), ):
		
		data = loader()
		
		# Tensor trains
		rel_errors = []
		compression_factors = []
		for rel_target_error in rel_target_errors:
			print("Testing dataset %s with tensor_trains with target error %s"%(dataset_name, rel_target_error))
			compressed = tensor_trains.compress(data, rel_target_error)
			rel_errors.append(rel_error(data, tensor_trains.decompress(compressed), preserve_decompressed=False))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
			compressed = None
			gc.collect()
		print("rel_errors =", rel_errors)
		print("compression_factors =", compression_factors)
		plt.plot(rel_errors, compression_factors, label="Tensor trains")
		
		# JPEG
		rel_errors = []
		compression_factors = []
		for quality in qualities:
			print("Testing dataset %s with JPEG with quality %s"%(dataset_name, quality))
			compressed = other_compression.compress_jpeg(data, quality)
			rel_errors.append(rel_error(data, other_compression.decompress_jpeg(compressed), preserve_decompressed=False))
			compression_factors.append(other_compression.get_compression_factor_jpeg(data, compressed))
			compressed = None
			gc.collect()
			print(quality, rel_errors[-1])
		print("rel_errors =", rel_errors)
		print("compression_factors =", compression_factors)
		plt.plot(rel_errors, compression_factors, label="JPEG")
		
		# x264 (medium)
		rel_errors = []
		compression_factors = []
		for crf in crfs_medium:
			print("Testing dataset %s with x264 (medium) with crf %s"%(dataset_name, crf))
			compressed = other_compression.compress_video(data, crf, preset="medium")
			rel_errors.append(rel_error(data, other_compression.decompress_video(compressed), preserve_decompressed=False))
			compression_factors.append(other_compression.get_compression_factor_video(data, compressed))
			compressed = None
			gc.collect()
			print(crf, rel_errors[-1])
		print("rel_errors =", rel_errors)
		print("compression_factors =", compression_factors)
		plt.plot(rel_errors, compression_factors, label="x264 (medium)")
			
		
		# x264 (ultrafast)
		rel_errors = []
		compression_factors = []
		for crf in crfs_ultrafast:
			print("Testing dataset %s with x264 (ultrafast) with crf %s"%(dataset_name, crf))
			compressed = other_compression.compress_video(data, crf, preset="ultrafast")
			rel_errors.append(rel_error(data, other_compression.decompress_video(compressed), preserve_decompressed=False))
			compression_factors.append(other_compression.get_compression_factor_video(data, compressed))
			compressed = None
			gc.collect()
			print(crf, rel_errors[-1])
		print("rel_errors =", rel_errors)
		print("compression_factors =", compression_factors)
		plt.plot(rel_errors, compression_factors, label="x264 (ultrafast)")
		
		use_adaptive_tucker = dataset_name in ("Indian_Pines", "Cuprite")
		
		# Tucker (non-adaptive)
		rel_errors = []
		compression_factors = []
		for rel_target_error in rel_target_errors:
			print("Testing dataset %s with Tucker (non-adaptive) with target error %s"%(dataset_name, rel_target_error))
			compressed = st_hosvd.compress(data, rel_target_error)
			rel_errors.append(rel_error(data, st_hosvd.decompress(compressed), preserve_decompressed=False))
			compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
			compressed = None
			gc.collect()
		print("rel_errors =", rel_errors)
		print("compression_factors =", compression_factors)
		plt.plot(rel_errors, compression_factors, label="Tucker (niet-adaptief)" if use_adaptive_tucker else "Tucker")
		
		# Tucker (adaptive)
		if use_adaptive_tucker:
			rel_errors = []
			compression_factors = []
			for rel_target_error in rel_target_errors:
				print("Testing dataset %s with Tucker (adaptive) with target error %s"%(dataset_name, rel_target_error))
				compressed = st_hosvd.compress(data, rel_target_error, adaptive=True)
				rel_errors.append(rel_error(data, st_hosvd.decompress(compressed), preserve_decompressed=False))
				compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
				compressed = None
				gc.collect()
			print("rel_errors =", rel_errors)
			print("compression_factors =", compression_factors)
			plt.plot(rel_errors, compression_factors, label="Tucker (adaptief)")
		
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressiefactor")
		plt.legend()
		plt.savefig("../tekst/images/general_comparison_new_%s.png"%dataset_name)
		plt.close()

def general_comparison_times():
	
	amount = default_experiments_amount
	
	#for dataset_name, loader, crfs_medium, crfs_ultrafast in (("Indian_Pines", load_indian_pines_cropped, range(10, 31, 5), range(10, 31, 5)), ("Cuprite", load_cuprite_cropped, range(15, 41, 5), range(15, 36, 5)), ("Pavia_Centre", load_pavia, range(0, 21, 5), range(0, 26, 5)), ("Mauna_Kea", load_mauna_kea, range(0, 31, 5), range(0, 31, 5)), ):
	for dataset_name, loader, crfs_medium, crfs_ultrafast in (("Mauna_Kea", load_mauna_kea, range(0, 31, 5), range(0, 31, 5)), ):
		
		data = loader()
		
		# Tensor trains
		rel_target_errors = np.linspace(0.01, 0.05, num=9)
		rel_errors = []
		times = []
		for rel_target_error in rel_target_errors:
			print("Testing dataset %s with tensor_trains with target error %s"%(dataset_name, rel_target_error))
			total_time = 0
			for i in range(amount):
				print("Running experiment", i)
				start = time()
				compressed = tensor_trains.compress(data, rel_target_error)
				total_time += time() - start
				if i == 0:
					rel_errors.append(rel_error(data, tensor_trains.decompress(compressed), preserve_decompressed=False))
				compressed = None
				gc.collect()
				print(total_time/(i + 1))
			times.append(total_time/amount)
		plt.plot(rel_errors, times, label="Tensor trains")
		
		# x264
		rel_errors = []
		times = []
		for crf in crfs_medium:
			print("Testing dataset %s with x264 (medium) with crf %s"%(dataset_name, crf))
			total_time = 0
			for i in range(amount):
				print("Running experiment", i)
				start = time()
				compressed = other_compression.compress_video(data, crf, preset="medium")
				total_time += time() - start
				if i == 0:
					rel_errors.append(rel_error(data, other_compression.decompress_video(compressed), preserve_decompressed=False))
				compressed = None
				gc.collect()
				print(total_time/(i + 1))
			times.append(total_time/amount)
		plt.plot(rel_errors, times, label="x264 (medium)")
		
		# x264
		rel_errors = []
		times = []
		for crf in crfs_ultrafast:
			print("Testing dataset %s with x264 (ultrafast) with crf %s"%(dataset_name, crf))
			total_time = 0
			for i in range(amount):
				print("Running experiment", i)
				start = time()
				compressed = other_compression.compress_video(data, crf, preset="ultrafast")
				total_time += time() - start
				if i == 0:
					rel_errors.append(rel_error(data, other_compression.decompress_video(compressed), preserve_decompressed=False))
				compressed = None
				gc.collect()
				print(total_time/(i + 1))
			times.append(total_time/amount)
		plt.plot(rel_errors, times, label="x264 (ultrafast)")
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Compressietijd (s)")
		plt.legend()
		plt.savefig("../tekst/images/general_comparison_times_%s.png"%dataset_name)
		plt.close()

def general_comparison_decompression_times():
	
	amount = default_experiments_amount
	
	#for dataset_name, loader, crfs_medium, crfs_ultrafast in (("Indian_Pines", load_indian_pines_cropped, range(10, 31, 5), range(10, 31, 5)), ("Cuprite", load_cuprite_cropped, range(15, 41, 5), range(15, 36, 5)), ("Pavia_Centre", load_pavia, range(0, 21, 5), range(0, 26, 5)), ("Mauna_Kea", load_mauna_kea, range(0, 31, 5), range(0, 31, 5)), ):
	for dataset_name, loader, crfs_medium, crfs_ultrafast in (("Mauna_Kea", load_mauna_kea, range(0, 31, 5), range(0, 31, 5)), ):
		
		data = loader()
		
		# Tensor trains
		rel_target_errors = np.linspace(0.01, 0.05, num=9)
		rel_errors = []
		times = []
		for rel_target_error in rel_target_errors:
			print("Testing dataset %s with tensor_trains with target error %s"%(dataset_name, rel_target_error))
			total_time = 0
			compressed = tensor_trains.compress(data, rel_target_error)
			for i in range(amount):
				print("Running experiment", i)
				start = time()
				decompressed = tensor_trains.decompress(compressed)
				total_time += time() - start
				if i == 0:
					rel_errors.append(rel_error(data, decompressed, preserve_decompressed=False))
				decompressed = None
				gc.collect()
				print(total_time/(i + 1))
			compressed = None
			gc.collect()
			times.append(total_time/amount)
		plt.plot(rel_errors, times, label="Tensor trains")
		
		# x264
		rel_errors = []
		times = []
		for crf in crfs_medium:
			print("Testing dataset %s with x264 (medium) with crf %s"%(dataset_name, crf))
			total_time = 0
			compressed = other_compression.compress_video(data, crf, preset="medium")
			for i in range(amount):
				print("Running experiment", i)
				start = time()
				decompressed = other_compression.decompress_video(compressed)
				total_time += time() - start
				if i == 0:
					rel_errors.append(rel_error(data, decompressed, preserve_decompressed=False))
				decompressed = None
				gc.collect()
				print(total_time/(i + 1))
			compressed = None
			gc.collect()
			times.append(total_time/amount)
		plt.plot(rel_errors, times, label="x264 (medium)")
		
		# x264
		rel_errors = []
		times = []
		for crf in crfs_ultrafast:
			print("Testing dataset %s with x264 (ultrafast) with crf %s"%(dataset_name, crf))
			total_time = 0
			compressed = other_compression.compress_video(data, crf, preset="ultrafast")
			for i in range(amount):
				print("Running experiment", i)
				start = time()
				decompressed = other_compression.decompress_video(compressed)
				total_time += time() - start
				if i == 0:
					rel_errors.append(rel_error(data, decompressed, preserve_decompressed=False))
				decompressed = None
				gc.collect()
				print(total_time/(i + 1))
			compressed = None
			gc.collect()
			times.append(total_time/amount)
		plt.plot(rel_errors, times, label="x264 (ultrafast)")
			
		plt.xlabel("Relatieve fout")
		plt.ylabel("Decompressietijd (s)")
		plt.legend()
		plt.savefig("../tekst/images/general_comparison_decompression_times_%s.png"%dataset_name)
		plt.close()

# Sectie 6.3: Vergelijking met de literatuur

def literature_comparison():
	
	# Make special plot for Cuprite with right dimensions and such
	
	# Plot dimensions in pixels
	plot_width = 785
	plt_height = 391
	
	# Axes ranges
	minx, maxx = 0, 2
	miny, maxy = 25, 55
	
	# Initialize plot
	my_dpi = 96
	plt.figure(figsize=(plot_width/my_dpi, plt_height/my_dpi), dpi=my_dpi)
	plt.axis("off")
	
	# Test
	#xs = np.linspace(minx, maxx, num=101)
	#plt.plot(xs, 5/0.2*(xs - 0.8) + 25, label="1")
	#plt.plot(xs, 10/0.2*xs + 25, label="2")
	
	# Plot data
	data = load_cuprite_cropped()
	def plot_rate_snr(rel_errors, compression_factors, label):
		plt.plot([data.itemsize*8/factor for factor in compression_factors], [20*math.log10(1/error) for error in rel_errors], label=label)
	
	# Tensor trains
	rel_errors = []
	compression_factors = []
	rel_target_errors = np.logspace(math.log10(0.001), math.log10(0.021), num=11)
	for rel_target_error in rel_target_errors:
		print("Testing tensor_trains with target error %s"%rel_target_error)
		compressed = tensor_trains.compress(data, rel_target_error)
		rel_errors.append(rel_error(data, tensor_trains.decompress(compressed), preserve_decompressed=False))
		compression_factors.append(st_hosvd.get_compression_factor_quantize(data, compressed))
		compressed = None
		gc.collect()
	print("rel_errors =", rel_errors)
	print("compression_factors =", compression_factors)
	plot_rate_snr(rel_errors, compression_factors, "Tensor trains")
	
	# x264 (medium)
	rel_errors = []
	compression_factors = []
	crfs_medium = [0, 2, 4, 6, 10, 13, 16, 20, 24]
	for crf in crfs_medium:
		print("Testing x264 (medium) with crf %s"%crf)
		compressed = other_compression.compress_video(data, crf, preset="medium")
		rel_errors.append(rel_error(data, other_compression.decompress_video(compressed), preserve_decompressed=False))
		compression_factors.append(other_compression.get_compression_factor_video(data, compressed))
		compressed = None
		gc.collect()
		print(crf, rel_errors[-1])
	print("rel_errors =", rel_errors)
	print("compression_factors =", compression_factors)
	plot_rate_snr(rel_errors, compression_factors, "x264 (medium)")
	
	# Finish plot
	plt.xlim(minx, maxx)
	plt.ylim(miny, maxy)
	plt.legend()
	plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
	plt.savefig("../tekst/images/literature_comparison.png", dpi=my_dpi)
	plt.close()

# Sectie 6.4: Voorbeeldcompressies

def save_example_compressions():
	
	qualities = [0.01, 0.025, 0.05]
	#for dataset_name, loader in (("Indian_Pines", load_indian_pines_cropped), ("Cuprite", load_cuprite_cropped), ("Pavia_Centre", load_pavia), ("Mauna_Kea", load_mauna_kea), ):
	for dataset_name, loader in (("Mauna_Kea", load_mauna_kea), ):
		
		print("")
		print(dataset_name)
		print("")
		
		data = loader()
		print("Original size (KB):\t", data.size*data.itemsize/1024)
		print("")
		for quality in qualities:
			compressed = tensor_trains.compress(data, quality)
			decompressed = tensor_trains.decompress(compressed)
			save_image(decompressed, "../tekst/images/example_compression_%s_%s.png"%(dataset_name, str(quality).replace(".", "_")))
			print("Relative error:\t\t", st_hosvd.rel_error(data, decompressed))
			decompressed = None
			gc.collect()
			print("Compression factor:\t", st_hosvd.get_compression_factor_quantize(data, compressed))
			print("Size (KB):\t\t", st_hosvd.get_compress_quantize_size(compressed)/1024)
			compressed = None
			gc.collect()
			print("")

literature_comparison()
