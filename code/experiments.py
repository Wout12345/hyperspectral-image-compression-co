import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from time import clock
import gc
from copy import deepcopy

from tools import *
import st_hosvd

default_experiments_amount = 10
# Hoofdstuk 3: Methodologie

def save_image(data, path):
	image = np.sum(data, axis=2)
	imsave(path, np.rint((image - np.amin(image))/(np.amax(image) - np.amin(image))*255).astype(int))

def save_cuprite_image():
	save_image(load_cuprite(), "../tekst/images/cuprite_sum.png")

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
	for name, quantize in (("Geen quantisatie", False), ("Quantisatie", True)):
		compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=None)
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
	for name, quantize in (("Geen quantisatie", False), ("Quantisatie", True)):
		compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=None)
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
	for name, quantize in (("Geen quantisatie", False), ("Quantisatie", True)):
		compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=None)
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
	for name, quantize in (("Geen quantisatie", False), ("Quantisatie", True)):
		decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=quantize, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=None))
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
			compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=True, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=None)
			total_time += clock() - start
		print(name, "{:.2f}".format(total_time/amount))

# Sectie 4.2.1.1: Hernormalisatie

def orthogonality_compression_norms():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	cols = range(compressed1["factor_matrices"][factor_matrix_index].shape[1])
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=True, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=None)
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
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=True, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=0, method="systems"), renormalize=True)
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
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=True, orthogonality_reconstruction_steps=10, orthogonality_reconstruction_margin=0, method="systems"), renormalize=True)
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
			compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(compressed1, quantize=True, orthogonality_reconstruction_steps=10, orthogonality_reconstruction_margin=0, method="systems"), renormalize=True)
			total_time += clock() - start
		print(name, "{:.2f}".format(total_time/amount))

# Sectie 4.2.1.3: Marge

def orthogonality_compression_margin():
	
	data = load_cuprite()
	compressed1 = st_hosvd.compress_tucker(data, 0.025)
	
	factor_matrix_index = 1
	reference = compressed1["factor_matrices"][factor_matrix_index].copy()
	cols = range(reference.shape[1])
	compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(deepcopy(compressed1), quantize=True, orthogonality_reconstruction_steps=500, orthogonality_reconstruction_margin=3, method="systems"), renormalize=True)
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
				compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(copy, quantize=True, orthogonality_reconstruction_steps=steps, orthogonality_reconstruction_margin=margin, method="systems"), renormalize=renormalize)
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
	settings = (("ST-HOSVD", False, False), ("Geen quantisatie", True, False), ("Quantisatie", True, True))
	
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
					compressed2 = st_hosvd.decompress_orthogonality(st_hosvd.compress_orthogonality(copy, quantize=quantize, method="householder"))
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
			lines.append("{} & {:.11f} & {:.11f} & {:.3f} & {:.3f} \\\\ \\hline".format(setting_name, measurements[0]["errors"][i], measurements[1]["errors"][i], measurements[0]["times"][i], measurements[1]["times"][i]))
		else:
			lines.append("{} & {:.11f} & {:.11f} & N/A & N/A \\\\ \\hline".format(setting_name, measurements[0]["errors"][i], measurements[1]["errors"][i]))
	
	with open("../tekst/data/orthogonality-compression-householder-summary.tex", "w") as f:
		f.writelines([line + "\n" for line in lines])

orthogonality_compression_basic_error()
