import numpy as np
import matplotlib.pyplot as plt

from tools import *
import st_hosvd

default_experiments_amount = 10

# Hoofdstuk 3: Methodologie

def save_cuprite_image():
	data = load_cuprite()
	image = np.sum(data, axis=2)
	imsave("../tekst/images/cuprite_sum.png", np.rint(image/np.amax(image)*255).astype(int))

def save_pavia_image():
	data = load_pavia()
	image = np.sum(data, axis=2)
	imsave("../tekst/images/pavia_sum.png", np.rint(image/np.amax(image)*255).astype(int))

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
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, randomized_svd=True, sample_ratio=sample_ratio)
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
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, use_pure_gramian=True, randomized_svd=randomized_svd, sample_ratio=0.2, samples_per_dimension=100, store_rel_estimated_S_errors=True)
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

randomized_svd_pavia_test()
