import numpy as np
import matplotlib.pyplot as plt
from time import time, clock
from math import floor

import st_hosvd
import tensor_trains
import other_compression
from tools import *

def main():
	test_compression_ratio_tucker()

def test_compression_ratio_tucker():
	
	print("="*20 + " Phase 1 " + "="*20)
	data = load_mauna_kea()
	compressed1 = st_hosvd.compress_tucker(data, 0.025, reshape=False, method="tucker")
	decompressed = st_hosvd.decompress_tucker(compressed1)
	st_hosvd.print_compression_rate_tucker(data, compressed1)
	print_difference(data, decompressed)
	
	print("="*20 + " Phase 2 " + "="*20)
	compressed2 = st_hosvd.compress_orthogonality(compressed1, method="householder")
	decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(compressed2, renormalize=True))
	print_difference(data, decompressed)
	
	print("="*20 + " Phase 3 " + "="*20)
	start_time = clock()
	compressed3 = st_hosvd.compress_quantize(compressed2, endian="little", encoding_method="adaptive", allow_approximate_huffman=False, use_zlib=True, core_tensor_method="layered-constant-step", core_tensor_parameter=12, core_tensor_unquantized_rel_norm=0.995, factor_matrix_method="layered", quantize_factor_matrices=True, factor_matrix_parameter=10, factor_matrix_columns_per_block=1, bits_amount_selection="norm-based")
	print("Finished quantizing and encoding in:", clock() - start_time)
	start_time = clock()
	decompressed1 = st_hosvd.decompress_quantize(compressed3)
	print("Finished decoding and dequantizing in:", clock() - start_time)
	decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_orthogonality(decompressed1))
	print("Compression ratio:", st_hosvd.get_compress_quantize_size(compressed3, print_intermediate_values=False)/st_hosvd.memory_size(data))
	print_difference(data, decompressed)

def plot_mauna_kea_range():
	
	data = load_mauna_kea()
	for i in range(3):
		axes = list(range(3))
		axes.remove(i)
		plt.plot(range(data.shape[i]), np.amax(data, axis=tuple(axes)))
		plt.plot(range(data.shape[i]), np.amin(data, axis=tuple(axes)))
		plt.show()

def compress_mauna_kea():
	
	data = load_mauna_kea()
	compressed = st_hosvd.compress_tucker(data, 0.025, print_progress=True, use_pure_gramian=True)
	print("Relative error:", custom_norm(st_hosvd.decompress_tucker(compressed).__isub__(data))/custom_norm(data))
	st_hosvd.print_compression_rate_tucker(data, compressed)

def compress_zip_cuprite():
	
	data = load_cuprite()
	data.tofile("../data/cuprite_raw")

def plot_intensities_pavia():
	
	data = load_pavia()
	samples = np.mean(data, axis=(0, 1))
	channels = range(samples.size)
	plt.plot(channels, samples)
	plt.scatter(channels, samples)
	plt.show()

def plot_intensities_mauna_kea():
	
	data = load_mauna_kea()
	samples = np.mean(data, axis=(0, 1))
	channels = range(samples.size)
	plt.plot(channels, samples)
	plt.scatter(channels, samples)
	plt.show()

def make_video_mauna_kea_raw():
	
	data = load_mauna_kea_raw()
	quality = 0
	compressed = other_compression.compress_video(data, quality)
	
	with open("../data/mauna_kea_raw.mkv", "wb") as f:
		f.write(compressed[0])

def make_video_mauna_kea():
	
	data = load_mauna_kea()
	quality = 0
	compressed = other_compression.compress_video(data, quality)
	
	with open("../data/mauna_kea.mkv", "wb") as f:
		f.write(compressed[0])
	
	print(rel_error(other_compression.decompress_video(compressed), data))

def test_sorting():
	
	amount = 10
	
	data = load_cuprite()
	for sort in (False, True):
		times = np.zeros(amount)
		for i in range(amount):
			output = st_hosvd.compress_tucker(data, 0.025, extra_output=True, randomized_svd=True, sort_indices=sort)
			times[i] = output["total_cpu_time"]
		print("Sort:", sort)
		print("Mean:", np.mean(times))
		print("Stdev:", np.std(times))

def test_compression_ratio_tensor_trains():
	
	data = load_pavia()
	compressed = tensor_trains.compress_tensor_trains(data, 0.025, print_progress=True)#(501, 596, 12))
	decompressed = tensor_trains.decompress_tensor_trains(compressed)
	tensor_trains.print_compression_rate_tensor_trains(data, compressed)
	print_difference(data, decompressed)
	#plot_comparison(data, decompressed)

def test_compress_jpeg():
	
	# Test series of JPEG compressions and plot results
	data = load_cuprite()
	original_size = data.dtype.itemsize*data.size
	rel_errors = []
	compression_ratios = []
	
	for quality in range(5, 100, 5):
		print(quality)
		compressed = other_compression.compress_jpeg(data, quality)
		decompressed = other_compression.decompress_jpeg(compressed)
		rel_errors.append(np.linalg.norm(data - decompressed)/np.linalg.norm(data))
		compression_ratios.append(other_compression.get_compress_jpeg_size(compressed)/original_size)
	
	print("rel_errors =", rel_errors)
	print("compression_ratios =", compression_ratios)
	plt.plot(rel_errors, compression_ratios, "bo")
	plt.plot(rel_errors, compression_ratios, "b")
	plt.show()

def test_compress_video():
	
	# Test series of video compressions and plot results
	data = load_cuprite()
	original_size = data.dtype.itemsize*data.size
	rel_errors = []
	compression_factors = []
	
	for quality in range(0, 55, 5):
		print(quality)
		start = time()
		compressed = other_compression.compress_video(data, crf=quality, preset="superfast")
		decompressed = other_compression.decompress_video(compressed)
		print("Time elapsed:", time() - start)
		rel_errors.append(np.linalg.norm(data - decompressed)/np.linalg.norm(data))
		compression_factors.append(original_size/other_compression.get_compress_video_size(compressed))
		print(rel_errors[-1])
		print(compression_factors[-1])
	
	print("rel_errors =", rel_errors)
	print("compression_factors =", compression_factors)
	plt.plot(rel_errors, compression_factors, "bo")
	plt.plot(rel_errors, compression_factors, "b")
	plt.show()

def test_compress_variable_tucker():
	
	# Test series of Tucker compressions
	data = load_cuprite()
	original_size = data.dtype.itemsize*data.size
	rel_errors = []
	compression_ratios = []
	
	for rel_error in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2):
		print(rel_error)
		compressed = st_hosvd.compress_quantize2(st_hosvd.compress_tucker(data, rel_error, print_progress=False))
		decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_quantize2(compressed))
		rel_errors.append(np.linalg.norm(data - decompressed)/np.linalg.norm(data))
		compression_ratios.append(st_hosvd.get_compress_quantize2_size(compressed)/original_size)
	
	print("rel_errors =", rel_errors)
	print("compression_ratios =", compression_ratios)

def compare_times():
	
	data = load_cuprite()
	original_size = data.dtype.itemsize*data.size
	
	st_hosvd.print_compression_rate_tucker(data, st_hosvd.compress_tucker(data, 0.025))
	start = time()
	compressed = st_hosvd.compress_quantize2(st_hosvd.compress_tucker(data, 0.025))
	print("Time for compression:", time() - start)
	start = time()
	decompressed = st_hosvd.decompress_tucker(st_hosvd.decompress_quantize2(compressed))
	print("Time for decompression:", time() - start)
	st_hosvd.print_compression_rate_quantize2(data, compressed)
	print_difference(data, decompressed)
	
	start = time()
	compressed = other_compression.compress_jpeg(data, 50)
	print("Time for compression:", time() - start)
	start = time()
	decompressed = other_compression.decompress_jpeg(compressed)
	print("Time for decompression:", time() - start)
	print("Compressed size:", other_compression.get_compress_jpeg_size(compressed))
	print("Compression ratio:", other_compression.get_compress_jpeg_size(compressed)/original_size)
	print_difference(data, decompressed)
	
	start = time()
	compressed = other_compression.compress_video(data, 28)
	print("Time for compression:", time() - start)
	start = time()
	decompressed = other_compression.decompress_video(compressed)
	print("Time for decompression:", time() - start)
	print("Compressed size:", other_compression.get_compress_video_size(compressed))
	print("Compression ratio:", other_compression.get_compress_video_size(compressed)/original_size)
	print_difference(data, decompressed)

def plot_compression_comparison(case):
	
	if case == "Cuprite":
		
		# Data from after removal of artifacts
		
		# JPEG
		rel_errors = [0.069768410441197526, 0.051173216770713778, 0.043522255430459282, 0.038944788220625359, 0.035658346265431483, 0.033276775378738473, 0.031179055306833577, 0.029885557750992806, 0.028402234192627676, 0.027363422064088927, 0.026670468078529231, 0.025822467909564634, 0.024491226346233783, 0.022801805007577097, 0.021616049683549157, 0.020205891672578732, 0.018518622063243624, 0.016236008280671756, 0.012827399346320779]
		compression_ratios = [0.0008665464477434425, 0.0011646822061707098, 0.0015114933185914196, 0.0018500736314396536, 0.002181488356254286, 0.002484479302835162, 0.0027840234843776786, 0.0030242485675520746, 0.0033047005587819303, 0.003544009015916981, 0.003788846527837305, 0.004094453416311289, 0.004482541999051732, 0.00493647279286495, 0.005430425404889208, 0.006210557874512472, 0.007317048976459369, 0.00934832367095084, 0.014015886008218327]
		plt.plot(rel_errors, compression_ratios, "bo", label="JPEG")
		plt.plot(rel_errors, compression_ratios, "b")

		plt.plot([0.002097713132559369], [0.0652392427446747], "go", label="PNG")
		
		# Tucker
		rel_errors = [0.014693139740447031, 0.018374881315268106, 0.022566815728485956, 0.027152063012008478, 0.031709331021400763, 0.041182250879728537, 0.050791957291718315, 0.060864564540400894, 0.07001124642517631, 0.080075736334746891, 0.098840662979154351, 0.10531174310127328, 0.10531143056310975, 0.10531147153484054, 0.1053116029285092, 0.10531231654627704]
		compression_ratios = [0.003823820624571404, 0.001676255802786945, 0.001002223843594634, 0.0006744170258389765, 0.0005096733764250814, 0.0002706599975355735, 0.00015068662405708896, 9.987456872535574e-05, 4.7591307672938456e-05, 2.1689297424138522e-05, 1.8667361349219956e-06, 1.485854995285445e-07, 7.743188003600205e-08, 4.1855070289730845e-08, 4.1855070289730845e-08, 4.1855070289730845e-08]
		plt.plot(rel_errors, compression_ratios, "ro", label="Tucker")
		plt.plot(rel_errors, compression_ratios, "r")
		
		# Video
		rel_errors = [0.002097713132559369, 0.0045815692796644727, 0.0066331424987280153, 0.0097938837928269254, 0.014852849349730041, 0.021659202931874257, 0.030225506603691398, 0.040570163008044717, 0.05211904497792906, 0.064799973736880748, 0.082406679060748014]
		compression_ratios = [0.03375462624091912, 0.017664591296958042, 0.008621292729004157, 0.0033488911419938283, 0.0011946441582376137, 0.0005533700698075604, 0.0002901246979738128, 	0.00015207202688367907, 7.965229151487228e-05, 4.2355238379693125e-05, 2.49895697164838e-05]
		compression_factors = [7.185725446452739, 15.733518669321322, 33.197323979749214, 81.01949326464369, 216.61642014865427, 494.3813603161794, 1130.3172575624249, 2737.0796196586093, 6836.04234620887, 13157.819143077431, 16449.991737813274] # Veryfast
		plt.plot(rel_errors, compression_ratios, "co", label="libx264")
		plt.plot(rel_errors, compression_ratios, "c")
	
	plt.xlabel("Relative error")
	plt.ylabel("Compression ratio")
	plt.title("Compression comparison (%s)"%case)
	plt.legend()
	plt.show()

def test_time():
	# Tests performance on random data, ignoring compression ratio
	data = np.random.rand(101, 101, 10001)
	compressed = st_hosvd.compress_tucker(data, 0, rank=(22, 21, 19), print_progress=True)
	decompressed = st_hosvd.decompress_tucker(compressed)
	print_difference(data, decompressed)
	st_hosvd.print_compression_rate_tucker(data, compressed)
	plot_comparison(data, decompressed)

main()
