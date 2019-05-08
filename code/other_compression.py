import numpy as np
from PIL import Image
from io import BytesIO
import subprocess as sp
import threading

def compress_jpeg(data, quality):
	
	# Compresses data to a series of RGB-formatted JPEG images with the given quality
	# Each value in the input tensor is always quantized to an 8-bit value so even with maximum quality there will be a small error
	# quality parameter: The image quality, on a scale from 1 (worst) to 95 (best). The default is 75. Values above 95 should be avoided; 100 disables portions of the JPEG compression algorithm, and results in large files with hardly any gain in image quality.
	
	# Shift and scale data to bytes
	min_value = np.amin(data)
	scale = (np.amax(data) - min_value)/255
	data = np.rint((data - min_value)/scale)
	
	# Extend data so that channels are a multiple of 3
	extended_data = np.empty((data.shape[0], data.shape[1], ((data.shape[2] - 1)//3 + 1)*3))
	extended_data[:, :, :data.shape[2]] = data
	for i in range(data.shape[2], extended_data.shape[2]):
		extended_data[:, :, i] = data[:, :, -1]
	
	# Compress to series of JPEG's
	images = []
	for i in range(0, data.shape[2], 3):
		#print(extended_data[-3:, -3:, i:i + 3])
		image = Image.fromarray(np.uint8(extended_data[:, :, i:i + 3]%256))
		out = BytesIO()
		image.save(out, format='jpeg', quality=quality)
		images.append(out.getvalue())
	
	return images, data.shape, min_value, scale
	
def decompress_jpeg(compressed):
	
	# Decompress series of RGB-formatted JPEG images
	
	images, shape, min_value, scale = compressed
	data = np.empty(shape)
	for i in range(0, data.shape[2], 3):
		#print(np.array(Image.open(BytesIO(images[i//3][0]))))
		image_array = np.array(Image.open(BytesIO(images[i//3])))
		data[:, :, i:i + 3] = image_array[:, :, :min(3, data.shape[2] - i)]
		#print(data[-3:, -3:, :])
	
	data = scale*data + min_value
	return data

def get_compress_jpeg_size(compressed):
	
	# Determines size of compression
	images, _, _, _ = compressed
	size = 0
	for image in images:
		size += len(image)
	
	return size

def compress_video(data, crf=23):
	
	# Compresses using video compression
	# crf is the quality parameter. The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible.
	
	# Quantize data and convert to bytes
	min_value = np.amin(data)
	scale = (np.amax(data) - min_value)/255
	data = np.rint((data - min_value)/scale).astype("uint8")
	
	# Set up process
	command = [
		"ffmpeg",
		"-loglevel", "quiet",
		"-y",
		"-pix_fmt", "gray",
		"-s", "%dx%d" % (data.shape[1], data.shape[0]),
		"-r", "30",
		"-c:v", "rawvideo",
		"-f", "rawvideo",
		"-i", "-", # Input from stdin
		"-crf", str(crf),
		"-c:v", "libx264",
		"-preset", "veryslow",
		"-f", "matroska",
		"-" # Output to stdout
	]
	buffer_size = 2**24
	p = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, bufsize=buffer_size)
	
	# Set up output handler
	video = bytearray()
	def read_output():
		video.extend(p.stdout.read())
	
	# Wait for output while sending input
	thread = threading.Thread(target=read_output)
	thread.start()
	p.stdin.write(np.transpose(data, (2, 0, 1)).tobytes())
	p.stdin.close()
	thread.join()
	
	return video, data.shape, min_value, scale

def decompress_video(compressed):
	
	# Decompresses using video compression
	
	video, shape, min_value, scale = compressed
	
	# Set up process
	command = [
		"ffmpeg",
		"-loglevel", "quiet",
		"-i", "-", # Input from stdin
		"-pix_fmt", "gray",
		"-s", "%dx%d" % (shape[1], shape[0]),
		"-r", "30",
		"-c:v", "rawvideo",
		"-f", "rawvideo",
		"-" # Output to stdout
	]
	buffer_size = 2**24
	p = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, bufsize=buffer_size)
	
	# Set up output handler
	raw_data = bytearray()
	def read_output():
		raw_data.extend(p.stdout.read())
	
	# Wait for output while sending input
	thread = threading.Thread(target=read_output)
	thread.start()
	p.stdin.write(video)
	p.stdin.close()
	thread.join()
	
	# Reshape and scale data
	data = np.fromstring(bytes(raw_data), dtype="uint8")
	data = np.reshape(data, (shape[2], shape[0], shape[1])) # Reshape into video-formatted shape
	data = np.transpose(data, (1, 2, 0)) # Re-order dimensions
	data = data*scale + min_value
	
	return data

def get_compress_video_size(compressed):
	video, _, _, _ = compressed
	return len(video)
