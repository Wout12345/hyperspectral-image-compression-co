import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
from time import time

print("Numpy version:", np.version.version)

def mode_k_product(U, X, mode):
	transposition_order = list(range(X.ndim))
	transposition_order[mode] = 0
	transposition_order[0] = mode
	Y = np.transpose(X, transposition_order)
	transposed_ranks = list(Y.shape)
	Y = np.reshape(Y, (Y.shape[0], -1))
	Y = U @ Y
	transposed_ranks[0] = Y.shape[0]
	Y = np.reshape(Y, transposed_ranks)
	Y = np.transpose(Y, transposition_order)
	return Y

def einsum_product(U, X, mode):
	axes1 = list(range(X.ndim))
	axes1[mode] = X.ndim + 1
	axes2 = list(range(X.ndim))
	axes2[mode] = X.ndim
	return np.einsum(U, [X.ndim, X.ndim + 1], X, axes1, axes2, optimize=True)

def test_correctness():
	A = np.random.rand(3, 4, 5)
	for i in range(3):
		B = np.random.rand(6, A.shape[i])
		X = mode_k_product(B, A, i)
		Y = einsum_product(B, A, i)
		print(np.allclose(X, Y))

def test_time(method, amount):
	U = np.random.rand(256, 512)
	X = np.random.rand(512, 512, 256)
	start = time()
	for i in range(amount):
		method(U, X, 1)
	return (time() - start)/amount

def test_times():
	print("Explicit:", test_time(mode_k_product, 10))
	print("Einsum:", test_time(einsum_product, 10))

test_correctness()
test_times()
