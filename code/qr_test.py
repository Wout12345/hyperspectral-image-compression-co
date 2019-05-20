import numpy as np
import scipy.linalg
import scipy.linalg.lapack as lapack
from math import sqrt

def reference(A):
	(h, tau), _ = scipy.linalg.qr(A, mode="raw")
	return h, tau

def custom_qr(A):
	
	# For cols <= rows
	# Matches sgeqrf except for signs of alpha
	
	cols = A.shape[1]
	h = np.copy(A)
	tau = np.empty(cols, dtype="float32")
	
	for i in range(cols):
		
		# Calculate v and tau
		# v is not normalized but scaled so that first element is 1
		v = h[i:, i].copy()
		v[0] = v[0] + np.sign(v[0])*np.linalg.norm(v) # Sign of alpha is chosen so that magnitude of v[0] increases, norm of vector increases so no problem with catastrophic cancellation
		inv_tau = 2*(v[0]/np.linalg.norm(v))**2 # Tau for inverse transformation, so H_i^-1 = I - inv_tau*v*v.T with H_i = I - tau*v*v.T
		v = v/v[0] # So v[0] == 1
		tau[i] = inv_tau # Actually, tau[i] = inv_tau/(inv_tau*(v.T@v) - 1), but because of choice of inv_tau in function of v we get tau[i] = inv_tau
		
		# Apply reflection to relevant part of A
		h[i:, i:] = h[i:, i:] - inv_tau*np.outer(v, v.T @ h[i:, i:])
		
		# Store v in unused part of A
		h[i + 1:, i] = v[1:]
	
	return h, tau

def reconstruct(h, tau):
	rows, cols = h.shape
	R = np.triu(h)
	for i in range(cols - 1, -1, -1):
		v = h[i:, i]
		v[0] = 1
		R[i:, i:] = R[i:, i:] - tau[i]*np.outer(v, v.T @ R[i:, i:])
	return R

def test():
	
	# Create random orthogonal matrix
	np.random.seed(598378)
	A = np.random.rand(100, 90)
	U, _, _ = scipy.linalg.svd(A, full_matrices=False)
	
	# Get reference
	h1, tau1 = reference(U)
	
	# Get our result
	h2, tau2 = custom_qr(U)
	
	# Compare
	print(np.linalg.norm(h1 - h2)/np.linalg.norm(h1))
	print(np.linalg.norm(tau1 - tau2)/np.linalg.norm(tau1))

test()
