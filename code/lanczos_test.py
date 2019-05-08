import numpy as np
import scipy.linalg
from math import sqrt
from st_hosvd import custom_lanczos

def compare_bases(U1, U2):
	sq_error = 0
	common_base = min(U1.shape[1], U2.shape[1])
	for i in range(common_base):
		print(min(np.linalg.norm(U1[:, i] - U2[:, i]), np.linalg.norm(U1[:, i] + U2[:, i]))/np.linalg.norm(U1[:, i]))
		sq_error += min(np.linalg.norm(U1[:, i] - U2[:, i]), np.linalg.norm(U1[:, i] + U2[:, i]))**2
	print(sqrt(sq_error)/np.linalg.norm(U1[:, :common_base]))

A = np.random.rand(100, 1000)
lambdas1, X1 = np.linalg.eigh(A @ A.T)
lambdas1 = lambdas1[::-1]
X1 = X1[:, ::-1]

relative_target_error = 0.05
bound = (1 - relative_target_error**2)*np.linalg.norm(A)**2
lambdas2, X2 = lanczos_gramian(A, bound)
print(lambdas1)
print(lambdas2, lambdas2.size)
print(np.linalg.norm(lambdas1[:lambdas2.size] - lambdas2)/np.linalg.norm(lambdas1[:lambdas2.size]))

print("Approximating A:")
print(np.linalg.norm(A - X1[:, :X2.shape[1]] @ (X1[:, :X2.shape[1]].T @ A))/np.linalg.norm(A))
print(np.linalg.norm(A - X2 @ (X2.T @ A))/np.linalg.norm(A))
compare_bases(X1, X2)
