import numba
import dask.array as da
import numpy as np

def ssa(array, dim, modes=None):
	"""
	Perform a singular spectrum analysis by computing the eigenvalues
	the lagged covariance matrix.

	Parameters
	----------
	x : dataArray
		A vector of length n.
	dim : str
	    Dimension along which compute the

	Returns
	-------
	res : dataArray
		A dataArray that returns the result from the singular spectrum analysis
	"""
	n = data.size(x)
	nprim = n - modes + 1
	elem_array = da.zeros((modes, nprim,))
	covmat = da.zeros((modes, modes,))
	_compute_ssa_covmat(array, covmat, n, modes)
	eigval, teof = linalg.eigh(covmat)
	idx = eigval.argsort()[::-1]
	eigval = eigval[idx]
	teof = teof[:, idx]
	tpc = np.dot(A.T, self.teof)
	# Compute an estimate of the error on the eigenvalues
	kappa = 1.5
	tau = - 1. / np.log(ar1(x))
	self.eigval_err = (2 * kappa * tau / n) * self.eigval
	# Build partial reconstructions


@numba.jit
def _compute_ssa_covmat(array, covmat, n, modes):
	"""Build the covariance matrix for singular spectrum analysis"""
	nprim = n - modes + 1
	for i in range(modes):
		for j in range(modes):
			ntilde = (n - abs(i - j))
			covmat[i, j,] = 1. / ntilde * np.sum(array[:ntilde,] * array[abs(i - j):,])
		elem_array[i,] = array[i:nprim + i,]
	covmat /= modes


@numba.jit
def _compute_ssa_rc(n, modes, tpc, teof):
	"""Compute the reconstructed component from the SSA decomposition"""
	nprim = n - modes + 1
	rc = np.zeros((n, modes))
	for i in range(n):
		if i < modes - 1:
			for j in range(i + 1):
				rc[i,] += 1. / (i + 1) * tpc[i - j,] * teof[j,]
		elif i >= nprim - 1:
			for j in range(i - nprim + 1, modes):
				rc[i,] += 1. / (n - i) * tpc[i - j,] * teof[j,]
		else:
			for j in range(m):
				rc[i,] += 1. / modes * tpc[i - j,] * teof[j,]
	return rc