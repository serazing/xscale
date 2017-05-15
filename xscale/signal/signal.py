# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Numpy
import numpy as np
# Xarray
import xarray as xr
# Dask
import dask.array as da

# Xarray-based function are public
def detrend(array, dim=None, typ='linear', chunks=None):
	"""
	Remove the mean, linear or quadratic trend and remove it.

	Parameters
	----------
	array : xarray.DataArray
		DataArray that needs to be detrended along t
	dim:
		Dimension over which the array will be detrended
	"""
	if dim is None:
		axis = 0
	else:
		axis = array.get_axis_num(dim)
	res = _detrend(array.data, axis, typ=typ)
	return xr.DataArray(res, dims=array.dims, coords=array.coords,
	                    name=array.name)


# Dask-based functions are for private use only
def _detrend(data, axis, typ='linear'):
	"""Detrend using dask.linalg.lstq`"""
	if typ is 'mean':
		res = data - data.mean(axis=axis)
	elif typ is 'linear':
		a, b, _, _, _ = _linreg(x, t)
		trend = a * T + b
		res =  x - trend
	elif typ is 'quadratic':
		a, b, c, _ = _quadreg(x, t)
		trend = a * T ** 2 + b * T + c
		res =  x - trend
		da.linalg.lstsq(data)
	return res


def linreg(data1, data2, dim):
	# Restructure data so that axis is along first dimension and
	# all other dimensions are collapsed into second dimension
	if dim not in data1.dims:
		raise ValueError("Cannot find dim in data dimensions.")
	other_dims = [di for di in data1.dims if not di == dim]
	data_stack = data1.stack(other=other_dims)
	data_stack = data_stack.transpose([dim, 'other'])


	n1 = shape[0]
	n2 = np.prod(shape[1:])



def _quadreg(data1, data2, axis):
	"""
	Fit a quadratic law x = a * t ** 2 + b * t + c.

	Parameters
	----------
	x : ndarray
		Input timeseries
	t : 1darray
		Type of interpolation

	Returns
	-------
	a : ndarray
		Quadratic coefficient
	b : ndarray
		Slope coefficient
	c : ndarray
		Barycentre coefficient
	stderr : ndarray
		Standard deviation of the error of the fit.
	"""
	dim = np.shape(x)
	n_t = dim[0]
	n_s = np.prod(dim[1:])
	X = np.reshape(x, (n_t, n_s), order='F')
	A = np.vstack([t ** 2, t, np.ones(n_t)]).T
	# Solve the least square system
	coeff, err, _, _ = linalg.lstsq(A, X)
	a = np.reshape(coeff[0,], dim[1:], order='F')
	b = np.reshape(coeff[1,], dim[1:], order='F')
	c = np.reshape(coeff[2,], dim[1:], order='F')
	stderr = 1. / np.sqrt(n_t - 2) * np.reshape(np.sqrt(err), dim[1:],
	                                            order='F')
	return a, b, c, stderr
