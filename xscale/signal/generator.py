# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Numpy
import numpy as np
# Scipy
import scipy.signal as sig
# Xarray
import xarray as xr
# Pandas
import pandas as pd
# Dask
import dask.array as da
# Numba
import numba
# Xscale
from .. import _utils


def rednoise(alpha, n, c=0.):
	"""
	Generate a red noise

	Parameters
	----------
	alpha: float
		Autocorrelation coefficient of the red noise
	n: int
		Size of the vector
	c: float
		Mean of the vector

	Returns
	-------
	res: 1darray
		A vector of size n describing a rednoise process

	Examples
	--------
	>>> rednoise(0.8, 6)
	"""
	if _utils.is_iterable(alpha):
		raise TypeError("The first argument must be a scalar, for higher order autoregressive process, please use the "
		                "function xscale.generator.ar ")
	return ar(alpha, n, c=0.)


@numba.jit
def ar(coeffs, n, c=0.):
	"""
	Generate a timeseries using an autoregressive process

	Parameters
	----------
	coeffs: iterable
		Coefficients of the autoregressive process
	n: int
		Size of the vector
	c: float
		Mean of the vector

	Returns
	-------
	res: 1darray
		A vector of size n describing an autoregressive process

	"""
	if not _utils.is_iterable(coeffs):
		coeffs = [coeffs]
	res = np.zeros(n)
	order = len(coeffs)
	for i in range(order):
		res[i] = c + np.random.normal(0, 1)
	for i in range(order, n):
		res[i] = c + np.random.normal(0, 1)
		for j in range(order):
			res[i] += coeffs[j] * res[i - (j + 1)]
	return res


def window1d(n, dim=None, coords=None, window='boxcar'):
	"""
	Generate a one dimensional window from `scipy.signal.get_window`

	Parameters
	----------

	"""
	if _utils.is_iterable(window):
		name = window[0]
	else:
		name = window
	return xr.DataArray(sig.get_window(window, n), dims=dim, coords=coords, name=name)


#@numba.guvectorize
def trend(x, slope, offset):
	"""
	Generate a trend
	"""
	return slope * x + offset


def example_xt():
	"""
	Generate a time-space DataArray to be used as an example with `xscale`
	functions
	"""
	nx = 128
	nt = 100
	time = pd.date_range('1/1/2011', periods=nt, freq='D')
	x1d = np.linspace(0, 2 * np.pi, nx)
	rand = xr.DataArray(np.random.rand(nt, nx), coords=[time, x1d],
	                    dims=['time', 'x'])
	slopes = 0.02 * xr.DataArray(np.cos(2 * np.pi * x1d / nx), coords=[x1d],
	                             dims=['x'])
	output = rand + slopes * rand.time.astype('f8') * 1. / (3600. * 24.)
	output.name = 'example_xt'
	return output.chunk({'time': 50, 'x': 70})


def example_xyt(boundaries=False):
	"""
	Generate a time-space DataArray with two spatial dimensions ['x', 'y'] on a
	regular grid.
	"""
	nt = 100
	nx = 128
	ny = 128
	time = pd.date_range('1/1/2011', periods=nt, freq='D')
	t1d =  np.asarray(time).astype('f8')
	x1d = np.linspace(0, 2 * np.pi, nx)
	y1d = np.linspace(0, 2 * np.pi, ny)
	t, y, x = np.meshgrid(t1d, y1d, x1d, indexing='ij')
	omega_daily = 2. * np.pi * (3600. * 24.)
	# Create four times modulation with
	m1 = np.cos(5 * omega_daily * t)
	m2 = np.cos(9 * omega_daily * t)
	m3 = np.cos(11 * omega_daily * t)
	m4 = np.cos(19 * omega_daily * t)
	# Create a spatio-temporal gaussian noise
	noise = 0.8 * np.random.normal(0, 0.2, (nt, ny, nx))
	# Create four spatial patterns
	z1 = np.sin(x) * np.sin(y) * m1
	z2 = np.sin(2.5 * x) * np.sin(y) * m2
	z3 = np.sin(x) * np.sin(2.5 * y) * m3
	z4 = np.sin(2.5 * x) * np.sin(2.5 * y) * m4
	z = z1 + z2 + z3 + z4 + noise
	if boundaries:
		z[:, 0:ny//2, 0:nx//4] = np.nan
		z[:, 0:nx//4, 0:nx//2] = np.nan
	output = xr.DataArray(z, coords=[time, y1d, x1d], dims=['time', 'y', 'x'],
	                      name='example_xyt')
	return output.chunk({'time': 50, 'x': 70, 'y':70})