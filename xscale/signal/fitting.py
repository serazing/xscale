# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Numpy
import numpy as np
# Xarray
import xarray as xr
# Dask
import dask.array as da
# Pandas
import pandas as pd
# Xscale
from .. import _utils


def polyfit(array, dim, deg=1):
	"""
	Least squares polynomial fit.
	Fit a polynomial ``p(x) = p[deg] * x ** deg + ... + p[0]`` of degree `deg`
	Returns a vector of coefficients `p` that minimises the squared error.

	Parameters
	----------
	x : xarray.DataArray
		The array to fit
	dim : str
		Dimension along which the fit is performed
	deg : int
		Degree of the fitting polynomial


	Returns
	-------
	output : xarray.DataArray
		Polynomial coefficients
	"""
	# Re-order the array to place the fitting dimension as the first dimension
	# + stack the other dimensions
	stacked_dims = [di for di in array.dims if di is not dim]
	new_dims = [dim, ] + stacked_dims
	stacked_array = array.transpose(*new_dims).stack(temp_dim=stacked_dims)
	dim_chunk = array.chunks[array.get_axis_num(dim)][0]
	# Build coefficient matrix for the fit
	x = da.vstack([array[dim].data ** d for d in range(deg + 1)]).T
	x = x.rechunk((dim_chunk, deg + 1))
	# Solve the least-square system
	p, err, _, _ = da.linalg.lstsq(x, stacked_array.data)
	# TO DO: Compute and store the errors associated to the fit
	# Store the result in a DataArray object
	new_dims = ('degree',) + stacked_array.dims[1:]
	ds = xr.DataArray(p, name='polynomial_coefficients',
	                  coords=stacked_array.coords, dims=new_dims)
	return ds.unstack('temp_dim').assign_coords(degree=range(deg + 1))


def polyval(coefficients, coord):
	"""
	Build an array from polynomial coefficients

	Parameters
	----------
	coefficients : xarray.DataArray
		The DataArray where the coefficients are stored
	coord : xarray.Coordinate
		The locations where the polynomials is evaluated

	Returns
	-------
	output : xarray.DataArray
		The polynomials evaluated at specified locations
	"""
	#TODO
	raise NotImplementedError


def sinfit(array, dim, periods, unit='s'):
	"""
	Least squares sinusoidal fit.
	Fit sinusoidal functions ``y = A[p] * sin(2 * pi * ax * f[1] + phi[1])``

	Parameters
	----------
	array : xarray.DataArray
		Data to be fitted
	dim : str
		The dimension along which the data will be fitted
	periods :
	unit : {'D', 'h', 'm', 's', 'ms', 'us', 'ns'}, optional
		If the fit uses a datetime dimension, the unit of the period may be
		specified here.

	Returns
	-------
	output : Dataset
		A Dataset with the amplitude and the phase for each periods
	"""
	if _utils.is_scalar(periods):
		periods = [periods, ]
	n = 2 * len(periods) + 1
	# Sort frequencies in ascending order
	periods.sort(reverse=True)
	# Re-order the array to place the fitting dimension as the first dimension
	# + stack the other dimensions
	array_stacked = _order_and_stack(array, dim)
	dim_chunk = array.chunks[array.get_axis_num(dim)][0]
	# Check if the dimension is associated with a numpy.datetime
	# and normalize to use periods and time in seconds
	if pd.core.common.is_datetime64_dtype(array[dim].data):
		# TODO: Check if there is a smarter way to convert time to second
		t = (array[dim] - array[dim][0]).data.astype('timedelta64[s]').\
			astype('f4')
		freqs = 1. / pd.to_timedelta(periods, unit=unit).total_seconds()
	else:
		t = array[dim]
		freqs = 1. / periods
	# Build coefficient matrix for the fit using the exponential form
	x = da.vstack([da.cos(2 * np.pi * f * t) for f in reversed(freqs)] +
	              [da.ones(len(t), chunks=dim_chunk), ] +
	              [da.sin(2 * np.pi * f * t) for f in freqs]).T
	x = x.rechunk((dim_chunk, n))
	# Solve the least-square system
	c, _, _, _ = da.linalg.lstsq(x, array_stacked.data)
	# Get cosine (a) and sine (b) ampitudes
	b = c[0:n//2, ][::-1]
	a = c[n//2 + 1:, ]
	# Compute amplitude and phase
	amplitude = da.sqrt(a ** 2 + b ** 2)
	phase = da.arctan2(b, a) * 180. / np.pi
	# Store the results
	new_dims = ('periods',) + array_stacked.dims[1:]
	var_dict = {'amplitude': (new_dims, amplitude),
	            'phase': (new_dims, phase),
	            'offset': (array_stacked.dims[1:], c[n//2, ])}
	ds = xr.Dataset(var_dict, coords=array_stacked.coords)
	ds = ds.assign_coords(periods=periods)
	ds['periods'].attrs['units'] = unit
	# Unstack the data
	output = _unstack(ds)
	return output


def sinval(modes, coords):
	raise NotImplementedError


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
	raise NotImplementedError


def _order_and_stack(obj, dim):
	"""
	Private function used to reorder to use the work dimension as the first
	dimension, stack all the dimensions except the first one
	"""
	dims_stacked = [di for di in obj.dims if di is not dim]
	new_dims = [dim, ] + dims_stacked
	if obj.ndim > 1:
		obj_stacked = obj.transpose(*new_dims).stack(temp_dim=dims_stacked)
	else:
		obj_stacked = obj
	return obj_stacked


def _unstack(obj):
	"""
	Private function used to reorder to use the work dimension as the first
	dimension, stack all the dimensions except the first one
	"""
	if 'temp_dim' in obj.dims:
		obj_unstacked = obj.unstack('temp_dim')
	else:
		obj_unstacked = obj
	return obj_unstacked
