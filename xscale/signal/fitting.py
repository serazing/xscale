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


def _polyfit(darray, x, deg, dim):
	# Re-order the array to place the fitting dimension as the first dimension
	# + stack the other dimensions
	dim_chunk = darray.chunks[darray.get_axis_num(dim)][0]
	x = x.chunk(chunks={dim: dim_chunk, 'degree': deg + 1})
	darray_stacked = _order_and_stack(darray, dim)
	# Solve the least-square system
	# TO DO: Compute and store the errors associated to the fit
	p, err, _, _ = da.linalg.lstsq(x.data, darray_stacked.data)
	# Store the result in a DataArray
	new_name = '%s_poly_coeffs' %darray.name
	new_dims = ('degree',) + darray_stacked.dims[1:]
	new_coords = {co: darray_stacked.coords[co] for co in darray_stacked.coords
			      if co != dim}
	lstsq_coeffs = xr.DataArray(p, name=new_name, 
							       coords=new_coords, 
							       dims=new_dims)
	return lstsq_coeffs


def polyfit(obj, deg=1, dim=None, coord=None):
	"""
	Least squares polynomial fit.
	Fit a polynomial ``p(x) = p[deg] * x ** deg + ... + p[0]`` of degree `deg`
	Returns a vector of coefficients `p` that minimises the squared error.

	Parameters
	----------
	x : xarray.DataArray
		The array to fit
	deg : int, optional
		Degree of the fitting polynomial, Default is 1.
	dim : str, optional
		The dimension along which the data will be fitted. If not precised,
		the first dimension will be used
	coord : xarray.Coordinate, optional
		The coordinates used to based the fitting on.

	Returns
	-------
	output : xarray.DataArray
		Polynomial coefficients with a new dimension to sort the polynomial
		coefficients by degree
	"""
	if dim is None:
		dim = obj.dims[0]
	if coord is None:
		coord = obj[dim]
	if pd.api.types.is_datetime64_dtype(coord.data):
		# Use the 1e-9 to scale nanoseconds to seconds (by default, xarray use
		# datetime in nanoseconds
		t = pd.to_numeric(coord) * 1e-9
	else:
		t = coord
	# Build coefficient matrix for the fit
	x = xr.concat([t ** d for d in range(deg + 1)], dim='degree')
	x = x.transpose(dim, 'degree')
	#x = da.vstack([t ** d for d in range(deg + 1)]).T	
	if isinstance(obj, xr.DataArray):
		coeffs = _polyfit(obj, x, deg, dim)
	elif isinstance(obj, xr.Dataset):
		coeffs = obj.apply(_polyfit, args=(x, deg, dim))
	coeffs = coeffs.assign_coords(degree=range(deg + 1)) 
	return _unstack(coeffs)


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


def linreg(obj, dim=None, coord=None):
	"""
	Compute a linear regression using a least-square method

	Parameters
	----------
	obj : xarray.DataArray or xarray.Dataset
		The array on which the linear regression is computed
	dim : str, optional
		The dimension along which the data will be fitted. If not precised,
		the first dimension will be used
	coord : xarray.Coordinate, optional
		The coordinates used to based the linear regression on.

	Returns
	-------
	slope : xarray.DataArray or xarray.Dataset
		An array containing the slope of the linear regression
	offset : xarray.DataArray or xarray.Dataset
		An array containing the offset of the linear regression
	"""
	linfit = polyfit(obj, dim=dim, coord=coord, deg=1)
	offset = linfit.sel(degree=0, drop=True)
	slope = linfit.sel(degree=1, drop=True)
	return slope, offset


def trend(obj, dim=None, type='linear'):
	"""
	Compute the trend over one dimension of the input array.

	Parameters
	----------
	array : xarray.DataArray
		The data on which the trend is computed
	dim : str, optional
		Dimension over which the array will be detrended
	type : {'constant', 'linear', 'quadratic'}, optional
		Type of trend to be computed. Default is 'linear'.

	Returns
	-------
	array_trend : xarray.DataArray
		The trend associated with the input data
	"""
	coord = obj[dim]
	if _utils.is_datetime(coord.data):
		# Use the 1e-9 to scale nanoseconds to seconds (by default, xarray use
		# datetime in nanoseconds
		t = pd.to_numeric(coord) * 1e-9
	else:
		t = coord 	
	if type is 'constant':
		obj_trend = obj.mean(dim=dim)
		_, obj_trend = xr.broadcast(obj, obj_trend)
	elif type is 'linear':
		slope, offset = linreg(obj, dim=dim)
		obj_trend = t * slope + offset
	elif type is 'quadratic':
		raise NotImplementedError
	else:
		raise ValueError('This type of trend is not supported')
	return obj_trend


def detrend(data, dim=None, type='linear'):
	"""
	Remove a trend over one dimension of the data.

	Parameters
	----------
	array : xarray.DataArray or xarray.Dataset
		Data to be detrended
	dim : str, optional
		Dimension over which the array will be detrended
	type : {'constant', 'linear', 'quadratic'}, optional
		Type of trend to be removed. Default is 'linear'.

	Returns
	-------
	data_detrended : xarray.DataArray or xarray.Dataset
		The detrended data
	"""
	if isinstance(data, xr.DataArray):
		data_trend = trend(data, dim=dim, type=type)
		data_detrended = data - data_trend
	elif isinstance(data, xr.Dataset):
		data_detrended = data.apply(detrend)
	return data_detrended


def sinfit(array, periods, dim=None, coord=None, unit='s'):
	"""
	Least squares sinusoidal fit.
	Fit sinusoidal functions ``y = A[p] * sin(2 * pi * ax * f[1] + phi[1])``

	Parameters
	----------
	array : xarray.DataArray
		Data to be fitted
	periods: float or list of float
		The periods of the sinusoidal functions to be fitted
	dim : str, optional
		The dimension along which the data will be fitted. If not precised,
		the first dimension will be used
	unit : {'D', 'h', 'm', 's', 'ms', 'us', 'ns'}, optional
		If the fit uses a datetime dimension, the unit of the period may be
		specified here.

	Returns
	-------
	modes : Dataset
		A Dataset with the amplitude and the phase for each periods
	"""
	if dim is None:
		dim = array.dims[0]
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
	if coord is None:
		coord = array[dim]
	if _utils.is_datetime(coord):
		# Use the 1e-9 to scale nanoseconds to seconds (by default, xarray use
		# datetime in nanoseconds
		t = coord.data.astype('f8') * 1e-9
		freqs = 1. / pd.to_timedelta(periods, unit=unit).total_seconds()
	else:
		t = coord.data
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
	new_coords = {co: array_stacked.coords[co] for co in array_stacked.coords
	              if co != dim}
	var_dict = {'amplitude': (new_dims, amplitude),
	            'phase': (new_dims, phase),
	            'offset': (array_stacked.dims[1:], c[n//2, ])}
	ds = xr.Dataset(var_dict, coords=new_coords)
	ds = ds.assign_coords(periods=periods)
	ds['periods'].attrs['units'] = unit
	# Unstack the data
	modes = _unstack(ds)
	return modes


def sinval(modes, coord):
	"""
	Evaluate a sinusoidal function based on a modal decomposition. Each mode is
	defined by a period, an amplitude and a phase. This function may usually
	be used after a sinusoidal fit using
	:py:func:`xscale.signal.fitting.sinfit`.

	Parameters
	----------
	modes : xarray.Dataset
		A dataset where the amplitude and phase are stored for each mode
	coord : xarray.Coordinates
		A coordinate array at which the sine functions are evaluated

	Returns
	-------
	res : xarray.DataArray

	"""
	modes_dims = tuple([di for di in modes.dims if di is not 'periods'])
	modes_shape = tuple([modes.dims[di] for di in modes_dims])
	modes_chunks = tuple(modes.chunks[di][0] for di in modes_dims)
	if coord.chunks is  None:
		coord_chunks = (coord.shape[0],)
	else:
		coord_chunks = (coord.chunks[0][0],)
	new_dims = coord.dims + modes_dims
	new_shape = coord.shape + modes_shape
	new_chunks = coord_chunks + modes_chunks
	ones = xr.DataArray(da.ones(new_shape, chunks=new_chunks), dims=new_dims)
	if _utils.is_datetime(coord):
		# TODO: Check if there is a smarter way to convert time to second
		t = ones * coord.astype('f8') * 1e-9
		pd_periods = pd.to_datetime(modes['periods'],
		                            unit=modes['periods'].units)
		if _utils.is_scalar(modes['periods'].data):
			periods = pd_periods.value.astype('f8') * 1e-9
		else:
			periods = pd_periods.values.astype('f8') * 1e-9
	else:
		t = ones * coord
		periods = modes['periods']
	res = ones * modes['offset']
	for p in range(len(periods)):
		modep = ones * modes.isel(periods=p)
		res += modep['amplitude'] * xr.ufuncs.sin(2 * np.pi * t / periods[p] +
		                                          modep['phase'] * np.pi / 180.)
	return res



def _order_and_stack(obj, dim):
	"""
	Private function used to reorder to use the work dimension as the first
	dimension, stack all the dimensions except the first one
	"""
	dims_stacked = [di for di in obj.dims if di != dim]
	new_dims = [dim, ] + dims_stacked
	if obj.ndim > 2:
		obj_stacked = (obj.transpose(*new_dims)
                          .stack(temp_dim=dims_stacked)
                          .dropna('temp_dim'))
	elif obj.ndim == 2:
		obj_stacked = obj.transpose(*new_dims)
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