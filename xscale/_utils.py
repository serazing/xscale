"""This is where useful internal functions are stored.
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
from collections import Iterable
# Pandas
import pandas as pd
# Numpy
import numpy as np
# Warnings
import warnings


def is_dict_like(value):
	return hasattr(value, '__getitem__') and hasattr(value, 'keys')


def is_scalar(value):
	""" Whether to treat a value as a scalar. Any non-iterable, string, or 0-D array """
	return (getattr(value, 'ndim', None) == 0
	        or isinstance(value, str)
	        or not isinstance(value, Iterable))


def is_iterable(value):
	return isinstance(value, Iterable) and not isinstance(value, str)


def is_datetime(value):
	return pd.api.types.is_datetime64_dtype(value)


def homogeneous_type(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def infer_n_and_dims(obj, n, dims):
	"""Logic for setting the window properties"""
	#TODO: Finish this function
	if n is None:
		if dims is None:
			new_n = obj.shape
			new_dims = obj.dims
		elif isinstance(dims, str):
			new_n = (obj.shape[obj.get_axis_num(dims)], )
			new_dims = (dims, )
		else:
			new_n = tuple()
			new_dims = tuple()
			for di in dims:
				if di in obj.dims:
					new_n += (obj.shape[obj.get_axis_num(di)], )
					new_dims += (di, )
				else:
				    warnings.warn("Cannot find dimension %s in DataArray" % di)
	elif is_dict_like(n):
		new_n = tuple(n.values())
		new_dims = tuple(n.keys())
	elif isinstance(n, int):
		if dims is None:
			new_n = tuple([n for number in range(obj.ndim)])
			new_dims = obj.dims
		elif isinstance(dims, str):
			if dims in obj.dims:
				new_n = (n, )
				new_dims = (dims, )
			else:
				warnings.warn("Cannot find dimension %s in DataArray" % dims)
		elif isinstance(dims, Iterable):
			new_n = tuple()
			new_dims = tuple()
			for di in dims:
				if di in obj.dims:
					new_n += (n, )
					new_dims += (di,)
				else:
					warnings.warn("Cannot find dimension %s in DataArray" % di)
		else:
			raise TypeError("This type of option is not supported for the "
			                "second argument")
	elif is_iterable(n):
		if is_iterable(dims):
			if len(n) == len(dims):
				new_n = tuple()
				new_dims = tuple()
				for i, di in zip(n, dims):
					if di in obj.dims:
						new_n += (i,)
						new_dims += (di,)
					else:
						warnings.warn("Cannot find dimension %s in "
						              "DataArray" % di)
			else:
				raise ValueError("Dimensions must have the same length as the "
				                 "first argument")
		else:
			raise TypeError("Dimensions must be specificed with an Iterable")
	else:
		raise TypeError("This type of option is not supported for the first "
		                "argument")
	return new_n, new_dims


def infer_arg(arg, dims, default_value=None):
	new_arg = dict()
	if arg is None:
		if isinstance(dims, str):
			new_arg[dims] = default_value
		else:
			new_arg = {di: default_value for di in dims}
	elif is_scalar(arg):
		if isinstance(dims, str):
			new_arg[dims] = arg
		else:
			new_arg = {di: arg for di in dims}
	elif is_dict_like(arg):
		if isinstance(dims, str):
			new_arg[dims] = arg[dims]
		else:
			for di in dims:
				try:
					new_arg[di] = arg[di]
				except (KeyError, IndexError):
					new_arg[di] = default_value
	elif isinstance(arg, Iterable) and not isinstance(arg, str):
		if isinstance(dims, str):
			if len(arg) == 1:
				new_arg[dims] = arg[0]
			elif not homogeneous_type(arg):
				new_arg[dims] = arg
			else:
				raise ValueError("The two arguments do not coincide")
		else:
			if homogeneous_type(arg):
				for i, di in enumerate(dims):
					# if not len(dims) == len(arg):
					try:
						new_arg[di] = arg[i]
					except (KeyError, IndexError):
						new_arg[di] = default_value
					except TypeError:
						new_arg[dims[di]] = arg
			else:
				for i, di in enumerate(dims):
					try:
						new_arg[di] = arg
					except TypeError:
						new_arg[dims[di]] = arg
	else:
		raise TypeError("This type of argument is not supported for the second "
		                "argument")
	return new_arg


def get_dx(obj, dim, unit='s'):
	"""Get the resolution over one the dimension dim.
	Warns the user if the coordinate is not evenly spaced.

	Parameters
	----------
	obj: `xarray.DataSet` or `xarray.DataArray`
		Self-described data with coordinates corresponding to the dimensions
	dim:
		Dimension along which compute the delta
	unit: {'D', 'h', 'm', 's', 'ms', 'us', 'ns'}, optional
		If the fit the coordinates associated to the dimension is a
		numpy.datetime object, the unit of the time delta may be specified here

	Returns
	-------
	dx: float
		The resolution of the coordinates associated to the dimension
	"""
	x = np.asarray(obj[dim])
	if is_datetime(x):
		dx = pd.Series(x[1:]) - pd.Series(x[:-1])
		dx /= np.timedelta64(1, unit)
	else:
		dx = np.diff(x)
	#TODO: Small issue this the function commented below
	#if not np.allclose(dx, dx[0]):
	#	warnings.warn("Coordinate %s is not evenly spaced" % dim)
	return dx[0]
