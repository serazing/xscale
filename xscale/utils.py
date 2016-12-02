"""This is where useful internal functions are stored.
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
from collections import Iterable
# Pandas
import pandas as pd
import numpy as np

def is_dict_like(value):
	return hasattr(value, '__getitem__') and hasattr(value, 'keys')


def is_scalar(value):
	""" Whether to treat a value as a scalar. Any non-iterable, string, or 0-D array """
	return (getattr(value, 'ndim', None) == 0
	        or isinstance(value, basestring)
	        or not isinstance(value, Iterable))

def is_iterable(value):
	return isinstance(value, Iterable) and not isinstance(value, basestring)

def infer_n_and_dims(obj, n, dims):
	"""Logic for setting the window properties"""
	#TODO: Finish this function
	if n is None:
		if dims is None:
			new_n = obj.shape
			new_dims = obj.dims
		else:
			new_n = tuple([obj.shape[obj.get_axis_num(di)] for di in dims])
			new_dims = dims
	elif is_dict_like(n):
		new_n = n.values()
		new_dims = n.keys()
	elif isinstance(n, int):
		if dims is None:
			new_n = tuple([n for number in range(obj.ndim)])
			new_dims = obj.dims
		elif isinstance(dims, basestring):
			new_n = (n, )
			new_dims = (dims, )
		elif isinstance(dims, Iterable):
			new_n = tuple([n for number in range(len(dims))])
			new_dims = dims
		else:
			raise TypeError("This type of option is not supported for the second argument")
	elif is_iterable(n):
		if is_iterable(dims):
			if len(n) == len(dims):
				new_n = n
				new_dims = dims
			else:
				raise ValueError("Dimensions must have the same length as the first argument")
		else:
			raise TypeError("Dimensions must be specificed with an Iterable")
	else:
		raise TypeError("This type of option is not supported for the first argument")
	return new_n, new_dims


def get_dx(obj, dim):
	"""Get the resolution over one the dimension dim"""
	delta = np.diff(obj[dim])
	if pd.core.common.is_timedelta64_dtype(delta):
		# Convert to seconds so we get hertz
		delta = delta.astype('timedelta64[s]').astype('f8')
	if not np.allclose(delta, delta[0]):
		raise Warning("Coordinate %s is not evenly spaced" % dim)
	dx = delta[0]
	return dx
