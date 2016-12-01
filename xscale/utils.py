"""This is where useful internal functions are stored.
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
from collections import Iterable


def is_dict_like(value):
	return hasattr(value, '__getitem__') and hasattr(value, 'keys')


def is_scalar(value):
	""" Whether to treat a value as a scalar. Any non-iterable, string, or 0-D array """
	return (getattr(value, 'ndim', None) == 0
	        or isinstance(value, basestring)
	        or not isinstance(value, Iterable))

def is_scalar(value):
	""" Whether to treat a value as a scalar. Any non-iterable, string, or 0-D array """
	return (getattr(value, 'ndim', None) == 0
	        or isinstance(value, basestring)
	        or not isinstance(value, Iterable))