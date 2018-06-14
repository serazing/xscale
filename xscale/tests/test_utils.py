# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Pytest
import pytest
# Numpy
import numpy as np
# Xarray
import xarray as xr
# Pandas
import pandas as pd
# Xscale
from xscale import _utils


# Testing array
shape = (50, 30, 12)
dims = ('x', 'y', 'time')
coords = {'x': np.pi * np.linspace(1, 50),
          'y': 10 ** np.linspace(0.01, 0.5, 30),
          'time': pd.date_range('2000-01-01', periods=12, freq='M')}
array = xr.DataArray(np.random.random(shape), dims=dims, coords=coords)


def test_is_dict_like():
	assert _utils.is_dict_like({'x': 5, 'y': 8})
	assert not _utils.is_dict_like(10)


def test_is_scalar():
	assert _utils.is_scalar(10)
	assert not _utils.is_scalar({'x': 5, 'y': 8})


def test_is_iterable():
	assert _utils.is_iterable([5, 8])
	assert _utils.is_dict_like({'x': 5, 'y': 8})
	assert _utils.is_iterable((10,))
	assert not _utils.is_iterable(10)
	assert not _utils.is_iterable('john_doe')

def test_is_datetime():
	assert _utils.is_datetime(pd.date_range('1990', '2000'))
	assert not _utils.is_datetime(np.linspace(0, 15))

def test_homogeneous_type():
	assert _utils.homogeneous_type((12, 54))
	assert _utils.homogeneous_type(('string1', 'string2'))
	assert not _utils.homogeneous_type(('string', 12))


def test_infer_n_and_dims():
	# Case n and dims are None -> returns all dimensions and all dimensions
	assert _utils.infer_n_and_dims(array, None, None) == (shape, dims)
	# Case n is None and dims is defined -> returns the dimensions and the associated shape
	assert _utils.infer_n_and_dims(array, None, ('x', 'time')) == \
	       ((50, 12), ('x', 'time'))
	# Case n is None and dims is a string -> returns the dimensions and the
	# associated shape
	assert _utils.infer_n_and_dims(array, None, 'time') == ((12,), ('time',))
	# Case n is a dictionary and dims is None or whatever
	dict_test = {'x': 10, 'y': 20}
	assert _utils.infer_n_and_dims(array, dict_test, None) == \
	       (tuple(dict_test.values()), tuple(dict_test.keys()))
	# Case n is an int and dims is None -> returns n over all dimensions
	assert _utils.infer_n_and_dims(array, 10, None) == ((10, 10, 10), dims)
	# Case n is an int and dims is a BaseString
	assert _utils.infer_n_and_dims(array, 15, 'time') == ((15,), ('time',))
	# Case n is an int and dims is an iterable
	assert _utils.infer_n_and_dims(array, 8, ['y', 'x']) == ((8, 8), ('y', 'x'))
	# Case n is an iterable and dims is an iterable
	assert _utils.infer_n_and_dims(array, (8, 15),
	                               ['y', 'time']) == ((8, 15), ('y', 'time'))
	# Test exceptions
	# Case n is an iterable and dims is not an iterable
	with pytest.raises(TypeError, message="Expecting TypeError"):
		_utils.infer_n_and_dims(array, (8, 15), 'y')
	# Case n is an iterable and dims is not an iterable
	with pytest.raises(ValueError, message="Expecting ValueError"):
		_utils.infer_n_and_dims(array, (8, 15), ['y'])
	# Case n is not valid
	with pytest.raises(TypeError, message="Expecting TypeError"):
		_utils.infer_n_and_dims(array, '8', ['y'])
	# Case n is not valid
	with pytest.raises(TypeError, message="Expecting TypeError"):
		_utils.infer_n_and_dims(array, 8, 14)
	# Raise a warning if the dimension is not found, and skipped it
	with pytest.warns(UserWarning):
		_utils.infer_n_and_dims(array, (8, 15), ['z', 'time'])
	assert _utils.infer_n_and_dims(array, (8, 15), ['z', 'time']) == ((15,),
	                                                                  ('time',))

def test_infer_arg():
	dims = ('time', 'y', 'x')
	# Case arg is None
	assert _utils.infer_arg(None, dims) == {'time': None, 'y': None, 'x': None}
	# Case arg is a float
	assert _utils.infer_arg(0.01, dims) == {'time': 0.01, 'y': 0.01, 'x': 0.01}
	# Case arg is iterable
	assert _utils.infer_arg([0.01, 0.05, 0.2], dims) == {'time': 0.01,
	                                                     'y': 0.05, 'x': 0.2}
	# Case arg is iterable
	assert _utils.infer_arg([0.01, 0.05], dims) == {'time': 0.01, 'y': 0.05,
	                                                'x': None}
	# Case arg is a dictionnary
	assert _utils.infer_arg({'y': 0.01, 'x': 0.05}, dims) == {'time': None,
	                                                          'y': 0.01,
	                                                          'x': 0.05}
	# Case arg is a dictionnary
	assert _utils.infer_arg({'time': [0.01, 0.05], 'y': 0.05},
	                        dims) == {'x': None, 'y': 0.05,
	                                  'time': [0.01, 0.05]}
	assert _utils.infer_arg([1,], 'x') == {'x': 1}
	assert _utils.infer_arg(1, 'time') == {'time': 1}
	assert _utils.infer_arg({'time': 1}, 'time') == {'time': 1}
	assert _utils.infer_arg([1, ], 'time') == {'time': 1}
	assert _utils.infer_arg([1, ], ['time',]) == {'time': 1}
	assert _utils.infer_arg(None, 'time') == {'time': None}
	assert _utils.infer_arg(('name', 36), 'time') == {'time': ('name', 36)}
	assert _utils.infer_arg({'x': ('name', 36)}, ('y', 'x')) == {'y': None,
	                                                       'x': ('name', 36)}
	assert _utils.infer_arg(('name', 36), ('y', 'x')) == {'y': ('name', 36),
	                                                       'x': ('name', 36)}
	with pytest.raises(TypeError, message="Expecting ValueError"):
		_utils.infer_n_and_dims((12, 36), 'time')

def test_get_dx():
	assert _utils.get_dx(array, 'x') == np.pi
	#with pytest.warns(UserWarning):
	#	_utils.get_dx(array, 'y')
	assert _utils.get_dx(array, 'time') == (29 * 24 * 3600.)
	assert _utils.get_dx(array, 'time', unit='h') == (29 * 24)
	assert _utils.get_dx(array, 'time', unit='D') == (29)
