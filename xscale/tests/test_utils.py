# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Pytest
import pytest
# Numpy
import numpy as np
# Xarray
import xarray as xr
# Xscale
from xscale import utils

# Testing array
shape = (50, 30, 40)
dims = ('x', 'y', 'z')
coords = {'x': np.pi * np.linspace(1, 50), 'y': 10 ** np.linspace(0.01, 0.5, 30)}
array = xr.DataArray(np.random.random(shape), dims=dims, coords=coords)


def test_infer_n_and_dims():
	# Case n and dims are None -> returns all dimensions and all dimensions
	assert utils.infer_n_and_dims(array, None, None) == (shape, dims)
	# Case n is None and dims is defined -> returns the dimensions and the associated shape
	assert utils.infer_n_and_dims(array, None, ('x', 'z')) == ((50, 40), ('x', 'z'))
	# Case n is a dictionary and dims is None or whatever
	dict_test = {'x': 10, 'y': 20}
	assert utils.infer_n_and_dims(array, dict_test, None) == (dict_test.values(), dict_test.keys())
	# Case n is an int and dims is None -> returns n over all dimensions
	assert utils.infer_n_and_dims(array, 10, None) == ((10, 10, 10), dims)
	# Case n is an int and dims is a BaseString
	assert utils.infer_n_and_dims(array, 15, 'z') == ((15,), ('z',))
	# Case n is an int and dims is an iterable
	assert utils.infer_n_and_dims(array, 8, ['y', 'x']) == ((8, 8), ['y', 'x'])
	# Case n is an iterable and dims is an iterable
	assert utils.infer_n_and_dims(array, (8, 15), ['y', 'z']) == ((8, 15), ['y', 'z'])
	# Test exceptions
	# Case n is an iterable and dims is not an iterable
	with pytest.raises(TypeError, message="Expecting TypeError"):
		utils.infer_n_and_dims(array, (8, 15), 'y')
	# Case n is an iterable and dims is not an iterable
	with pytest.raises(ValueError, message="Expecting ValueError"):
		utils.infer_n_and_dims(array, (8, 15), ['y'])
	# Case n is not valid
	with pytest.raises(TypeError, message="Expecting TypeError"):
		utils.infer_n_and_dims(array, '8', ['y'])
	# Case n is not valid
	with pytest.raises(TypeError, message="Expecting TypeError"):
		utils.infer_n_and_dims(array, 8, 14)


def test_get_dx():
	assert utils.get_dx(array, 'x') == np.pi
	with pytest.raises(Warning):
		utils.get_dx(array, 'y')