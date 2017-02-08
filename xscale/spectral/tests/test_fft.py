# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

import xscale.spectral.fft as sp
import xarray as xr
import numpy as np
import pytest

# TODO: Check the coordinates

def test_fft_real_1d():
	""" Compare the result from the spectrum._fft function to numpy.fft.rfft
	"""
	a = [0, 1, 0, 0]
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = sp._fft(chunked_array, dim=['x'], dx=1.)
	assert np.array_equal(np.asarray(spectrum_array), np.fft.rfft(a))


def test_fft_complex_1d():
	""" Compare the result from the spectrum.fft function to numpy.fft.fft
	"""
	a = np.exp(2j * np.pi * np.arange(8) / 8)
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, _, _ = sp._fft(chunked_array, dim=['x'], dx=1.)
	assert  np.array_equal(np.asarray(spectrum_array), np.fft.fft(a))


def test_fft_real_2d():
	""" Compare the result from the spectrum.fft function to numpy.fft.rfftn
	"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, _, _ = sp._fft(chunked_array, dim=['y', 'z'], dx=1.)
	assert np.array_equal(np.asarray(spectrum_array), np.fft.rfftn(a, axes=(2, 1)))


def test_fft_complex_2d():
	""" Compare the result from the spectrum.fft function to
	numpy.fft.fftn
	"""
	a, b, c = np.meshgrid([0, 1, 0, 0], [0, 1j, 1j], [0, 1, 1, 1])
	dummy_array = xr.DataArray(a * b * c, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, _, _ = sp._fft(chunked_array, dim=['y', 'z'], dx=1.)
	assert np.array_equal(np.asarray(spectrum_array),
	                      np.fft.fftn(a * b * c, axes=(-2, -1)))


def test_fft_real_3d():
	""" Compare the result from the spectrum.fft function to numpy.fft.rfftn
	"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, _, _ = sp._fft(chunked_array, dim=['x', 'y', 'z'], dx=1.)
	assert np.array_equal(np.asarray(spectrum_array),
	                      np.fft.rfftn(a, axes=(1, 2, 0)))


def test_fft_complex_3d():
	""" Compare the result from the spectrum.fft function to numpy.fft.fftn
	"""
	a, b, c = np.meshgrid([0, 1, 0, 0], [0, 1j, 1j], [0, 1, 1, 1])
	dummy_array = xr.DataArray(a * b * c, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, _, _ = sp.fft(chunked_array, dim=['x', 'y', 'z'])
	assert np.array_equal(np.asarray(spectrum_array), np.fft.fftn(a * b * c))


def test_fft_warning():
	"""Test if a warning is raise if a wrong dimension is used
	"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	with pytest.warns(UserWarning):
		sp.fft(chunked_array, dim=['x', 'y', 'time'])


def test_spectrum_1d():
	a = [0, 1, 0, 0]
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum1d = sp.fft(chunked_array, dim=['x'])


def test_spectrum_2d():
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum2d = sp.fft(chunked_array, dim=['y', 'z'])


def test_psd_1d():
	a = [0, 1, 0, 0]
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	psd1d = sp.psd(chunked_array, dim=['x'])


def test_psd_2d():
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	psd2d = sp.psd(chunked_array, dim=['y', 'z'])


def test_parserval():
	"""Test if the Parseval theorem is verified"""
	pytest.skip("Parseval test is not coded yet.")
