# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

import xscale.spectral.fft as xfft
import xarray as xr
import numpy as np
import pytest


def test_fft_real_1d():
	""" Compare the result from the spectrum._fft function to numpy.fft.rfft
	"""
	a = [0, 1, 0, 0]
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, spectrum_coords, spectrum_dims = \
		xfft._fft(chunked_array, nfft={'x': 4}, dim=['x'], dx={'x': 0.01},
		          sym=False)
	assert np.array_equal(spectrum_array.compute(), np.fft.rfft(a))
	assert np.array_equal(spectrum_coords['f_x'], np.fft.rfftfreq(4, d=0.01))
	assert 'f_x' in spectrum_dims


def test_fft_complex_1d():
	""" Compare the result from the spectrum.fft function to numpy.fft.fft
	"""
	a = np.exp(2j * np.pi * np.arange(8) / 8)
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spectrum_array, spectrum_coords, spectrum_dims = \
		xfft._fft(chunked_array, nfft={'x': 16}, dim=['x'], dx={'x': 0.5})
	assert np.array_equal(spectrum_array.compute(), np.fft.fft(a, n=16))
	assert np.array_equal(spectrum_coords['f_x'], np.fft.fftfreq(16, d=0.5))
	assert 'f_x' in spectrum_dims


def test_fft_real_2d():
	""" Compare the result from the spectrum.fft function to numpy.fft.rfftn
	"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, spectrum_coords, spectrum_dims = \
		xfft._fft(chunked_array, nfft={'y': 14, 'z': 18}, dim=['y', 'z'],
		          dx={'y': 0.01, 'z': 0.02}, sym=False)
	assert np.allclose(spectrum_array.compute(),
	                   np.fft.rfftn(a, s=(18, 14), axes=(2, 1)))
	assert np.array_equal(spectrum_coords['f_y'], np.fft.rfftfreq(14, d=0.01))
	assert np.array_equal(spectrum_coords['f_z'], np.fft.fftfreq(18, d=0.02))
	assert ('x', 'f_y', 'f_z') == spectrum_dims


def test_fft_complex_2d():
	""" Compare the result from the spectrum.fft function to
	numpy.fft.fftn
	"""
	a, b, c = np.meshgrid([0, 1, 0, 0], [0, 1j, 1j], [0, 1, 1, 1])
	dummy_array = xr.DataArray(a * b * c, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, spectrum_coords, spectrum_dims = \
		xfft._fft(chunked_array, nfft={'y': 6, 'z': 8}, dim=['y', 'z'],
		          dx={'y': 0.01, 'z': 0.02})
	assert np.allclose(spectrum_array.compute(),
	                   np.fft.fftn(a * b * c, s=(8, 6), axes=(2, 1)))
	assert np.array_equal(spectrum_coords['f_y'], np.fft.fftfreq(6, d=0.01))
	assert np.array_equal(spectrum_coords['f_z'], np.fft.fftfreq(8, d=0.02))
	assert ('x', 'f_y', 'f_z') == spectrum_dims


#@pytest.mark.skip(reason="Core dumped")
def test_fft_real_3d():
	""" Compare the result from the spectrum.fft function to numpy.fft.rfftn
	"""
	a = np.mgrid[:7, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 7, 'y': 5, 'z': 5})
	spectrum_array, spectrum_coords, spectrum_dims = \
		xfft._fft(chunked_array, nfft={'x': 11, 'y': 14, 'z': 18},
		          dim=['x', 'y', 'z'], dx={'x':12, 'y': 0.01, 'z': 0.02},
		          sym=False)
	assert np.allclose(spectrum_array.compute(),
	                   np.fft.rfftn(a.T, s=(18, 14, 11)).T)
	assert np.array_equal(spectrum_coords['f_x'], np.fft.rfftfreq(11, d=12))
	assert np.array_equal(spectrum_coords['f_y'], np.fft.fftfreq(14, d=0.01))
	assert np.array_equal(spectrum_coords['f_z'], np.fft.fftfreq(18, d=0.02))
	assert ('f_x', 'f_y', 'f_z') == spectrum_dims


def test_fft_complex_3d():
	""" Compare the result from the spectrum.fft function to numpy.fft.fftn
	"""
	a, b, c = np.meshgrid([0, 1, 0, 0], [0, 1j, 1j], [0, 1, 1, 1])
	dummy_array = xr.DataArray(a * b * c, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	spectrum_array, spectrum_coords, spectrum_dims = \
		xfft._fft(chunked_array, nfft={'x': 8, 'y': 6, 'z': 8},
		          dim=['x', 'y', 'z'], dx={'x':12, 'y': 0.01, 'z': 0.02})
	assert np.allclose(spectrum_array.compute(),
	                      np.fft.fftn(a * b * c, s=(8, 6, 8)))
	assert np.array_equal(spectrum_coords['f_x'], np.fft.fftfreq(8, d=12))
	assert np.array_equal(spectrum_coords['f_y'], np.fft.fftfreq(6, d=0.01))
	assert np.array_equal(spectrum_coords['f_z'], np.fft.fftfreq(8, d=0.02))
	assert ('f_x', 'f_y', 'f_z') == spectrum_dims


def test_fft_warning():
	"""Test if a warning is raise if a wrong dimension is used
	"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	with pytest.warns(UserWarning):
		xfft.fft(chunked_array, dim=['x', 'y', 'time'])


@pytest.mark.parametrize("tapering",  [True, False])
def test_spectrum_1d(tapering):
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['time', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'time': 2, 'y': 2, 'z': 2})
	xfft.fft(chunked_array, dim='time', dx=1., tapering=tapering).load()


@pytest.mark.parametrize("tapering",  [True, False])
def test_spectrum_2d(tapering):
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	xfft.fft(chunked_array, dim=['y', 'z'], tapering=tapering).load()


def test_parserval_real_1d():
	"""Test if the Parseval theorem is verified"""
	a = [0, 1, 0, 1, 1, 0, 1]
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spec = xfft.fft(chunked_array, dim=['x'], detrend='mean')
	assert np.isclose(np.var(a), xfft.ps(spec).sum())


def test_parserval_complex_1d():
	"""Test if the Parseval theorem is verified"""
	a = np.exp(2j * np.pi * np.arange(8) / 8)
	dummy_array = xr.DataArray(a, dims=['x'])
	chunked_array = dummy_array.chunk(chunks={'x': 2})
	spec = xfft.fft(chunked_array, dim=['x'], detrend='mean')
	assert np.var(a) == xfft.ps(spec).sum()


def test_parseval_real_2d():
	"""Test if the Parseval theorem is verified"""
	a = np.mgrid[:5, :5, :5][0]
	dummy_array = xr.DataArray(a, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	chunked_array_zeromean = chunked_array - chunked_array.mean(dim=['y', 'z'])
	spec = xfft.fft(chunked_array_zeromean, dim=['y', 'z'])
	assert np.allclose(np.var(a, axis=(1, 2)),
	                      xfft.ps(spec).sum(dim=['f_y','f_z']))


def test_parserval_complex_2d():
	""" Compare the result from the spectrum.fft function to
	numpy.fft.fftn
	"""
	a, b, c = np.meshgrid([1j, 1, 1, 1j], [0, 1j, 1j], [0, 1, 1, 1])
	dummy_array = xr.DataArray(a * b * c, dims=['x', 'y', 'z'])
	chunked_array = dummy_array.chunk(chunks={'x': 2, 'y': 2, 'z': 2})
	chunked_array_zeromean = chunked_array - chunked_array.mean(dim=['y', 'z'])
	spec = xfft.fft(chunked_array_zeromean, dim=['y', 'z'], sym=True)
	assert np.allclose(np.var(a * b * c, axis=(1, 2)),
	                      xfft.ps(spec).sum(dim=['f_y','f_z']))