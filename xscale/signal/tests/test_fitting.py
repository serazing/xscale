# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
import xarray as xr
import numpy as np
import pandas as pd
import xscale.signal.fitting as xfit


def test_sinfit():
	Nt, Nx, Ny = 100, 128, 128
	zeros = xr.DataArray(np.zeros((Nt, Nx, Ny)), dims=['time', 'x', 'y'])
	zeros = zeros.assign_coords(time=pd.date_range(start='2011-01-01',
	                                             periods=100, freq='H'))
	offset = 0.4
	amp1, phi1 = 1.2, 0.
	wave1 = amp1 * np.sin(2 * np.pi * zeros['time.hour'] / 24. +
	                      phi1 * np.pi / 180.)
	amp2, phi2 = 1.9, 60.
	wave2 = amp2 * np.sin(2 * np.pi * zeros['time.hour'] / 12. +
	                      phi2 * np.pi / 180.)
	truth = offset + zeros + wave1 + wave2
	truth = truth.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	# Fit both waves
	fit2w = xfit.sinfit(truth, dim='time', periods=[24, 12], unit='h').load()
	assert np.isclose(fit2w['amplitude'].sel(periods=24).isel(x=10, y=10), amp1)
	assert np.isclose(fit2w['phase'].sel(periods=24).isel(x=10, y=10), phi1,
	                  atol=1e-4)
	assert np.isclose(fit2w['amplitude'].sel(periods=12).isel(x=10, y=10), amp2)
	assert np.isclose(fit2w['phase'].sel(periods=12).isel(x=10, y=10), phi2)
	assert np.isclose(fit2w['offset'].isel(x=10, y=10), offset)
	# Fit only one wave (wave2)
	fit1w = xfit.sinfit(truth, dim='time', periods=12, unit='h').load()
	# Compare with 5% relative tolerance (error induced by wave1)
	assert np.isclose(fit1w['amplitude'].sel(periods=12).isel(x=10, y=10),
	                  amp2, rtol=5e-2)
	assert np.isclose(fit1w['phase'].sel(periods=12).isel(x=10, y=10),
	                  phi2, rtol=5e-2)
	# Fit only one dimensional data
	xfit.sinfit(truth.isel(x=0, y=0), dim='time',
	            periods=[24, 12],
	            unit='h').load()

def test_sinval():
	pass

def test_order_and_stack():
	rand = xr.DataArray(np.random.rand(100, 128, 128), dims=['time', 'x', 'y'])
	rand = rand.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	rand_stacked = xfit._order_and_stack(rand, 'y')
	assert rand_stacked.dims[0] is 'y'
	assert rand_stacked.dims[-1] is 'temp_dim'
	assert rand_stacked.shape[-1] == 128 * 100
	# Test the exception for 1d array
	rand1d = rand.isel(time=0, x=0)
	rand1d_stacked = xfit._order_and_stack(rand1d, 'y')
	assert np.array_equal(rand1d_stacked, rand1d)


def test_unstack():
	rand = xr.DataArray(np.random.rand(100, 128, 128), dims=['time', 'x', 'y'])
	rand = rand.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	rand_stacked = xfit._order_and_stack(rand, 'y')
	rand_unstacked = xfit._unstack(rand_stacked.mean(dim='y'))
	assert rand_unstacked.dims == ('time', 'x')
	assert rand_unstacked.shape == (100, 128)