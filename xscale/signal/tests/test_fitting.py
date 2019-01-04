# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
import xarray as xr
import numpy as np
import pandas as pd
import xscale.signal.fitting as xfit

def test_polyfit():
	Nt, Nx, Ny = 100, 128, 128
	rand = xr.DataArray(np.random.rand(Nt, Nx, Ny), dims=['time', 'x', 'y'])
	slopes = 0.02 * xr.DataArray(np.cos(2 * np.pi * rand.x / Nx), dims=['x'])
	truth = rand + slopes * rand.time
	truth = truth.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	linfit = xfit.polyfit(truth, dim='time').load()
	xfit.polyfit(truth.to_dataset(name='truth'), dim='time').load()
	assert np.allclose(linfit.sel(degree=1).mean(dim='y').data, slopes.data,
	                   rtol=5e-2, atol=1e-3)

def test_linreg():
	nt, nx, ny = 100, 128, 128
	offset = 0.7 * xr.DataArray(np.ones((nt, nx, ny)), dims=['time', 'x', 'y'])
	slopes = 0.02 * xr.DataArray(np.cos(2 * np.pi * offset.x / nx), dims=['x'])
	truth = offset + slopes * offset.time
	truth = truth.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	xfit.polyfit(truth.to_dataset(name='truth'), dim='time').load()
	slopes_fitted, offsets_fitted = xfit.linreg(truth, dim='time')
	assert np.allclose(slopes, slopes_fitted.mean(dim='y').load())
	assert np.allclose(offset, offsets_fitted.mean(dim='y').load())

def test_trend():
	nt, nx, ny = 100, 128, 128
	offset = 0.7 * xr.DataArray(np.ones((nt, nx, ny)), dims=['time', 'x', 'y'])
	slopes = 0.02 * xr.DataArray(np.cos(2 * np.pi * offset.x / nx), dims=['x'])
	truth = offset + slopes * offset.time
	truth = truth.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	trend_mean = xfit.trend(offset, dim='time', type='constant')
	trend_linear = xfit.trend(truth, dim='time', type='linear')
	assert np.allclose(offset, trend_mean.load())
	assert np.allclose(truth, trend_linear.load())

def test_detrend():
	nt, nx, ny = 100, 128, 128
	offset = 0.7 * xr.DataArray(np.ones((nt, nx, ny)), dims=['time', 'x', 'y'])
	slopes = 0.02 * xr.DataArray(np.cos(2 * np.pi * offset.x / nx), dims=['x'])
	truth = offset + slopes * offset.time
	truth = truth.chunk(chunks={'time': 20, 'x': 50, 'y': 50})
	assert np.allclose(0 * offset, xfit.detrend(offset, dim='time',
	                                            type='constant').load())
	assert np.allclose(0 * offset, xfit.detrend(truth, dim='time',
	                                            type='linear').load())

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
	Nt, Nx, Ny = 100, 128, 128
	offset = 0.4
	periods = [24., 12.]
	amp1, phi1 = 1.2, 0.
	amp2, phi2 = 1.9, 60.
	time = xr.DataArray(pd.date_range(start='2011-01-01',
	                                  periods=Nt,
	                                  freq='H'),
	                    dims='time')
	amp = xr.DataArray([amp1, amp2], dims='periods')
	phi = xr.DataArray([phi1, phi2], dims='periods')
	ones = xr.DataArray(np.ones((Nx, Ny)), dims=['x', 'y'])
	var_dict = {'amplitude': amp * ones,
	            'phase': phi * ones,
	            'offset': offset * ones}
	ds = xr.Dataset(var_dict).chunk(chunks={'x': 50, 'y': 50})
	ds = ds.assign_coords(periods=periods)
	ds['periods'].attrs['units'] = 'h'
	xfit.sinval(ds, time)
	#One mode reconstruction
	xfit.sinval(ds.sel(periods=[24,]), time)


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