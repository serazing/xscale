# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Pytest
import pytest
# Xscale
from xscale.filtering import linearfilters
from xscale.signal.generator import signaltest_xyt
# Pandas
import pandas as pd
# Xarray
import xarray as xr
# Dask
import dask.array as da
# Numpy
import numpy as np
# Scipy
import scipy.signal as sig
# Matlplotlib
import matplotlib.pyplot as plt

window_list = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
               'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann',
               'hamming', ('kaiser', 2), ('tukey', 0.5)]
               #'gaussian',
               #'general_gaussian', 'chebwin',
               #'slepian', 'cosine', 'hann']


sig_xyt = signaltest_xyt()
sig_xyt_wth_coast = signaltest_xyt(coastlines=True)

# Testing array
shape = (48, 30, 40)
dims = ('time', 'y', 'x')
ctime = pd.date_range('2000-01-01', periods=48, freq='M')
cy = np.linspace(0.01, 0.5, 30)
cx = np.pi * np.linspace(0, 2, 40)
coords = {'time': ctime, 'y': cy, 'x': cx}
dummy_array = xr.DataArray(np.random.random(shape), dims=dims, coords=coords)


def test_set_nyquist():
	w = dummy_array.window
	w.set(dim=['y', 'x'])
	assert w.fnyq == {'x': 1. / (2. * (cx[1] - cx[0])),
	                  'y': 1. / (2. * (cy[1] - cy[0]))}


def test_init_window():
	sig_xyt.window


def test_wrong_window_name():
	""" Test the exception if the window_name is not recognize """
	w = sig_xyt.window
	with pytest.raises(ValueError, message="Expecting ValueError"):
		w.set(window='circlecar')


def test_wrong_dimension():
	"""Test if an exception is returned if the dimension is not in the associated array """
	w = sig_xyt.window
	with pytest.warns(UserWarning, message="Expecting a message to warns that"
	                                       "the user used a wrong dimension"):
		w.set(dim=['z'])


@pytest.mark.parametrize("window",  window_list)
def test_compute_boundary_weights(window):
	pytest.skip()
	win2d = sig_xyt_wth_coast.window
	win2d.set(window=window, cutoff=20, dim=['y', 'x'], n=[24, 36])
	win2d.boundary_weights(drop_dims=['time'])


@pytest.mark.parametrize("window",  window_list)
def test_window_plot1d(window):
	win = sig_xyt.window
	win.set(window=window, dim='time', cutoff=6., dx=1, n=25)
	win.plot()


def test_window_plot2d():
	win = sig_xyt.window
	win.set(window='hanning', dim='time', n=12)
	win.plot()


@pytest.mark.parametrize("window",  window_list)
def test_tapper_1d(window):
	dummy_array = xr.DataArray(da.ones((10), chunks=(3,)), dims='x')
	win = dummy_array.window
	win.set(window=window, dim='x')
	assert np.array_equal(win.tapper(), sig.get_window(window, 10))