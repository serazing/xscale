# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Pytest
import pytest
# Xscale
from xscale.filtering import linearfilters
from xscale.signals.signaltests import signaltest_xyt1
# Xarray
import xarray as xr
# Numpy
import numpy as np

window_list = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall', 'blackmanharris', 'flattop', 'bartlett',
               'hanning', 'barthann', 'hamming', 'kaiser', 'gaussian', 'general_gaussian', 'chebwin', 'slepian',
               'cosine', 'hann', 'get_window']

signal_xyt = signaltest_xyt1()
signal_xyt_wth_coast = signaltest_xyt1(coastlines=True)

# Testing array
shape = (50, 30, 40)
dims = ('x', 'y', 'z')
cx = np.pi * np.linspace(0, 2, 50)
cy = np.linspace(0.01, 0.5, 30)
coords = {'x': cx, 'y': cy}
dummy_array = xr.DataArray(np.random.random(shape), dims=dims, coords=coords)


def test_init_window():
	signal_xyt.window


def test_infer_arg():
	# Case cutoff is None
	assert linearfilters._infer_arg(None, dims) == {'x': None, 'y': None, 'z': None}
	# Case cutoff is a float
	assert linearfilters._infer_arg(0.01, dims) == {'x': 0.01, 'y': 0.01, 'z': 0.01}
	# Case cutoff is iterable
	assert linearfilters._infer_arg([0.01, 0.05, 0.2], dims) == {'x': 0.01, 'y': 0.05, 'z': 0.2}
	# Case cutoff is iterable
	assert linearfilters._infer_arg([0.01, 0.05], dims) == {'x': 0.01, 'y': 0.05, 'z': None}
	# Case cutoff is a dictionnary
	assert linearfilters._infer_arg({'z': 0.01, 'y': 0.05}, dims) == {'x': None, 'y': 0.05, 'z': 0.01}
	# Case cutoff is a dictionnary
	assert linearfilters._infer_arg({'z': [0.01, 0.05], 'y': 0.05}, dims) == {'x': None, 'y': 0.05, 'z': [0.01, 0.05]}


def test_set_nyquist():
	w = dummy_array.window
	w.set(dims=['y', 'x'])
	assert w.fnyq == {'x': 1. / (2. * (cx[1] - cx[0])), 'y': 1. / (2. * (cy[1] - cy[0]))}


# def test_set_all_windows1d():
#	""" Test the setting of all the available 1D window """
#	w = signal_xyt.win
#	for window_name in window_list:
#		w.set(window=(window_name, 0.5), dims='time', n=24)


# def test_set_all_window2d():
#	win2d = signal_xyt.win
#	for window_name in window_list:
#		win2d.set(window_name=window_name, dims=['y', 'x'], n=[24, 36])

def test_wrong_window_name():
	""" Test the exception if the window_name is not recognize """
	w = signal_xyt.window
	with pytest.raises(ValueError, message="Expecting ValueError"):
		w.set(window='circlecar')

def test_wrong_dimension():
	"""Test if an exception is returned if the dimension is not in the associated array """
	win = signal_xyt.window
	with pytest.raises(ValueError, message="Expecting ValueError"):
		win.set(dims=['z'])

def test_compute_boundary_weights():
	win2d = signal_xyt_wth_coast.window
	win2d.set(window='hanning', cutoff=0.05, dims=['y', 'x'], n=[24, 36])
	win2d.boundary_weights(drop_dims=['time'])

# def test_window_plot1d():
#	win = signal_xyt.win
#	win.set(window_name='lanczos', dims=['time'], n=[12])
#	win.plot()


# def test_window_plot2d():
#	win = signal_xyt.win
#	win.set(window_name='lanczos', dims=['y','x'], n=[24, 36])
#	win.plot()
