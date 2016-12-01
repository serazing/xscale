# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
# Pytest
import pytest
# Xscale
from xscale.filtering import linearfilters
from xscale.signals.signaltests import signaltest_xyt1


window_list = linearfilters._scipy_window_dict.keys() + linearfilters._local_window_dict.keys()
signal_xyt = signaltest_xyt1()
signal_xyt_wth_coast = signaltest_xyt1(coastlines=True)

def test_init_window():
	assert signal_xyt.win

def test_set_all_windows1d():
	""" Test the setting of all the available 1D window
	"""
	win1d = signal_xyt.win
	for window_name in window_list:
		win1d.set(window_name=window_name, dims=['time'], n=[24])


def test_set_all_window2d():
	win2d = signal_xyt.win
	for window_name in window_list:
		win2d.set(window_name=window_name, dims=['y', 'x'], n=[24, 36])


def test_wrong_window_name():
	""" Test the exception if the window_name is not recognize
	"""
	win = signal_xyt.win
	with pytest.raises(ValueError, message="Expecting ValueError"):
		win.set(window_name='circlecar')


def test_wrong_dimension():
	"""Test if an exception is returned if the dimension is not in the associated array
	"""
	win = signal_xyt.win
	with pytest.raises(ValueError, message="Expecting ValueError"):
		win.set(dims=['z'])

def test_window_plot1d():
	win = signal_xyt.win
	win.set(window_name='lanczos', dims=['time'], n=[12])
	win.plot()


def test_window_plot2d():
	win = signal_xyt.win
	win.set(window_name='lanczos', dims=['y','x'], n=[24, 36])
	win.plot()


def test_compute_boundary_weights():
	win2d = signal_xyt_wth_coast.win
	win2d.set(window_name='tukey', dims=['y', 'x'], n=[24, 36])
	win2d.boundary_weights(drop_dims=['time'])