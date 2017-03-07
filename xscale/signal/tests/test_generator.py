# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
import xscale.signal.generator as xgen
import numpy as np
import pytest


def test_ar():
	xgen.ar(0.3, 100, c=0.1)


def test_rednoise():
	xgen.rednoise(0.3, 100, c=0.1)
	with pytest.raises(TypeError, message="Expecting TypeError"):
		xgen.rednoise((0.3, 0.24), 100)


def test_trend():
	x = np.arange(100)
	xgen.trend(x, 1.2, 3.4)


def test_example_xt():
	xgen.example_xt()


@pytest.mark.parametrize("boundaries",  [False, True])
def test_example_xyt(boundaries):
	xgen.example_xyt(boundaries=boundaries)