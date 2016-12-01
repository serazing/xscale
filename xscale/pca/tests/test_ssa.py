import pytest
import dask.array as da


def test_compute_ssa_covmat():
	array = da.random((n, m))