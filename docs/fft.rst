Fast Fourier Transform
======================

.. ipython:: python
   :suppress:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt


Estimating using FFT
--------------------

The package `xscale.spectral.fft` provides easy and comprehensive functions
to estimate a multi-dimensional spectrum.


.. note::

   * Only the `zeromean` option is available to detrend the data before FFT
   computation.
   * The `tapering` option is not implemented yet. The FFT computation may
   therefore induce substantial ringing effects.


Multi-tapering estimate
-----------------------

This functionality is not coded yet but it will be available soon.



