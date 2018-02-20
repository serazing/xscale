.. _fitting:


Fitting methods
===============

.. ipython:: python
   :suppress:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import pandas as pd

This section describes different fitting techniques that uses parametric
and non-parametric regression.

.. ipython:: python

   import xscale.signal.fitting as xfit

Parametric methods
------------------


Polynomial fitting
******************

Coming soon


Harmonic fitting
****************

Harmonic fitting is used to capture periodic oscillation in the data when
the oscillation periods :math:`T_k=1/f_k` are known (where :math:`f_k` is the
frequency). The harmonic fitting is thus used to estimate the amplitude
:math:`A_k` and the phase :math:`\Phi_k` of the oscillations.

For real data, the solution of the problem can be written:

.. math::

   y(x, y, t) = c_0 + \sum_1^N A_k(x,y) \sin(2 \pi f_k t + \Phi(x,y)).

For the least-square fit, it is common to use the cosine and sine form:

.. math::

   y(x, y, t) = c_0 + \sum_1^N a_k(x,y) \cos(2 \pi f_k t) + b_k(x,y) \sin(2
   \pi f_k t),

because the first problem is nonconvex and may have multiple minima. The
equivalence between the two form is given by:

.. math::

    A_k = \sqrt{a_k^2 + b_k^2}, \,
    \Phi_k = \arctan\left(\frac{b}{a}\right)

An idealized signal is built with two oscillations at 24 and 12 hours with a
phase that depends on space:

.. ipython:: python

    Nt, Nx, Ny = 100, 128, 128
    rand = xr.DataArray(np.random.rand(Nx, Ny, Nt), dims=['x', 'y', 'time'])
    rand = rand.assign_coords(time=pd.date_range(start='2011-01-01',
                                                 periods=100, freq='H'))
    offset = 0.4
    amp1, phi1 = 1.2, 0.
    wave1 = amp1 * np.sin(2 * np.pi * rand['time.hour'] / 24. +
                          phi1 * np.pi / 180 + 0.05 * rand.x)
    amp2, phi2 = 1.9, 60.
    wave2 = amp2 * np.sin(2 * np.pi * rand['time.hour'] / 12. +
                          phi2 * np.pi / 180. + 0.2 * rand.y)
    truth = offset + rand + wave1 + wave2
    truth = truth.chunk(chunks={'x': 50, 'y': 50, 'time': 20})
    fig, (ax1, ax2) = plt.subplots(2, 1)
    truth.isel(y=0).plot(ax=ax1)
    truth.isel(x=0).plot(ax=ax2)
    plt.tight_layout()
    @savefig sine_truth.png
    plt.show()


The fit is performed by using the :py:func:`xscale.signal.fitting.sinfit`
functions:

.. ipython:: python

   fit2w = xfit.sinfit(truth, dim='time', periods=[24, 12], unit='h')
   print(fit2w)


.. ipython:: python

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fit2w.sel(periods=24)['amplitude'].plot(ax=ax1)
    fit2w.sel(periods=24)['phase'].plot(ax=ax2)
    fit2w.sel(periods=12)['amplitude'].plot(ax=ax3)
    fit2w.sel(periods=12)['phase'].plot(ax=ax4)
    plt.tight_layout()
    @savefig sine_fit.png
    plt.show()

.. ipython:: python

   rec = xfit.sinval(fit2w, truth.time)
   print(rec)

   rec12 = xfit.sinval(fit2w.sel(periods=[12,]), truth.time)
   rec24 = xfit.sinval(fit2w.sel(periods=[24,]), truth.time)

.. ipython:: python

   truth.isel(x=10, y=10).plot(label='Truth')
   rec.isel(x=10, y=10).plot(label='Recontruction with 2 modes')
   rec24.isel(x=10, y=10).plot(label='Recontruction with mode 1, T=24h',
   ls='--')
   rec12.isel(x=10, y=10).plot(label='Recontruction with mode 2, T=12h',
   ls='--')
   plt.legend()
   @savefig sine_reconstruction.png
   plt.show()

.. note::

    For complex signals, the harmonic fitting is not available yet. This
    should be done for future versions


Exponential fitting
*******************

Coming soon

