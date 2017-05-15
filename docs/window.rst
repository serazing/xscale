.. _window:

Window
======

.. ipython:: python
   :suppress:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt


Description
-----------

The ``Window`` object is implemented in :py:mod:`xscale.window`, which
uses the decorator :py:dec:`xarray.register_dataarray_accessor` to
associate a window to a :py:class:`~xarray.DataArray`. The ``Window`` object
extends :py:class:`xarray.DataArray.rolling` to multi-dimensional arrays and
benefits from the power of :py:mod:`dask.array` for multi-processing
computation. As an example, a 3-dimensional testing ``DataArray`` is loaded
from the :py:mod:`~xscale.signal.generator`:

.. ipython:: python

    import xscale.signal.generator as xgen
    foo = xgen.signaltest_xyt()
    foo

The ``Window`` object may be simply linked to the latter ``DataArray`` using
the attribute ``.window``:

.. ipython:: python

    import xscale
    w = foo.window


Defining
--------

The :py:meth:`xscale.Window.set` method takes optionally six parameters:

 - ``n``: the size of the window
 - ``dim``: dimension names for each axis (e.g., ``('x', 'y', 'z')``).
 - ``dx``: the sampling along the dimensions
 - ``cutoff``: the cutoff of the window, used for defining window for linear
   filtering.
 - ``window``: the name of the window used, and other window parameters passed
   through a tuple
 - ``chunk``: set or modify the chunks of the :py:mod:`dask.array` object
   associated to the :py:mod:`xarray.DataArray`

.. note::

   There is no need to define ``dim`` if the other parameters are passed as
   dictionaries.


.. ipython:: python

    w.set()

Once a ``Window`` is set, one can check the status of the ``Window`` usìng

.. ipython:: python

    print(w)

If the ``cutoff`` parameter is not defined the
:py:meth:`scipy.signal.get_window` is used to build the window along each
dimensions passed through the other parameters.

.. ipython:: python

    w.set(n=15, dim='time', window='boxcar')
    w.plot()
    @savefig boxcar_time_n15.png
    plt.show()

If the ``cutoff`` parameter is defined, the :py:meth:`scipy.signal.get_window`
is used to generate a Finite Impulse Response filter based on the cutoff and
in respect of the window properties:

.. ipython:: python

    cutoff_10d = 10 # A 10-day frequency in seconds
    dx_1d = 1 # Define the sampling period (one day)
    w.set(n=20, dim='time', cutoff=cutoff_10d, dx=dx_1d, window='boxcar')
    w.plot()
    @savefig boxcar_time_n20_10d.png
    plt.show()

.. note::

    Every time one uses the :py:meth:`xscale.Window.set` method, all the
    window parameters are automatically reset.

There are several default options that allow a flexible use of ``Window``. By
 default, if no ``n`` argument is passed, the total length fo the
 corresponding dimensions are taken. This latter option is useful to taper
 the entire data along one dimension with a window.


Plotting
--------

Plotting the window is useful to check its physical and spectral properties.
For 1-dimensional and 2-dimensional windows, the ``plot`` function can be used
to display the weight distribution as well as the spectral response of the
window. The cutoff periods for -3 dB and -6 dB damping and are very useful to
assess the selectivity of the window.

For one-dimensional window:

.. ipython:: python

    w.set(n=15, dim='time', window='hanning')
    w.plot()
    @savefig hanning_time_n15.png
    plt.show()

For two-dimensional window:

.. ipython:: python

    w.set(n={'x': 10, 'y': 15}, window={'x':'hanning', 'y':('tukey', 0.25)})
    w.plot()
    @savefig hanning_nx10_ny15.png
    plt.show()

.. note::

    The ``plot`` function will not work for windows with more than 2 dimensions.


Convolution
-----------

The designed window can be then used to filter the data using a
multi-dimensional convolution by calling the
:py:meth:`xarray.DataArray.Window.convolve` method. When this method is
called the dask graph is implemented by mapping and ghosting the
:py:func:`scipy.ndimage.convolve` function.

.. ipython:: python

    res = w.convolve()
    res.visualize()

.. note::

   If the keyword parameter ``compute`` is set to ``True``, the computation
   will be performed and a progress bar will be displayed.



The convolution is

.. ipython:: python

    foo = xgen.signaltest_xyt()
    w = foo.window
    w.set(n={'x': 11, 'y': 21}, window={'x':'hanning', 'y':('tukey', 0.25)})
    w.convolve(compute=False).visualize()
    @savefig convolve_graph.png


The :py:class:`xarray.DataArray.Window` can be applied on dataset with missing
values such as land areas for oceanographic data. In this case, the filter
weights are normalized to take into account only valid data. In general,
such a normalization is applied by computing the low-passed data :math:`Y_{LP}`:

.. math::

   Y_{LP} = \frac{W * Y}{W * M},


where :math:`Y` is the raw data, :math:`W` the window used, and :math:`M` a
   mask that is 1 for valid data and 0 for missing values.

.. note::

    Once a filtering has been performed, the current ``DataArray`` the
    :py:mod:`dask` graph is destroyed and need to
     be created again using the :py:meth:`xscale.Window.set` method.

Tapering
--------

This functionality is not coded yet but it will be available soon.

