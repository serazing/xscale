# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# __all__ = ['filtering', 'pca', 'signal', 'plot', 'spectral']

from . import signal
from . import spectral
from .filtering.linearfilters import Window

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
