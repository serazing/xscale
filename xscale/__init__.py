# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

#__all__ = ['filtering', 'pca', 'signals', 'plot', 'spectral']

from . import filtering
from . import signals

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
