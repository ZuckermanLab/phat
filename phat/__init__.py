"""T Tools for Pathway Histogram Analysis of Trajectories."""

# Add imports here
from .phat import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
