from .globals import VERSION

__version__ = VERSION


# deprecation warning for analyses
from . import analyze
from warnings import warn

def __getattr__(name):
    if name == 'analyses':
        warn("the module `analyses` has been renamed to `analyze`")
        return analyze
    raise AttributeError('No module named ' + name)