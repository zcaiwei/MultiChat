"""Heterogeneous Graph Embedding Module of MultiChat"""

from ._settings import settings
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from .readwrite import *
from ._version import __version__


import sys
sys.modules.update(
    {f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl']})
