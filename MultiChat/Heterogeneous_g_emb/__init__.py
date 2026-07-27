"""Heterogeneous Graph Embedding Module of MultiChat"""

from ._settings import settings
from . import preprocessing as pp

import sys
from pathlib import Path

_hge_dir = Path(__file__).resolve().parent
if str(_hge_dir) not in sys.path:
    sys.path.insert(0, str(_hge_dir))
from . import torchbiggraph as _torchbiggraph
sys.modules.setdefault("torchbiggraph", _torchbiggraph)

from . import tools as tl
from . import plotting as pl
from .readwrite import *
from ._version import __version__


import sys
sys.modules.update(
    {f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl']})
