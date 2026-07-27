"""
MultiChat
# Version: 0.2.2
# Author: Caiwei Zhen
"""

from .Model import utilities, model_training
from .Analysis import Intra_strength as tl
from .Analysis import Processing as pp
from .Analysis import Bg_training as bg
from .Analysis import Joint_embedding as je
from .Analysis import Pipeline as pipeline 
from .Analysis.Pipeline import run_multichat
from .Plot import Visualization as pl

__version__ = "0.2.2"
__author__ = "Caiwei Zhen"
