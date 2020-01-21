# --- built in ---
import os
import abc
import sys
import time
import logging

# --- 3rd party ---
import numpy as numpy

# loading plotting package
backend = os.environ('CYAEGHA_PLOT_BACKEND', 'plotly')

try:
    Trace, Graph = _load_backend(backend)
except:
    if backend != 'plotly':
        backend = 'plotly'
        Trace, Graph = _load_backend(backend)

# --- my module ---
from cyaegha import logger


__all__ = [
    'backend',
    'Trace',
    'Graph'
]


LOG = logger.getLogger('plot', 'INFO')


def _load_backend(backend):
    if backend == 'seaborn':
        # seaborn backend
        _load_seaborn()
        
    elif backend == 'matplotlib':
        # matplotlib backend
        _load_seaborn()
    else:
        # default using plotly
        _load_plotly()


def _load_plotly():
    import plotly

    from cyaegha.plot._plotly import Trace, Graph

    return Trace, Graph

def _load_seaborn():

    raise NotImplementedError('Seaborn backend not implemented')

def _load_matplotlib():

    raise NotImplementedError('Matplotlib backend not implemented')
    




