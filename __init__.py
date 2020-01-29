# --- built in ---
import os
import sys
import time
import logging

# --- 3rd party ---
import numpy as np
import pandas as pd

# --- my module ---
from cyaegha import logger

LOG = logger.getLogger('plot', 'INFO')

__all__ = [
    'backend',
    'Trace',
    'Graph',
    'Source'
]


# === import dependencies ===

def prepare():
    # import dependencies to prevent circular import

    from . import process
    from . import source
    from . import base


prepare()
del prepare

# === import source ===

from .source import Source
from .source import StaticSource

# === import object ===

def _load_backend(backend):
    if backend == 'seaborn':
        # seaborn backend
        return _load_seaborn()
        
    elif backend == 'matplotlib':
        # matplotlib backend
        return _load_seaborn()
    else:
        # default using plotly
        return _load_plotly()


def _load_seaborn():

    raise NotImplementedError('Seaborn backend not implemented')


def _load_matplotlib():

    raise NotImplementedError('Matplotlib backend not implemented')


def _load_plotly():
    import plotly

    from cyaegha.plot._plotly import Trace, Graph

    return Trace, Graph
    

# get plotting backend
backend = os.environ.get('CYAEGHA_PLOT_BACKEND', 'plotly')

try:
    Trace, Graph = _load_backend(backend)

    LOG.info('cyaegha.plot backend: {}'.format(backend))

except:
    if backend != 'plotly':
        try:
            Trace = _load_backend('plotly')
            backend = 'plotly'
        except:
            LOG.error('Failed to load cyaegha.plot backend: {}'.format(backend))
            raise
    else:
        LOG.error('Failed to load cyaegha.plot backend: {}'.format(backend))
        raise



