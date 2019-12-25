# --- built in ---
import os
import sys
import time
import logging
import importlib

# --- 3rd party ---
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- my module ---
from cyaegha import logger

from cyaegha.common.route import Route

from cyaegha.plot.base import BaseConfigObject


class Graph(BaseConfigObject):
    def __init__(self, name, **kwargs):

    def add_trace(self):

    def add_source(self):

    def _dump_field(self):
        pass

    def _load_field(self):
        pass



pipeline = Pipeline({
    Interpolation(),
    BatchAverage(),
    Smoothing()
})

Source(pipeline)
