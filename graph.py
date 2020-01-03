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
    StableBaselinesProcess(),
    Interpolation(),
    BatchAverage(),
    Smoothing()
})

source = StaticSource(sources=[], process=pipeline, slice=True)

group = PlotlyGraphs()
graph = group.new_graph()

trace1 = ConfIntTrace(source)
graph.add_trace(trace1)
graph.add_trace(trace2)


