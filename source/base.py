# --- built in ---
import os
import abc
import sys
import time
import logging

# --- 3rd party ---
import numpy as np

# --- my module ---
from cyaegha import logger

from cyaegha.common.base import BaseModule

from cyaegha.common.route import Route
from cyaegha.common.utils import ParameterPack

__all__ = [
    'BaseSource',
]


class BasePlottingModule(BaseModule):

    def __init__(self, name, slice_inputs=True, **kwargs):

        super(BaseDataFlow, self).__init__(name=name, **kwargs)

        self._slice_inputs = slice_inputs

    def _setup_module(self, **kwargs):
        pass

    def _forward_module(self, input, **kwargs):
        

    def _error_message(self, exception, **kwargs):

    def _forward_process(self, input)

class BaseSource(BaseModule):
    '''
    BaseSource
    '''
    def __init__(self, name, slice_inputs=True, **kwargs):

        super(BaseSource, self).__init__(name=name, **kwargs)

        self._slice_inputs = slice_inputs

    def _setup_module(self, **kwargs):
        pass

    def _forward_module(self, input, **kwargs):
        pass



pipeline = Pipeline(pipeline=(
    Interpolation(),
    BatchAverage(),
    Smoothing(),
))


source = Source(data=['./dqn', './nao', './rnn'], slice_inputs=True, process=SBProcess(), static=True)
source1 = Source(src=source, process=pipeline, static=True)
source2 = Source(src=source, process=pipeline2, static=True)

trace1 = Trace(src=source1, type='interval', legend=name1)
trace2 = Trace(src=source2, type='interval', legend=name2)

graph = (Graph(traces=[trace1, trace2])
            .add_trace(trace1, legend=name1)
            .add_trace(trace2, legend=name2)
            .load_preset(preset=preset))

graph.plot()

