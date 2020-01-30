# --- built in ---
import os
import abc
import sys
import copy
import time
import json
import logging

from typing import Any
from typing import Union
from typing import List
from typing import Tuple
from typing import Callable
from typing import NoReturn
from typing import Optional
from typing import Hashable

# --- 3rd party ---
import numpy as np
import pandas as pd
import scipy.stats as st

import plotly
import plotly.io
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# --- my module ---

from cyaegha import logger

from cyaegha.common.utils import counter
from cyaegha.common.utils import is_array

from cyaegha.plot.base import BaseGraph



__all__ = [
    'Graph'
]

class Graph(BaseGraph):

    # === Properties ===
    @property
    def rows(self) -> int:
        return self.config['subplot/rows']

    @property
    def cols(self) -> int:
        return self.config['subplot/cols']

    # === Main interfaces ===

    def __init__(self, name: str, rows: int=1, cols: int=1, specs=None, **kwargs) -> NoReturn:
        '''
        Create Graph
        
        Kwargs:
            refer to make_subplots: 
                * https://plot.ly/python-api-reference/generated/plotly.subplots.make_subplots.html
        '''

        super(Graph, self).__init__(name=name, rows=rows, cols=cols, **kwargs)

        # save settings
        self.config['subplot'] = {}
        self.config['layout'] = {}
        self.config['traces'] = {}
        self.config['plot'] = {}

        self.update_subplot(rows=rows, cols=cols, specs=specs)
        self.update_subplot(kwargs)
        


    def plot(self, offline: Optional[bool] =None,
                   **kwargs):
        
        # assert all traces loaded

        assert self.loaded, 'The traces must be loaded'

        self.LOG.info('Plotting graph')
        start = time.time()

        # create figure
        fig = make_subplots(**self.config['subplot'])

        # create traces
        trace_subplots = self._generate_traces()

        for trace_subplot in trace_subplots:
            trace, (row, col) = trace_subplot

            fig.add_trace(trace, row=row, col=col)
        
        # if offline:
        #     config = Route(self.config['plot'])
        #     config.update(kwargs)

        #     fig.write_image(**config)

        fig.update_layout(self.config['layout'])

        self.LOG.info('Done in {} secs'.format(time.time() - start))
        return fig

    def update_subplot(self, arg: dict=None, **kwargs):
        '''
        Update subplot configurations
        '''

        reindex = False

        if arg is not None:
            assert isinstance(arg, dict), 'arg must be a dict type'

            self.config['subplot'].update(arg)

            reindex = (('rows' in arg.keys()) or ('cols' in arg.keys()))


        self.config['subplot'].update(kwargs)


        # re-indexing
        if ((reindex) or ('rows' in kwargs) or ('cols' in kwargs)):
            specs = self.config.get('subplot/specs', None)
            
            count = counter()

            # indexing subplot
            for r in range(self.rows):
                for c in range(self.cols):
                    if specs is not None:
                        if specs[r][c] is not None:
                            self._subplot_indices[next(count)] = (r, c)
                    else:
                        self._subplot_indices[next(count)] = (r, c)



    def update_layout(self, arg: dict=None, **kwargs):
        '''
        Update layout configurations

        Args:
            arg: dict-like object
        '''

        if arg is not None:
            self.config['layout'].update(arg)
        
        self.config['layout'].update(kwargs)
        
    def update_traces(self, arg: dict=None, **kwargs):
        '''
        Update trace configurations
        '''

        if arg is not None:
            self.config['traces'].update(arg)

        self.config['traces'].update(kwargs)
    
    def update_plot(self, arg: dict=None, **kwargs):
        '''
        Update plot configurations
        '''

        if arg is not None:
            self.config['plot'].update(arg)

        self.config['plot'].update(kwargs)
    

    def load_preset(self, arg):
        '''
        Args:
            arg: (str) filename, load from file, in json format
                 (dict) dict like object, load from dict
        '''
        if isinstance(arg, str):
            with open(arg, 'r') as f:
                preset = json.load(f)
        elif isinstance(arg, dict):
            preset = dict
        
        self.update_subplot(preset.get('subplot', {}))
        self.update_layout(preset.get('layout', {}))
        self.update_traces(preset.get('traces', {}))
        self.update_plot(preset.get('plot', {}))

    def dump_preset(self, arg=None):
        '''
        Args:
            arg: (None) dump to dict like object
                 (str) dump to filename
        '''

        preset = self.config

        if isinstance(arg, str):
            with open(arg, 'w') as f:
                json.dump(preset, f)
        elif arg is None:
            return copy.deepcopy(preset)

    # === Sub interfaces ===

    def _forward_object(self, **kwargs):

        return self.plot(**kwargs)

    # === Private ===

    def _generate_traces(self):
        '''
        Generate traces
        TODO: dirty code
        return (trace, (row, col))
        '''

        all_traces = []

        for k, ts in self._subplot_traces.items():
            for t in ts:
                
                params = copy.deepcopy(t.params)
                configs = copy.deepcopy(self.config['traces'].get(t.trace.type, {}))
                trace_configs = params.pop('trace_configs', {})

                if is_array(configs) and is_array(trace_configs):
                    assert len(configs) == len(trace_configs), 'configurations must have same length'

                    for idx, (c, tc) in enumerate(zip(configs, trace_configs)):
                        c.update(tc)
                        
                elif is_array(configs):
                    for idx, c in enumerate(configs):
                        c.update(trace_configs)
                elif is_array(trace_configs):
                    c = []
                    for idx, tc in enumerate(trace_configs):
                        c.append( copy.deepcopy(configs).update(trace_configs) )
                else:
                    configs.update(trace_configs)

                traces = t.trace(trace_configs=configs, **params)

                if is_array(traces):
                    for trace in traces:
                        all_traces.append( (trace, k) )
                else:
                    all_traces.append( (traces, k) )

        return all_traces