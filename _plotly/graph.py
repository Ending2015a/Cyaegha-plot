# --- built in ---
import os
import abc
import sys
import time
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
        self.config['plot'] = {}

        self.update_subplot(rows=rows, cols=cols, specs=specs)
        self.update_subplot(kwargs)
        


    def plot(self, offline: Optional[bool] =None,
                   **kwargs):
        
        # assert all traces loaded

        assert self.loaded, 'The traces must be loaded'

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
        
    def update_plot(self, arg: dict=None, **kwargs):

        if arg is not None:
            self.config['plot'].update(arg)

        self.config['plot'].update(kwargs)
    

    def load_preset():
        pass

    def dump_preset():
        pass

    # === Sub interfaces ===

    def _forward_object(self, **kwargs):

        return self.plot(**kwargs)
