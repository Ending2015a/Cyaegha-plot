# --- built in ---
import os
import abc
import sys
import copy
import time
import json
import logging

from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import NoReturn
from typing import Optional
from typing import Hashable

from collections import Mapping

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
from cyaegha.plot.base import BasePreset


__all__ = [
    'Graph'
]

LOG = logger.getLogger('cyaegha.graph', 'INFO')

class Preset(BasePreset):

    # === Properties ===
    
    def default_format(self) -> str:
        '''
        Return default preset format
        '''
        return 'json'
    
    @property
    def support_formats(self) -> List[str]:
        '''
        Return preset supported format
        '''
        return self._get_format_list()

    # === Main interfaces ===

    def __init__(self):
        self._preset = Route()

    def __getitem__(self, keys: Hashable) -> Any:
        
        return self._preset[keys]

    def __setitem__(self, keys: Hashable, value: Any) -> NoReturn:

        self._preset[keys] = value

    def __delitem__(self, keys: Hashable) -> NoReturn:

        del self._preset[keys]

    def get(self, keys: Hashable, default: Any =None) -> Any:

        return self._preset.get(keys, default)

    def pop(self, keys: Hashable, default: Any =None) -> Any:

        return self._preset.pop(keys, default)

    def update(self, *args, overwrite=False, **kwargs) -> __qualname__:
        '''
        Update preset

        Args:
            args[0]: (dict) preset
            overwrite: (bool) whether to overwrite existing properties. 
                If False, apply updates to existing properties

        Kwargs:
            (preset)

        Returns:
            (self)
        '''

        if len(args) > 0:
            assert isinstance(args[0], Mapping), 'The first argument must be a dict like object'

        if overwrite:
            self._preset.update(*args, **kwargs)
        
        else:
            # TODO: if list in the Route, this feature will break.

            args = Route(*args).plain()
            kwargs = Route(**kwargs).plain()

            for k, v in args.items():
                self._preset[k] = v

            for k , v in kwargs.items():
                self._preset[k] = v

        return self

    def load(self, arg, *, overwrite=True, format=None, **kwargs) -> __qualname__:
        '''
        Load preset

        Args:
            arg: (str) filename, load preset from file. 
                 (dict) dict preset, load preset from dict.
            overwrite: (bool) whether to overwrite existing properties. If False, apply updates to existing properties.
            format: (str) file format. If given, the extention of the given filename will be ignored.
        
        Kwargs:
            (kwargs for loader)
        
        Returns:
            (self)
        '''

        if isinstance(arg, str):
            # get format
            if format is None:
                _, ext = os.path.splitext(arg)
                format = self._get_format(ext)
            
            loader = self._get_loader(format)

            # load preset
            loaded_preset = loader(arg, **kwargs)

            # update current preset
            self.update(loaded_preset, overwrite=overwrite)

        elif isinstance(arg, Mapping):
            self.update(arg, overwrite=overwrite)

        else:
            raise TypeError('The first argument must be a `str` or `dict`, got {}'.format(type(arg)))

        return self

    def dump(self, *args, format=None, **kwargs):
        '''
        Dump preset

        Args:
            args[0]: (str) filename, dump preset to file.
                     (None) dump and return dict object
            format: (str or None) file format. If given, the extention of the given filename will be ignored.

        Kwargs:
            (kwargs for dumper)

        Return:
            (dict or None) If arg[0] is None, return (dict), which represents the dumped preset.
                Otherwise, return (None).
        '''

        if len(args) > 0:
            arg = args[0]
        else:
            arg = None

        # dump to file
        if isinstance(arg, str):
            # get format
            if format is None:
                _, ext = os.path.splitext(arg)
                format = self._get_format(ext)
            
            dumper = self._get_dumper(format)

            # dump preset
            dumper(self._preset, arg, **kwargs)

        elif arg is None:
            # dump to base dict
            return copy.deepcopy(self._preset.to_base())

        else:
            raise TypeError('The first argument must be a `str` or `None`, got {}'.format(type(arg)))


    @classmethod
    def support(cls, *args):
        '''
        Return support format

        Args:
            arg[0]: (str) format name, whether support this format. if specified, return (bool)
                    (None) return supported formats (list).

        Returns:
            (bool or list) If arg[0] is None, return (list), which represents the supported format.
                Otherwise, return (bool), representing whether supporting the format specified in arg[0]
        '''

        if len(args) > 0:
            arg = args[0]
        else:
            arg = None

        if arg is None:
            return cls._get_format_list()
        else:
            return cls._has_format(arg)


# === json support ===
# suppot json format
def _load_json(self, file, **kwargs):
    '''
    Load object from json
    '''
    try:
        with open(file, 'r') as f:
            return json.load(file, **kwargs)

    except json.JSONDecodeError as e:
        # failed to load json
        self.LOG.error('Failed to load preset from: {}'.format(file))

def _dump_json(self, data, file, **kwargs):
    '''
    Dump object to json
    '''
    try:
        with open(file, 'w') as f:
            json.dump(data, f, **kwargs)
    
    except TypeError as e:
        self.LOG.error('Failed to dump preset to: {}'.format(file))

Preset.register(format='json', alias=['JSON', '.json'], _load_json, _dump_json)


# === yaml support ===
# trying to supporting yaml format
try:
    import yaml
    
    def _load_yaml(self, file, **kwargs):
        '''
        Load object from yaml
        '''
        try:
            with open(file, 'r') as f:
                return yaml.safe_load(f, **kwargs)

        except yaml.YAMLError as e:
            # failed to load yaml
            self.LOG.error('Failed to load preset from: {}'.format(file))

        return None

    def _dump_yaml(self, data, file, **kwargs):
        '''
        Dump object to file
        '''
        try:
            with open(file, 'w') as f:
                yaml.dump(data, f, **kwargs)

        except yaml.YAMLError as e:
            # failed to dump yaml
            self.LOG.error('Failed to dump preset to: {}'.format(file))


    Preset.register(format='yaml', alias=['YAML', 'YML', 'yml', '.yaml', 'yml'], _load_yaml, _dump_yaml)

except ImportError:
    
    LOG.warning('{} does not support yaml format.'.format(Preset.__qualname__))
    LOG.warning('Please install PyYAML for supporting yaml.')



class Graph(BaseGraph):

    # === Properties ===
    @property
    def rows(self) -> int:
        return self.preset['subplot/rows']

    @property
    def cols(self) -> int:
        return self.preset['subplot/cols']

    # === Main interfaces ===

    def __init__(self, name: str, rows: int=1, cols: int=1, specs=None, **kwargs) -> NoReturn:
        '''
        Create Graph
        
        Kwargs:
            refer to make_subplots: 
                * https://plot.ly/python-api-reference/generated/plotly.subplots.make_subplots.html
        '''

        super(Graph, self).__init__(name=name, rows=rows, cols=cols, **kwargs)

        # update preset
        self.preset.update(overwrite=True, subplot=dict(rows=rows, cols=cols, specs=specs),
                                           layout=dict(),
                                           traces=dict(),
                                           plot=dict())

        self.preset.update(subplot=kwargs)

        # generate indices
        self._generate_subplot_indices()
        


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


    def load_preset(self, arg, *, overwrite=True, format=None, **kwargs):
        '''
        Load preset

        Args:
            arg: (str) filename, load preset from file. 
                 (dict) dict preset, load preset from dict.
            overwrite: (bool) whether to overwrite existing properties. If False, apply updates to existing properties.
            format: (str) file format. If given, the extention of the given filename will be ignored.
        
        Kwargs:
            (kwargs for loader)
        
        Returns:
            (self)
        '''
        
        self.preset.load(arg, overwrite=overwrite, format=format, **kwargs)

        self._generate_subplot_indices()

        return self

    def dump_preset(self, *args, format=None, **kwargs):
        '''
        Dump preset

        Args:
            args[0]: (str) filename, dump preset to file.
                     (None) dump and return dict object
            format: (str or None) file format. If given, the extention of the given filename will be ignored.

        Kwargs:
            (kwargs for dumper)

        Return:
            (dict or None) If arg[0] is None, return (dict), which represents the dumped preset.
                Otherwise, return (None).
        '''

        return self.preset.dump(*args, format=format, **kwargs)

    @classmethod
    def from_preset(cls, arg, *, format=None, **kwargs):
        '''
        Create graph from preset

        Args:
            arg: (str) filename, load preset from file. 
                 (dict) dict preset, load preset from dict.
            format: (str) file format. If given, the extention of the given filename will be ignored.
        
        Kwargs:
            (kwargs for loader)
        
        Returns:
            (self)
        '''

        #TODO: name
        self = cls(name='<anony>')

        return self.load_preset(arg, format=format, **kwargs)

    # === Sub interfaces ===

    def _forward_object(self, **kwargs):

        return self.plot(**kwargs)


    def _generate_subplot_indices(self):

        specs = self.preset.get('subplot/specs', None)
            
        count = counter()

        # indexing subplot
        for r in range(self.rows):
            for c in range(self.cols):
                if specs is not None:
                    if specs[r][c] is not None:
                        self._subplot_indices[next(count)] = (r, c)
                else:
                    self._subplot_indices[next(count)] = (r, c)

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