# --- built in ---
import re
import os
import sys
import time
import logging

from glob import glob

# --- 3rd party ---
import plotly.graph_objects as go

# --- my module ---
from cyaegha import logger

from cyaegha.common.route import Route
from cyaegha.plot.process import BaseProcess
from cyaegha.plot.base import BaseSource

__all__ = []

def _grab_name(path, pattern='(.+)/monitor'):
    '''
    Grab path name
    '''
    p = re.compile(pattern)

    name = p.match(path).group(1)

    return name

def _grab_folders(root, folder_pattern='**/monitor'):
    '''
    Grab folders by giving folder pattern
    '''

    # concat folder path with pattern
    glob_path = os.path.join(root, folder_pattern)
    # using glob to grab folders matched to the pattern
    data_path = glob(glob_path, recursive=True)
    # if 'delete' in the path then ignore it
    data_path = [ p.replace(root+'/', '') for p in data_path if not 'delete' in p]

    return data_path



class Source(BaseSource):
    def __init__(self, name, src=None, data=None, process=None, static=False, slice_inputs=True, **kwargs):
        '''
        Initialize Source

        Args:
            name: (str or None)
            src: (BaseModule)
            data: 
            process: 
            static: 
            slice_inputs:
        '''

        super(Source, self).__init__(name=name, slice_inputs=slice_inputs, **kwargs)

        # draft
        self.draft.src = src
        self.draft.process = process
        
        self._data = data
        self._static = static
        
        # instances
        self._src = None
        self._process = None
        
        
    def _setup_module(self, **kwargs):

        if self.draft.src is not None:
            self._src = Instantiate(self.draft.src).setup(**kwargs)

        if self.draft.process is not None:
            self._process = Instantiate(self.draft.process).setup(**kwargs)


    def _forward_module(self, input, **kwargs):



class StaticSource():
    '''
    StaticSource

    A source that do not change any time.

    If the source is static, which means the contant will not change, it will be loaded only for once, 
    and then the processed results will be cached in the instance. The cached results 
    '''
    pass

class DynamicSource():
    pass


source = Source()

