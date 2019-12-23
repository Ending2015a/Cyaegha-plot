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
from cyaegha.plot.base import BaseConfigObject
from cyaegha.plot.process import BaseProcess

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



class Source(BaseConfigObject):
    
    def __init__(self, name, pipeline):
        self._name = name

    # === properties ===

    @property
    def name(self):
        return self._name

    @property
    def process(self):
        return self._process

    @process.setter
    def process(self, proc):
        assert isinstance(proc, BaseProcess)
        self._process = proc

    # === override BaseConfigObject ===

    def _dump_field(self):

        d = Route()

    def _load_field(self):
        