# --- built in ---
import os
import abc
import sys
import time
import logging

# --- 3rd party ---
import numpy as np
import tensorflow as tf


# --- my module ---
from cyaegha.common.base import BaseModule

class Source(BaseModule):
    def __init__(self, name, src=None, data=None, process=None, static=False, slice_inputs=True, **kwargs):
        '''
        Initialize Source

        Args:
            name: (str or None)
            src: (BaseModule) call to get data
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
        self._static = True if static else False
        self._slice_inputs = slice_inputs
        self._source_loaded = False
        
        # instances
        self._src = None
        self._process = None

        

    # === properties ===
    @property
    def static(self):
        return self._static

    @property
    def src(self):
        return self._src
    
    @property
    def process(self):
        return self._process

    # === override BaseModule ===

    def _setup_module(self, **kwargs):
        '''
        override BaseModule._setup_module
        '''

        if self.draft.src is not None:
            self._src = Instantiate(self.draft.src).setup(**kwargs)

        if self.draft.process is not None:
            self._process = Instantiate(self.draft.process).setup(**kwargs)


    def _forward_module(self, input, **kwargs):
        '''
        override BaseModule._forward_module
        '''

        if not self._source_loaded:

            # get input from self._src
            if self._src is not None:
                input = self._src(input, **kwargs)
            # get input from self._data
            elif self._data is not None:
                input = self._data

            # processing data
            if self._slice_inputs:
                output = []
                for d in input:
                    output.append(
                        self._process(d, **kwargs))

            else:
                output = self._process(input, **kwargs)       

            # if static, mark as loaded, else not loaded
            self._source_loaded = self._static

            return output

        else:
            return self.cached_output