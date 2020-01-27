# --- built in ---
import os
import abc
import sys
import time
import logging

# --- 3rd party ---
import numpy as np
import pandas as pd

# --- my module ---
from cyaegha import logger

from cyaegha.common.utils import is_array
from cyaegha.common.utils import error_msg

from cyaegha.common.base import BaseModule
from cyaegha.nn.base import BaseNetworkBuilder

__all__ = [
    'BaseProcess',
    'BasePipeline'
]

    

class BaseProcess(BaseModule):
    '''
    BaseProcess
    '''

    def __init__(self, name, slice_inputs=False, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify
            slice_inputs: (bool) whether to slice input list/tuple. If True, the batch (list/tuple) inputs will be sliced into sliced input, 
                each slice will then be fed into the process function.
        '''

        super(BaseProcess, self).__init__(name=name, **kwargs)

        self._slice_inputs = slice_inputs

    def _setup_module(self, **kwargs):
        '''
        Setup module

        override BaseModule._setup_module
        '''
        pass

    def _forward_module(self, input, **kwargs):
        '''
        Forward module

        override BaseModule._forward_module
        '''

        try:
            # check if input can be sliced
            if self._slice_inputs and is_array(input):
                outputs = []
                for data in input:
                    output = self._forward_process(data, **kwargs)
                    outputs.append(output)
            else:
                outputs = self._forward_process(data, **kwargs)
        
        except Exception as e:

            self.LOG.exception(
                error_msg(e, '{ERROR_TYPE}: {msg}: {ERROR_MSG}', msg=self._error_message()))

        return outputs

    def _update_module(self, *args, **kwargs):
        '''
        Update module

        override BaseModule._update_module
        '''
        pass

    @abc.abstractmethod
    def _forward_process(self, input, **kwargs):
        '''
        Forward process
        '''
        return input

    def _error_message(self):
        return 'Failed to run process "{}"'.format(self.name)


    


class BasePipeline(BaseProcess):
    '''
    BasePipeline
    '''

    # === Attributes ===

    network_builder = BaseNetworkBuilder

    # === Properties ===

    @property
    def layers(self):
        '''
        Return each layer instance in the same structure as configuration.
        '''
        return self._layers

    @property
    def outputs(self):
        '''
        Return output from each layer in the same structure as configuration.

        * Note that it is recommanded to cache the result of calling this property since the performance issue.
        '''
        return self._builder.outputs

    @property
    def flattened_layers(self):
        '''
        Return layer instances and stored as a flattened list
        '''
        return self._builder.flattened_layers

    @property
    def flattened_outputs(self):
        '''
        Return outputs from each layer and stored them as a flattened list
        '''
        return self._builder.flattened_outputs

    # === Main interfaces ===

    def __init__(self, name, pipeline, slice_inputs=False, **kwargs):

        super(BasePipeline, self).__init__(name=name, slice_inputs=slice_inputs)

        self.drafts.pipeline = pipeline

        # using network_builder to build the network
        self._builder = self.network_builder(pipeline)

        self._layers = None

    
    # === Sub interfaces ===

    def _setup_module(self, **kwargs):
        '''
        Setup module

        override BaseProcess._setup_module
        '''
        self._builder._setup_network(**kwargs)
        self._layers = self._builder.layers

    def _forward_process(self, input, **kwargs):
        '''
        Forward process

        override BaseProcess._forward_process
        '''
        # forward network
        output = self._builder._forward_network(input, **kwargs)

        return output

    def _update_module(self, *args, **kwargs):
        '''
        Update module

        override BaseProcess._update_module
        '''

        self._builder._update_network(*args, **kwargs)

    # === override SavableObject ===

    def _dump_field(self):
        '''
        Override SavableObject._dump_field
        '''

        raise NotImplementedError("TODO: Method not implemented")

    def _load_field(self):
        '''
        Override SavableObject._load_field
        '''

        raise NotImplementedError("TODO: Method not implemented")

