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
from cyaegha.network.base import BaseConfigurableNetwork

__all__ = [
    'BaseProcess',
    'BasePipeline',
    'Pipeline'
]

    

class BaseProcess(BaseModule):

    def __init__(self, name, unpack_batch=False, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify
            unpack_batch: (bool) whether to unpack input list/tuple. If True, the batch (list/tuple) inputs will be unpacked into sliced input, 
                each slice will then be fed into the process function.
        '''

        super(BaseProcess, self).__init__(name=name, **kwargs)

        self._unpack_batch = unpack_batch

    def _setup_module(self, **kwargs):
        pass

    def _forward_module(self, input, **kwargs):

        try:
            if self._unpack_batch and is_array(input):
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

    def _error_message(self):
        return 'Failed to run process "{}"'.format(self.name)


    @abc.abstractmethod
    def _forward_process(self, input, **kwargs):
        return input


class BasePipeline(BaseConfigurableNetwork):

    def __init__(self, name, pipeline):

        super(BasePipeline, self).__init__(name=name, backbone=None)

        self.drafts.pipeline = pipeline

        # using BaseNetworkBuilder to build the network
        self._builder = self.default_network_builder(pipeline)

        self._layers = None

    # === properties (public) ===

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
        return self._builder.layer_outputs

    # === properties (private) ===

    @property
    def _flattened_layers(self):
        '''
        Return layer instances as a flattened list
        '''
        return self._builder._flattened_layers

    @property
    def _flattened_outputs(self):
        '''
        Return outputs from each layer as a flattened list
        '''
        return self._builder._flattened_outputs

    

    # override BaseNetwork._setup_network
    def _setup_network(self, **kwargs):
        '''
        Setup network
        '''
        self._builder._setup_network(**kwargs)
        self._layers = self._builder.layers


    # override BaseNetwork._forward_network
    def _forward_network(self, input, **kwargs):
        # forward network
        output = self._builder._forward_network(input)

        return output

# aliaing
Pipeline = BasePipeline