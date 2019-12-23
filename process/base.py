# --- built in ---
import os
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
    
    def __init__(self, **kwargs):
        super(BaseProcess, self).__init__(**kwargs)

    


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