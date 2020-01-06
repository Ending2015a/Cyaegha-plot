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

from cyaegha.nn.builder import CyaeghaDagNetworkBuilder
from cyaegha.plot.process.base import BasePipeline

__all__ = [
    'CyaeghaDagPipeline',
    'Pipeline'
]

class CyaeghaDagPipeline(BasePipeline):
    
    network_builder = CyaeghaDagNetworkBuilder

# alias
Pipeline = CyaeghaDagPipeline