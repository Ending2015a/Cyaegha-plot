from .base import BaseProcess
from .base import BasePipeline

from .pipeline import Pipeline

from .mid_ware import Process
from .mid_ware import Interpolation
from .mid_ware import BatchAverage
from .mid_ware import Smoothing

__all__ = [
    'BaseProcess',
    'BasePipeline',
    'Pipeline',
    'Process',
    'Interpolation',
    'BatchAverage',
    'Smoothing'
]

try:
    import stable_baselines
    from .sb_process import StableBaselinesProcess

    # alias
    SBProcess = StableBaselinesProcess

    # append to * list
    __all__.append('StableBaselinesProcess')
    __all__.append('SBProcess')
except:
    pass

