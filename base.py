# --- built in ---
import os
import abc
import sys
import time
import inspect
import logging

from typing import Any
from typing import Union
from typing import Tuple
from typing import Callable
from typing import NoReturn
from typing import Optional
from typing import Generator
from typing import cast

# --- 3rd party ---
import numpy as np

# --- my module ---
from cyaegha import logger

from cyaegha.common.draft import is_draft
from cyaegha.common.draft import Key
from cyaegha.common.draft import Draft
from cyaegha.common.draft import Instantiate

from cyaegha.common.forge import argshandler
from cyaegha.common.route import Route

from cyaegha.common.utils import ParameterPack
from cyaegha.common.utils import counter

from cyaegha.common.parallel import parallelizable
from cyaegha.common.parallel import is_parallelizable
from cyaegha.common.parallel import is_unrollable

from cyaegha.plot.source import Source

__all__ = [
    'BaseGraph'
]

class BasePlotObject(metaclass=abc.ABCMeta):

    _already_setup_object: bool = False

    # draft container
    _drafts = ParameterPack()

    # meta info
    # Whether to enable `BasePlotObject._update_step()` for each timestep.
    step_update: bool = True
    # Whether to enable `BasePlotObject._update_epoch()` for each epoch.
    epoch_update: bool = True
    # Whether to enable `BasePlotObject._update_episode()` for each episode.
    episode_update: bool = True

    # === Properties ===

    @property
    def drafts(self) -> ParameterPack:
        '''
        Storing `Draftable` objects
        '''
        return self._drafts

    @property
    def outputs(self) -> Any:
        '''
        Caching outputs
        '''
        return self._cached_outputs

    @property
    def name(self) -> Optional[str]:
        '''
        Return object's name
        '''
        return self._name

    # === Main interfaces ===

    def __init__(self, name: Optional[str], log_level: str ='INFO', **kwargs) -> NoReturn:
        '''
        Initialize Object

        Args:
            name: (str or None) object name or logger name
            log_level: (str) logging level
        '''

        self._name: Optioanl[str] = name

        # create logger
        if name is None: 
            name = '<anony>'
        self.LOG = logger.getLogger(name='{}.{}'.format(type(self).__name__, name), level=log_level)

        # initialize cached outputs
        self._cached_outputs: Any = None

    def __call__(self, *args, **kwargs) -> Any:
        '''
        Main operation

        The results are cached on `outputs` property.
        '''

        self._cached_outputs = self._execute_object(*args, **kwargs)
    
        return self._cached_outputs

    def setup(self, *args, force: bool =True, **kwargs) -> 'BasePlotObject':
        '''
        Setup object or sub-modules

        Args:
            force: (bool) whether to force to setup the object even if it is already set up.

        Returns:
            self
        '''

        # check if already set up
        if (not self._already_setup_object) or (force):
            # setup objects
            self._setup_object(*args, force=force, **kwargs)
            # mark as set up
            self._already_setup_object = True

        return self

    def update(self, invoke_step: bool =False,
                     invoke_epoch: bool =False,
                     invoke_episode: bool =False, **kwargs) -> 'BasePlotObject':
        '''
        Update module. 
        
        Calling update operations.

        Args:
            invoke_step: (bool) whether to call `BaseModule._update_step`, if `BaseModule.step_update` is `True`
            invoke_epoch: (bool) whether to call `BaseModule._update_epoch`, if `BaseModule.epoch_update` is `True`
            invoke_episode: (bool) whether to call `BaseModule._update_episode`, if `BaseModule.episode_update` is `True`

        Kwargs:
            timestep: (int) current timesteps
            epoch: (int) current epoch
            episode: (int) current episode

        Returns:
            self
        '''

        if invoke_step and self.step_update:
            self._update_step(invoke_step=invoke_step, invoke_epoch=invoke_epoch, invoke_episode=invoke_episode, **kwargs)

        if invoke_epoch and self.epoch_update:
            self._update_epoch(invoke_step=invoke_step, invoke_epoch=invoke_epoch, invoke_episode=invoke_episode, **kwargs)
        
        if invoke_episode and self.episode_update:
            self._update_episode(invoke_step=invoke_step, invoke_epoch=invoke_epoch, invoke_episode=invoke_episode, **kwargs)

        return self                 


    # === Sub interfaces ===

    @abc.abstractmethod
    def _setup_object(self, *args, **kwargs):
        '''
        Define your setup instructions (setup sub-modules ...etc) for this object.
        '''
        pass

    @abc.abstractmethod
    def _execute_object(self, *args, **kwargs):
        '''
        Define the forward pass of this module.
        '''
        pass

    @abc.abstractmethod
    def _update_object(self, *args, **kwargs):
        '''
        Define the default update instructions of this module
        '''
        pass

    def _update_step(self, *args, **kwargs):
        '''
        Define the update instructions of this module for the beginning of each timestep. 
        
        In default, this function will invoke `BaseModule._setup_module`
        '''
        self._update_object(*args, **kwargs)

    def _update_epoch(self, *args, **kwargs):
        '''
        Define the update instructions of this module for the beginning of each epoch. 
        
        In default, this function will invoke `BaseModule._setup_module`
        '''
        self._update_object(*args, **kwargs)

    def _update_episode(self, *args, **kwargs):
        '''
        Define the update instructions of this module for the beginning of each episode. 
        
        In default, this function will invoke `BaseModule._setup_module`
        '''
        self._update_object(*args, **kwargs)



class BaseTrace(BasePlotObject):
    '''
    BaseTrace
    '''

    # === Properties ===

    @property
    def loaded(self) -> bool:
        '''
        Whether the data is loaded
        '''
        return (self._already_setup_trace) and (self._source.loaded)

    # === Main interfaces ===

    def __init__(self, name: str, type: Any, source: Source, **kwargs) -> NoReturn:
        '''
        Args:
            type: (Any) trace type
            source: (Source)
        '''

        # initialize BasePlotObject
        super(BaseTrace, self).__init__(name=name, **kwargs)

        assert is_draft(source, Source)

        # draft
        self.drafts.source = source

        # get trace type
        self._trace_type = self._get_type(type)
        self._kwargs = ParameterPack(**kwargs)

        # initialize instances
        self._source = None

    @parallelizable
    def load(self, input: Any =None, force: bool =False, **kwargs) -> bool:
        '''
        Args:
            input: (Any)
        Returns:
            (bool) if is loaded
        '''

        assert self._already_setup_trace, 'The trace is not ready, please call setup(), first.'
        
        if (not self.loaded) or (force):
            self._loaded_source = self._source(input=input, force=force, **kwargs)

        return self.loaded


    # === Sub interfaces ===

    @abc.abstractmethod
    def _execute_object(self, input: Any =None, combine_source=False, **kwargs) -> Any:
        '''
        Args:
            combine_source: (bool) if the loaded source is a list of pandas.Dataframe, 
                combine all of them into a single pandas.Dataframe. Otherwise, create
                a trace for each Dataframe.
        Returns:
            (list of traces)
        '''

        raise NotImplementedError('Method not implemented!')

    def _setup_object(self, key: Key =Draft.default, **kwargs) -> NoReturn:

        # setup source
        if self.drafts.source is not None:
            self._source = Instantiate(self.drafts.source, key=key).setup(key=key, **kwargs)
    

    @load.unroll
    def _load_unroll(self, input: Any =None, force: bool =False, context: Any =None, **kwargs) -> Generator[Callable, None, None]:
        '''
        Args:
            input: (list of Any)
            force: (bool) force forward
            context: (Any) context objects
        '''

        assert self._already_setup_trace, 'The trace is not ready, please call setup(), first.'

        if (not self.loaded) or (force):
            # parallelize if parallelizable
            if is_unrollable(self._source):
                yield from self._source.unrolled(input=input, force=force, context=context, **kwargs)
            else:
                self._loaded_source = self._source(input=input, force=force, **kwargs)

        else:
            return
            yield

    @load.callback
    def _load_callback(self, res: Any, context: Any) -> bool:
        '''
        Callback function
        '''

        if context.fin:
            self._loaded_source = self._source.outputs

        return res

    @abc.abstractmethod
    def _get_type(self, type: Any) -> Any:
        '''
        Get trace type
        '''
        return type


class BaseSubplotHandler(argshandler(sig='self, row, col')):
    __doc__ = '''\
    BaseSubplotHandler

    A wrapper to wrap BaseGraph
    '''
    def __init__(self, graph: BaseGraph, row: int, col: int, **kwargs) -> NoReturn:
        super(BaseSubplotHandler, self).__init__(graph, row, col, **kwargs)
        self.graph = graph

    @property
    def traces(self):
        return self.get_traces()

    
    # served funcs:
    #     def add_trace(self, trace)
    #     def get_traces(self)
    #     def remove_trace(self, trace)
    


class BaseGraph():
    # === built-in methods ===
    def __init__(self, rows: int=1, cols: int=1, **kwargs) -> NoReturn:
        '''
        Create Graph

        rows: (int)
        cols: (int)
        '''

        self._subplot = (rows, cols)
        self._kwargs = ParameterPack(**kwargs)

        self._subplot_traces = {}
        self._subplot_indices = {}

        # create a counter starting from 0, steping 1
        count = counter()

        # indexing subplot
        for r in range(self._subplot[0]):
            for c in range(self._subplot[1]):
                if 'specs' in self._kwargs.keys():
                    if self._kwargs.specs[r][c] is not None:
                        self._subplot_indices[count()] = (r, c)
                else:
                    self._subplot_indices[count()] = (r, c)

    def __call__(self, *args, **kwargs):
        return self.plot()

    def __getitem__(self, key: Union[int, Tuple(int, int)]) -> BaseSubplotHandler:
        '''
        Args:
            key: (int or Tuple(int, int)) index or (row, col)
        '''
        if isinstance(key, int):
            return self.subplot(key)
        elif isinstance(key, tuple):
            return self.subplot(*key)
        else:
            raise ValueError('Unknown arguments: {}'.format(key))

    # === properties ===
    @property
    def rows(self) -> int:
        return self._subplot[0]

    @property
    def cols(self) -> int:
        return self._subplot[1]

    # === functions ===

    @abc.abstractmethod
    def plot(self):
        '''
        Plot graph
        '''
        raise NotImplementedError('Method not implemented')


    def subplot(self, *args) -> BaseSubplotHandler:
        if len(args) == 1:
            row, col = self._subplot_indices[args[0]]
        elif len(args) == 2:
            row, col = *args
        else:
            raise ValueError('Unknown args: {}'.format(args))
        
        return BaseSubplotHandler(self, row, col)

    @BaseSubplotHandler.serve()
    def add_trace(self, trace, row=1, col=1):
        '''
        Add new trace to subplot

        Args:
            trace: (BaseTrace) new trace to add
            row: (int) subplot row
            col: (int) subplot col
        '''

        # assert (row, col)
        assert row <= self._subplot[0] and col <= self._subplot[1]

        # create array
        if (row, col) not in self._subplot_traces.keys():
            self._subplot_traces[(row, col)] = []

        self._subplot_traces.append(trace)

        return self

    @BaseSubplotHandler.serve()
    def get_traces(self, row=1, col=1):
        '''
        Get traces from subplot

        Args:
            row: (int) subplot row
            col: (int) subplot col
        ''' 

        # assert (row, col)
        assert row <= self._subplot[0] and col <= self._subplot[1]

        # create array
        if (row, col) not in self._subplot_traces.keys():
            self._subplot_traces[(row, col)] = []

        return self._subplot_traces[(row, col)]

    @BaseSubplotHandler.serve()
    def remove_trace(self, trace, row=1, col=1):
        '''
        Remove existing trace

        Args:
            trace: (BaseTrace or int) the trace object or index in the order of appended to subplot array
            row: (int) subplot row
            col: (int) subplot col
        '''

        assert row <= self._subplot[0] and col <= self._subplot[1]

        if (row, col) not in self._subplot_traces.keys():
            self._subplot_traces[(row, col)] = []

        if isinstance(trace, int):
            if trace < len(self._subplot_traces[(row, col)]):
                del self._subplot_traces[(row, col)][trace]

        elif isinstance(trace, BaseTrace):
            # remove trace if it contains in the array
            if trace in self._subplot_traces[(row, col)]:
                self._subplot_traces[(row, col)].remove(trace)
            
        return self