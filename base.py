# --- built in ---
import os
import abc
import sys
import copy
import time
import inspect
import logging

from typing import Any
from typing import Type
from typing import Union
from typing import Tuple
from typing import Generic
from typing import Callable
from typing import Hashable
from typing import NoReturn
from typing import Optional
from typing import Generator
from typing import cast

from collections import OrderedDict

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
from cyaegha.common.utils import is_array

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

        self._cached_outputs = self._forward_object(*args, **kwargs)
    
        return self._cached_outputs

    def setup(self, *args, force: bool =False, **kwargs) -> 'BasePlotObject':
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
    def _forward_object(self, *args, **kwargs):
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

    The base class of Trace for each plotting backend

    Properties:
        loaded: (bool) whether the data is loaded
        loaded_source: (Any) loaded outputs from source
        type: (Hashable) trace type
        config: (dict) trace configurations

    Main interfaces:
        __init__: 
        __call__:
        load: (Parallelizable, Unrollable) load sources
        unload: unload sources

    Sub interfaces:
        _setup_object: 
        _forward_object: (override)
        _update_object:
    '''

    # === Properties ===

    @property
    def loaded(self) -> bool:
        '''
        Whether the data is loaded
        '''
        if self._source is not None:
            return (self._loaded) and (self._source.loaded)
        else:
            return self._loaded

    @property
    def loaded_source(self) -> Any:
        '''
        Loaded source
        '''
        return self._loaded_source

    @property
    def type(self) -> str:
        '''
        Trace type
        '''
        return self._trace_type

    @property
    def config(self) -> Route:
        '''
        Trace configurations
        '''
        return self._trace_config

    # === Main interfaces ===

    def __init__(self, name: str, type: Hashable, source: Source =None, **kwargs) -> NoReturn:
        '''
        Args:
            type: (Any) trace type
            source: (Source)
        '''

        # initialize BasePlotObject
        super(BaseTrace, self).__init__(name=name, **kwargs)
        # pop out unused args
        kwargs.pop('log_level', None)

        if source is not None:
            assert is_draft(source, Source)

        # draft
        self.drafts.source = source

        # get trace type
        self._trace_type = self.get_typename(type)
        self._trace_config = Route(kwargs)

        # initialize instances
        self._source: Optional[Source] = None
        self._loaded_source: Any = None
        self._loaded: bool = False


    def __call__(self, input: Any =None, force: bool =False, **kwargs) -> Any:
        '''
        Generate traces

        The results are cached on `outputs` property.
        '''

        # load sources
        if (not self._loaded) or (input is not None):

            self.LOG.debug('Loading sources')
            assert self.load(input=input, force=force, **kwargs), 'Failed to load sources'
        
        # generate traces
        self.LOG.debug('Generating traces')
        self._cached_outputs = self._forward_object(input=self.loaded_source, force=force, **kwargs)
    
        return self._cached_outputs

    @parallelizable
    def load(self, input: Any =None, force: bool =False, **kwargs) -> bool:
        '''
        Args:
            input: (Any)
        Returns:
            (bool) if is loaded
        '''

        assert self._already_setup_object, 'The trace is not ready, please call setup() before calling load()'
        
        # load from source
        if self._source is not None:
            self._loaded_source = self._source(input=input, force=force, **kwargs)
        # load from input
        else:
            self._loaded_source = copy.deepcopy(input)
        
        self._loaded = True

        return self.loaded

    def unload(self):
        '''
        Unload sources

        Set loaded flag to False
        '''

        # unload source
        if self._source is not None:
            self._source.unload()

        # unload self
        self._loaded = False


    @classmethod
    @abc.abstractmethod
    def get_typename(cls, obj: Hashable):
        pass

    # === Sub interfaces ===

    def _setup_object(self, key: Key =Draft.default, **kwargs) -> NoReturn:
        '''
        Setup object
        '''

        # setup source
        if self.drafts.source is not None:
            self._source = Instantiate(self.drafts.source, key=key).setup(key=key, **kwargs)

    @abc.abstractmethod
    def _forward_object(self, input: Any =None, combine_source=False, **kwargs) -> Any:
        '''
        Args:
            combine_source: (bool) if the loaded source is a list of pandas.Dataframe, 
                combine all of them into a single pandas.Dataframe. Otherwise, create
                a trace for each Dataframe.
        Returns:
            (list of traces)
        '''

        raise NotImplementedError('Method not implemented!')

    def _update_object(self, *args, **kwargs):
        '''
        Update object
        '''

        if self._source is not None:
            self._source.update(*args, **kwargs)
    
    # === Private ===

    @load.unroll
    def _load_unroll(self, input: Any =None, force: bool =False, context: Any =None, **kwargs) -> Generator[Callable, None, None]:
        '''
        Args:
            input: (list of Any)
            force: (bool) force forward
            context: (Any) context objects

        TODO: return self._loaded
        '''

        assert self._already_setup_object, 'The trace is not ready, please call setup() before calling load()'
        
        # load from source
        if self._source is not None:
            # parallelize
            if is_unrollable(self._source):

                yield from self._source.unrolled(input=input, force=force, context=context, **kwargs)

            # sequential
            else:

                # TODO
                # manually setting context.fin to True
                def _callback(result, context):
                    with context.lock:
                        context.fin = True
                    return result
                
                # code: self._loaded_source = self._source(input=input, force=force, **kwargs)
                yield parallelizable.wrap(self._source, callback=_callback)(input=input, force=force, **kwargs)

                

        # load from input
        else:

            # TODO
            # manually setting context.fin to True
            def _callback(result, context):
                with context.load:
                    context.fin = True
                return result
            
            # code: self._loaded_source = copy.deepcopy(input)
            yield parallelizable.wrap(copy.deepcopy, callback=_callback)(input)


    @load.callback
    def _load_callback(self, res: Any, context: Any) -> bool:
        '''
        Callback function
        '''

        if context.fin: # context.fin attribute inherits from source
            self._loaded_source = self._source.outputs
            self._loaded = True

        return res

    @load.parallel
    def _load_parallel(self, threads: int =1, pool: Any =None) -> Callable:
        '''
        Custom parallel function

        Note, this method only supports thread-based parallelization

        Example:
        >>> trace.load.parallelize(threads=3)(input=input, force=force, **kwargs)
        True

        Args:
            threads: (int) thread count, must greater or equal to 1
            pool: (Any) any pool implementations that has map function.
        '''

        if pool is None:
            assert threads >= 1, 'Thread count must greater or equal to 1'
        
        create_pool = True if pool is None else False

        def _exec(*_args, **_kwargs):
            if create_pool:
                # creating thread pool, mapping
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    # execute all functions
                    list(pool.map(lambda f: f(), self.load.unrolled(*_args, **_kwargs)))
            else:
                # using exists thread pool, mapping
                list(pool.map(lambda f: f(), self.load.unrolled(*_args, **_kwargs)))

            return self.loaded

        return _exec


class _TraceWrapper():

    # === Properties ===

    @property
    def params(self) -> ParameterPack:
        '''
        Return predefined parameters
        '''

        return self._wrapped_params

    @property
    def trace(self):
        '''
        Return wrapped trace
        '''

        return self._wrapped_trace

    # === Main interfaces ===

    def __init__(self, trace, kwargs):
        
        # unwrap redundant TraceWrapper(s)
        self._wrapped_trace = self._unwrap(trace)
        self._wrapped_params = Route(kwargs)

    def __call__(self, *args, **kwargs):
        return self._wrapped_trace(*args, **kwargs)

    # === Private ===

    @classmethod
    def _unwrap(cls, trace):
        '''
        Unwrap TraceWrapper(s)

        The trace can only be wrapped in one layer of TraceWrapper
        '''

        while isinstance(trace, cls):
            trace = trace._wrapped_trace

        return trace



class BaseSubplotHandler(argshandler(sig='self, row, col')):
    __doc__ = '''\
    BaseSubplotHandler

    A wrapper to wrap BaseGraph
    '''
    # === Properties ===

    @property
    def traces(self):
        return self.get_traces()

    # === Main interfaces ===

    def __init__(self, graph: 'BaseGraph', row: int, col: int, **kwargs) -> NoReturn:
        super(BaseSubplotHandler, self).__init__(graph, row, col, **kwargs)
        self.graph = graph
    
    # served funcs:
    #     def add_trace(self, trace)
    #     def get_traces(self)
    #     def remove_trace(self, trace)



class BasePreset():

    # === Attributes ===

    # mapping format name to preset loader
    _loader_map = {} 
    # mapping format name to preset dumper
    _dumper_map = {}
    # mapping alias to format name
    _alias_map = {}

    # === Properties ===

    @property
    @abc.abstractmethod
    def default_format(self):
        '''
        Return default preset format
        '''
        pass

    @property
    @abc.abstractmethod
    def support_formats(self) -> list:
        '''
        Return preset supported formats
        '''
        pass

    # === Main interfaces ===

    @abc.abstractmethod
    def update(self, *args, overwrite=False, **kwargs):
        '''
        Update preset

        Args:
            args[0]: (dict) preset
            overwrite: (bool) whether to overwrite existing properties. If False, apply updates to existing properties.

        Kwargs:
            (preset)
        '''
        pass


    @abc.abstractmethod
    def load(self, arg, *, overwrite=False, format=None, **kwargs):
        '''
        Load preset

        Args:
            arg: (str) filename, load preset from file. 
                 (dict) dict preset, load preset from dict.
            overwrite: (bool) whether to overwrite existing properties. If False, apply updates to existing properties.
            format: (str) file format. If given, the extention of the given filename will be ignored.
        '''
        pass
        
    @abc.abstractmethod
    def dump(self, *args, format=None, **kwargs):
        '''
        Dump preset

        Args:
            args[0]: (str) filename, dump preset to file.
                     (None) dump and return dict object
            format: (str) file format. If given, the extention of the given filename will be ignored.

        Return:
            (dict or None) If arg[0] is None, return (dict), which represents the dumped preset.
                Otherwise, return (None).
        '''
        pass


    @classmethod
    @abc.abstractmethod
    def support(cls, *arg):
        '''
        Return support format

        Args:
            arg[0]: (str) format name, whether support this format. if specified, return (bool)
                    (None) return supported formats (list).

        Returns:
            (bool or list) If arg[0] is None, return (list), which represents the supported format.
                Otherwise, return (bool), representing whether supporting the format specified in arg[0]
        '''
        pass

    @classmethod
    def register(cls, *, format=None, alias=None, loader=None, dumper=None):
        '''
        Register new format
        '''

        assert (format is not None) and (loader is not None) and (dumper is not None)

        cls._register(format, alias, loader, dumper)
            
            

    # === Sub interfaces ===

    @classmethod
    def _register(cls, format: str, alias: Any, loader: Callable, dumper: Callable):

        assert isinstance(format, str), 'format must be a string object'
        assert callable(dumper), 'dumper must be a callable object'
        assert callable(loader), 'loader must be a callable object'

        if format in cls._loader_map.keys():
            raise RuntimeError('The format `{}` is already registered'.format(format))

        cls._loader_map[format] = loader
        cls._dumper_map[format] = dumper

        if alias is not None:

            if is_array(alias):
                for a in alias:
                    # alias is already in used
                    if a in cls._alias_map.keys():
                        raise RuntimeError('The alias `{}` for format `{}` is already used by format `{}`'.format(a, format, cls._alias_map[a]))
                    
                    cls._alias_map[a] = format
            else:

                # alias is already in used
                if alias in cls._alias_map.keys():
                    raise RuntimeError('The alias `{}` for format `{}` is already used by format `{}`'.format(alias, format, cls._alias_map[alias]))

                cls._alias_map[alias] = format

    @classmethod
    def _get_format_list(cls):
        '''
        Return the list of valid format names
        '''
        return list(cls._loader_map.keys())

    @classmethod
    def _get_format(cls, obj: Hashable):
        '''
        Return the format name by giving the format name or its alias
        '''
        if obj in cls._alias_map.keys(): 
            obj = cls._alias_map[obj]
        elif obj not in cls._loader_map.keys():
            raise ValueError('Unknown format name: {}'.format(obj))

        return obj

    @classmethod
    def _has_format(cls, obj: Hashable):
        '''
        Return if the given format is registered
        '''
        return (obj in cls._alias_map.keys()) or (obj in cls._loader_map.keys())

    @classmethod
    def _get_loader(cls, obj: Hashable):
        '''
        Return the preset loader by specifing the format name or its alias
        '''
        name = cls._get_format(obj)

        return self._loader_map[name]
    
    @classmethod
    def _get_dumper(cls, obj: Hashable):
        '''
        Return the preset dumper by specifing the format name or its alias
        '''
        name = cls._get_format(obj)

        return self._dumper_map[name]



class BaseGraph(BasePlotObject):
    '''
    BaseGraph

    The base class of cyaegha.plot Graph

    Attributes:
        _Preset_class: The class of the preset of the graph. If you are inheriting
            the BaseGraph, do not forget to change this attribute to your custom preset.

    Properties:
        rows: (int) rows of the subplots
        cols: (int) cols of the subplots
        loaded: (bool) whether all of the traces are loaded
        preset: (BasePreset) graph preset
    
    Main interfaces:
        __init__:
        __call__:
        __getitem__:
        load:
        unload:
    '''

    # === Attributes ===
    _Preset_class: Type[BasePreset] = BasePreset

    # === Properties ===
    @property
    def rows(self) -> int:
        return self._subplot[0]

    @property
    def cols(self) -> int:
        return self._subplot[1]

    @property
    def loaded(self) -> bool:
        '''
        Whether all traces are set up
        '''
        return all([t.loaded for t in self._flatten_traces()])

    @property
    def preset(self) -> Type[BasePreset]:
        '''
        Return prestes
        '''
        return self._graph_preset

    # === Main interfaces ===

    def __init__(self, name: str, rows: int=1, cols: int=1, **kwargs) -> NoReturn:
        '''
        Create Graph

        rows: (int)
        cols: (int)
        '''

        super(BaseGraph, self).__init__(name=name, **kwargs)

        # initialize
        self._subplot = (rows, cols)
        self._graph_preset = _Preset_class()

        self._subplot_traces = OrderedDict()
        self._subplot_indices = {}

        # generate subplot indices
        self._generate_subplot_indices()


    def __call__(self, input: Any =None, force: bool =False, **kwargs) -> Any:
        '''
        Create graph

        The results are cached on `outputs` property.
        '''

        if (not self.loaded) or (force):

            assert self.load(input=input, force=force, **kwargs), 'Failed to load traces'

        self._cached_outputs = self._forward_object(input=input, force=force, **kwargs)
    
        return self._cached_outputs

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Type[BaseSubplotHandler]:
        '''
        Get SubplotHandler

        Args:
            key: (int or Tuple(int, int)) index or (row, col)
        '''
        if isinstance(key, int):
            return self.subplot(key)
        elif isinstance(key, tuple):
            return self.subplot(*key)
        else:
            raise ValueError('Unknown arguments: {}'.format(key))

    @parallelizable
    def load(self, input: Any =None, force: bool =False, **kwargs) -> bool:
        '''
        Load traces
        '''
        
        return all([t.load(input=input, force=force, **kwargs) 
                            for t in self._flatten_traces()])

    def unload(self):
        '''
        Unload traces
        '''
        for t in self._flatten_traces():
            t.unload()

    @abc.abstractmethod
    def plot(self, *args, **kwargs):
        '''
        Plot graph
        '''
        raise NotImplementedError('Method not implemented')


    def subplot(self, *args) -> BaseSubplotHandler:
        if len(args) == 1:
            row, col = self._subplot_indices[args[0]]
        elif len(args) == 2:
            row, col, *_ = args
        else:
            raise ValueError('Unknown arguments: {}'.format(args))
        
        return BaseSubplotHandler(self, row, col)

    @BaseSubplotHandler.serve()
    def add_trace(self, trace, row=1, col=1, ignore=True, **kwargs):
        '''
        Add a new trace to the subplot

        Args:
            trace: (BaseTrace) new trace to add
            row: (int) subplot row
            col: (int) subplot col
            ignore: (bool) whether to ignore trace exists warning
        
        Kwargs:
            (parameters for generating traces) these parameters are used at calling trace.__call__()

        Returns:
            (self)
        '''

        # assert (row, col)
        assert (row <= self.rows) and (col <= self.cols)

        # create mapping (trace, params)
        if (row, col) not in self._subplot_traces.keys():
            self._subplot_traces[(row, col)] = OrderedDict()

        # print exists warning
        if not ignore:
            if trace in self._subplot_traces[(row, col)]:
                self.LOG.warning('The trace {} for subplot ({}, {}) already exists'.format(trace.name, row, col))
        
        # add to the subplot a mapping of (trace, params)
        self._subplot_traces[(row, col)][trace] = kwargs

        return self


    @BaseSubplotHandler.serve()
    def get_traces(self, row=1, col=1):
        '''
        Get traces from subplot

        Args:
            row: (int) subplot row
            col: (int) subplot col

        Returns:
            (list of traces)
        ''' 

        # assert (row, col)
        assert row <= self.rows and col <= self.cols

        # create mapping (trace, params)
        if (row, col) not in self._subplot_traces.keys():
            self._subplot_traces[(row, col)] = OrderedDict()

        # return traces
        return [t for t, p in self._subplot_traces[(row, col)].items()]


    @BaseSubplotHandler.serve()
    def remove_trace(self, trace, row=1, col=1, ignore=True):
        '''
        Remove existing trace from the subplot

        Args:
            trace: (BaseTrace or int) the trace object or index in the order of appended to subplot array
            row: (int) subplot row
            col: (int) subplot col
            ignore: (bool) whether to ignore trace not exists warning

        Returns:
            (self)
        '''

        # assert (row, col)
        assert (row <= self.rows) and (col <= self.cols)

        # create mapping (trace, params)
        if (row, col) not in self._subplot_traces.keys():
            self._subplot_traces[(row, col)] = OrderedDict()

        # specifing the trace to remove by the trace index
        if isinstance(trace, int):
            
            # check if the i-th trace exists in the mapping
            if trace < len(self._subplot_traces[(row, col)]):
                # get trace by index
                trace = list(self._subplot_traces[(row, col)].keys())[trace]
                # delete trace
                del self._subplot_traces[(row, col)][trace]

        # specifing the trace to remove by the trace instance
        elif isinstance(trace, BaseTrace):

            # remove the trace from the subplot if it exists in the mapping
            if trace in self._subplot_traces[(row, col)]:
                del self._subplot_traces[(row, col)][trace]
            
        return self

    @abc.abstractmethod
    def load_preset(self, *args, **kwargs):
        '''
        Load preset 
        '''
        pass

    @abc.abstractmethod
    def dump_preset(self, *args, **kwargs):
        '''
        Dump preset
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def from_preset(self, *args, **kwargs):
        '''
        Create graph from preset
        '''
        pass

    # === Sub interfaces ===

    def _setup_object(self, *args, **kwargs):
        '''
        Setup traces

        #TODO: the original `setup` function would protect the second call to this function,
            unless the users assign `force=True` when calling the function. The question is:
            is it better to remove such the protextion to let users can call `setup` many times?
        '''

        for trace in self._flatten_traces():
            trace.setup(*args, **kwargs)

    @abc.abstractmethod
    def _forward_object(self, *args, **kwargs):
        '''
        Load traces and plot graph
        
        1. call load()
        2. call plot()
        '''
        pass

    def _update_object(self, *args, **kwargs):
        '''
        Update traces
        '''

        for trace in self._flatten_traces():
            trace.update(*args, **kwargs)

    def _flatten_traces(self) -> list:
        '''
        Flatten traces list

        Returns:
            (list of traces)

        NOTE: This function returns a list of traces. If you need the trace with its params, 
            please use self._flatten_traces_with_params(), instead.
        '''

        traces = []

        for ts in self._subplot_traces.values():
            traces.extend([t for t in ts.keys()])

        return traces

    def _flatten_wrapped_traces(self) -> list:
        '''
        Flatten traces list with its params

        Returns:
            (list of tuples (trace, params))

        NOTE: This function returns a list of traces with its params. Each list element is a 
            tuple of (trace, params). If you only need the traces, please use self._flatten_traces(), instead.
        '''

        traces = []

        for k, ts in self._subplot_traces.items():
            traces.extend( list(ts.items()) )

        return traces

    def _generate_subplot_indices(self):
        '''
        Generate index for each subplot so as to access the subplot by its index
        '''
        # create a counter starting from 0, steping 1
        count = counter()

        # indexing subplot
        for r in range(self.rows):
            for c in range(self.cols):
                self._subplot_indices[next(count)] = (r, c)


    def _generate_traces(self):
        '''
        Generate traces
        TODO

        return (trace, (row, col))
        '''

        all_traces = []

        for k, ts in self._subplot_traces.items():
            for t in ts:
                traces = t.trace(**t.params)

                if is_array(traces):
                    for trace in traces:
                        all_traces.append( (trace, k) )
                else:
                    all_traces.append( (traces, k) )

        return all_traces

    # === Private ===

    @load.parallel
    def _load_parallel(self, threads: int =1, pool: Any =None) -> Callable:
        '''
        Load parallel

        NOTE: Currently only support parallelizing in a single Trace
        NOTE: thread-based 
        '''
        
        if pool is None:
            assert threads >= 1, 'Thread count must greater or equal to 1'

        create_pool = True if pool is None else False

        traces = self._flatten_traces()

        def _exec(*_args, **_kwargs):
            
            results = []
            if create_pool:
                # create thread pool
                with ThreadPoolExecutor(max_workers=threads) as _pool:
                    for trace in traces:
                        # if trace.load can be paralellized
                        if is_parallelizable(trace.load):
                            results.append( trace.load.parallelize(pool=_pool)(*_args, **_kwargs) )
                        # sequential call
                        else:
                            results.append( trace.load(*_args, **_kwargs) )
            else:

                for trace in traces:
                    # if trace.load can be paralellized
                    if is_parallelizable(trace.load):
                        results.append( trace.load.parallelize(pool=pool)(*_args, **_kwargs) )
                    # sequential call
                    else:
                        results.append( trace.laod(*_args, **_kwargs))


            return all(results)

        return _exec

        
