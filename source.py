# --- built in ---
import os
import abc
import sys
import copy
import time
import types
import logging
import threading

from typing import Any
from typing import Union
from typing import NoReturn
from typing import Optional
from typing import Callable
from typing import Generator

# --- 3rd party ---
import numpy as np
import tensorflow as tf

# --- my module ---
from cyaegha import logger 

from cyaegha.common.base import BaseModule
from cyaegha.common.draft import Key
from cyaegha.common.draft import Draft
from cyaegha.common.draft import Instantiate
from cyaegha.common.draft import is_draft
from cyaegha.common.utils import is_array

from cyaegha.common.parallel import parallelizable
from cyaegha.common.parallel import is_parallelizable
from cyaegha.common.parallel import is_unrollable

from cyaegha.plot.process import BaseProcess


__all__ = [
    'Source', 
    'StaticSource'
]

class Source(BaseModule):

    # === Attributes ===

    # mark as parallelizable
    __parallelizable__ = True
    # mark as unrollable
    __unrollable__ = True

    # === properties ===
    @property
    def static(self) -> bool:
        '''
        Whether the source is a static source (only load for once)
        '''
        return self._static

    @property
    def src(self) -> Optional[BaseModule]:
        '''
        Source instance
        '''
        return self._src
    
    @property
    def process(self) -> Optional[BaseProcess]:
        '''
        Process instance
        '''
        return self._process

    @property
    def loaded(self) -> bool:
        '''
        Whether the source is loaded
        '''
        return self._loaded

    # === Main interfaces ===

    def __init__(self, name: Optional[str], 
                        src: Optional[BaseModule] =None, 
                       data: Any =None, 
                    process: Optional[BaseProcess] =None, 
                     static: bool =False, 
               slice_inputs: bool =True, **kwargs) -> NoReturn:
        '''
        Initialize Source

        Args:
            name: (str or None)
            src: (BaseModule) callable data
            data: (list)
            process: (BaseProcess)
            static: (bool) Whether this is a static source. If True, the source would be loaded only for once. 
                However, you can still reload data by assign force=True when calling Source.__call__
            slice_inputs: (bool) Whether to slice inputs before applying to the process module. BTW, the process
                of sliced inputs can be accelerated by calling Source.parallelize

        Kwargs:
            log_level: (str, default: 'INFO') loggin level
        '''

        super(Source, self).__init__(name=name, **kwargs)

        if src is not None:
            assert is_draft(src, BaseModule)

        if process is not None:
            assert is_draft(process, BaseProcess)

        # draft
        self.drafts.src = src
        self.drafts.process = process
        
        # initialize
        self._data: Any = data
        self._static: bool = True if static else False
        self._slice_inputs: bool = True if slice_inputs else False
        self._loaded: bool = False
        
        # initialize instances
        self._src = None
        self._process = None

    @parallelizable
    def __call__(self, input: Any =None, force: bool =False, **kwargs):
        '''
        Load source

        Args:
            input: (Any) input data
            force: (bool) force forward

        Example:
        * Static source (static=True)
        >>> source = Instantiate(Source(static=True)).setup()
        >>> output = source(input) # loaded
        >>> output2 = source(input2) # blocked
        >>> output2 = source(input2, force=True) #loaded

        # Dynamic source (static=False)
        >>> source = Instantiate(Source(static=False)).setup()
        >>> output = source(input) # loaded
        >>> output2 = source(input2) # loaded
        '''

        if (not self.static) or (not self._loaded) or (force):
            # forward module
            self._cached_outputs = self._forward_module(input=input, force=force, **kwargs)
            # mark as loaded
            self._loaded = True
        else:
            # user is trying to reload static source but force=False
            if (self.static) and (input is not None):
                self.LOG.warning('trying to reload static source, set force=True to reload')

        return self._cached_outputs


    def unload(self):
        '''
        Unload source

        Set loaded flag to False
        '''

        self._loaded = False

    # === Sub interfaces ===

    def _setup_module(self, key: Key =Draft.default, **kwargs) -> NoReturn:
        '''
        override BaseModule._setup_module
        '''

        if self.drafts.src is not None:
            self._src = Instantiate(self.drafts.src, key=key).setup(key=key, **kwargs)

        if self.drafts.process is not None:
            self._process = Instantiate(self.drafts.process, key=key).setup(key=key, **kwargs)

    @parallelizable
    def _forward_module(self, input: Any =None, **kwargs) -> Any:
        '''
        override BaseModule._forward_module

        Input priority: src > input > data

        # TODO support for process-based parallelism
        Currently only support thread-based prallelism. It is unsafe to use proecssed_based parallelization.
        '''


        # Input priority: input > src > data
        # get input from self._src
        if self._src is not None:
            input = self._src(input, **kwargs)

        # get input from self._data
        if self._data is not None:
            input = input or self._data
        
        if self._process is not None:

            if self._slice_inputs and is_array(input):
                # slice inputs
                output = [self._process(input=d, **kwargs) for d in input]
            else:
                output = self._process(input=input, **kwargs)
        # no process specified
        else:
            output = input

        return output

    def _update_module(self, *args, **kwargs):
        '''
        override BaseModule._update_module
        '''

        if self._src is not None:
            self._src.update(*args, **kwargs)

        if self._process is not None:
            self._process.update(*args, **kwargs)


    @__call__.unroll
    def _call_unroll(self, input: Any =None, force: bool =False, context: Any =None, **kwargs) -> Generator[Callable, None, None]:
        '''
        Parallelize __call__

        NOTE: each coroutine returns an object `result`

        Args:
            input: (Any) input data
            force: (bool) force call
            context: (Any) context objects

        Returns:
            (Generator)
        '''

        if (not self.static) or (not self._loaded) or (force):
            yield from self._forward_module.unrolled(input=input, force=force, context=context, **kwargs)

        else:
            # user is trying to reload static source but force=False
            if (self.static) and (input is not None):
                self.LOG.warning('trying to reload static source, set force=True to reload')

            # return empty generator (yield nothing)
            return
            yield

    @__call__.callback
    def _call_callback(self, res: Any, context: Any, **kwargs) -> Any:
        '''
        Return post-processed result

        Args:
            res: (Any) result from _forward_module_callback()
            context: (Any) context object

        Returns:
            (Any) result
        '''

        if context.fin:
            # set outputs
            self._cached_outputs = context.outputs
            # set loaded to True
            self._loaded = True
        
        return res

    @_forward_module.unroll
    def _forward_module_unroll(self, input: Any =None, context: Any=None, **kwargs) -> Generator[Callable, None, None]:
        '''
        Parallelize _forward_module

        NOTE: each coroutine returns (index, result) tuple

        Args:
            input: (Any) input data
            context: (Any) context objects

        Returns:
            (Generator)
        '''

        # Input priority: input > src > data
        # get input from self._src
        if self._src is not None:
            # TODO: async nested parallelism
            # if is_parallelizable(self._src):
            #     # nested parallelism
            #     yield from self._src.unrolled(input, **kwargs)
            #     input = self._src.outputs
            # else:
            
            input = self._src(input, **kwargs)

        # get input from self._data
        if self._data is not None:
            input = input or self._data

        # if input can be sliced
        slice_inputs = self._slice_inputs and is_array(input)

        # initialize context
        context.sliced = slice_inputs
        context.total_completed = 0
        context.outputs = []
        context.fin = False

        if self._process is not None:

            if slice_inputs:
                
                # initialize context
                context.total_coroutines = len(input)
                
                # create coroutines for each slice
                for idx, d in enumerate(input):
                    def _coroutine_factory(index, process, input, kwargs):

                        def _coroutine(*_args, **_kwargs):
                            result = process(input, **kwargs)
                            return (index, result)

                        return _coroutine

                    yield _coroutine_factory(idx, self._process, d, kwargs)

            else:

                # initialize context
                context.total_coroutines = 1
                    
                def _coroutine_factory(index, process, input, kwargs):

                    def _coroutine(*_args, **_kwargs):
                        result = process(input, **kwargs)
                        return (index, result)

                    return _coroutine
                    
                yield _coroutine_factory(0, self._process, input, kwargs)

        # no process specified
        else:

            # initialize context
            context.total_coroutines = 1
            
            def _coroutine_factory(index, input):

                def coroutine(*_args, **_kwargs):
                    result = input
                    return (index, result)

                return _coroutine

            yield _coroutine_factory(0, input)


    @_forward_module.callback # thread safe
    def _forward_module_callback(self, res: tuple, context: Any, **kwargs) -> Any:
        '''
        Return post-processed result

        Args:
            res: (tuple) a tuple of (index, result)
            context: (Any) context objects
        
        Returns:
            (Any) result
        '''

        # append results
        context.temp_outputs.append(res)
        # increase counter
        context.total_completed += 1

        # finalize
        context.fin = (context.total_coroutines == context.total_completed)
        if context.fin:

            # only one output
            if not context.sliced:
                # only one output
                context.outputs = context.temp_outputs[0]
            else:
                # multiple outputs, sorted outputs
                context.outputs = [x[1] for x in sorted(context.temp_outputs, key=lambda x: x[0])]

        return res[1]


    # === alias ===

    def parallelize(self, *args, **kwargs):
        return self.__call__.parallelize(*args, **kwargs)

    def unrolled(self, *args, **kwargs):
        return self.__call__.unrolled(*args, **kwargs)


class StaticSource(Source):
    
    def __init__(self, name: Optional[str], 
                        src: Optional[BaseModule] =None, 
                       data: Any =None, 
                    process: Optional[BaseProcess] =None, 
                     static: bool =False, 
               slice_inputs: bool =True, **kwargs) -> NoReturn:

        super(StaticSource, self).__init__(name=name, 
                                           src=src, 
                                           data=data, 
                                           process=process, 
                                           static=True, 
                                           slice_inputs=slice_inputs, **kwargs)