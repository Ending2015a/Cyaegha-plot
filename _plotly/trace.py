# --- built in ---
import os
import abc
import sys
import time
import logging

from typing import Any
from typing import Union
from typing import List
from typing import Tuple
from typing import Callable
from typing import NoReturn
from typing import Optional
from typing import Hashable

# --- 3rd party ---
import numpy as np
import pandas as pd
import scipy.stats as st

import plotly
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# --- my module ---

from cyaegha import logger

from cyaegha.common.utils import is_array

from cyaegha.plot.base import BaseTrace



__all__ = [
    'Trace'
]



# TODO: trace preset

class Trace(BaseTrace):

    # === Attributes ===

    _type_mapping: dict = dict()
    _alias_mapping: dict = dict()

    # === Main interfaces ===

    @classmethod
    def register(cls, *args, name: Optional[str] =None, force: bool =False, 
                            alias: Optional[Union[Hashable, List[Hashable]]]):
        '''
        Register new trace type

        Args:
            args[0]: (FunctionType) Function to generate traces
            name: (str or None) type name. If None, args[0].__name__ is used
            force: (bool) whether to override registered trace types
            alias: (Hashable, list of Hashable) alias name for the type, can be any hashable objects

        
        1. decorator type A: 
            @Trace.register
            def hello():

        2. decorator type B:
            @Trace.register(name='Hello')
            def _hello():

        3. function:
            Trace.register(_hello, name='Hello', force=True)
        '''

        def _wrapper(func, _name=name, _force=force, _alias=alias):

            # in default using function name as type name
            if _name is None:
                _name = func.__name__

            # check if func is callable
            if not callable(func):
                raise RuntimeError('Type function must be a callable object')

            # register
            cls._add_type(_name, func, force=_force)
            cls._add_alias(_name, alias=_alias, force=_force)

            # return original function
            return func


        if len(args) == 0:
            # Type B
            return _wrapper
        else:
            # Type A, function
            _wrapper(args[0], _name=name, _force=force)

            
        return func

    @classmethod
    def registered(cls):
        '''
        Return registered types
        '''

        return dict(cls._type_mapping)

    # === Sub interfaces ===

    def _execute_object(self, input: Any =None, 
                              combine_source: bool =False, 
                              legend_group: Optional[str] =None, 
                              trace_configs: Union[list, dict] =dict(), **kwargs) -> Any:
        '''
        Execute object

        Args:
            input: (str)
            combine_source: (bool)
            legend_group: (str or None)
            trace_configs: (list of dict or dict)
        '''

        if not self.load(input=input, **kwargs):
            raise RuntimeError('Failed to load sources')

        # preprocess input
        if (combine_source) and is_array(self.sources):
            input = pd.concat(self.sources)
        else:
            input = self.sources

        # legend name
        self.config['name'] = self.name

        # legend group
        if legend_group is not None:
            self.config['legendgroup'] = legend_group

        
        # create traces 
        if is_array(input):
            traces = []
            # slice input
            for d in input:
                trace = self._generate_trace(d, self.config, trace_configs)

                if is_array(trace):
                    traces.extend(trace)
                else:
                    traces.append(trace)
        else:
            traces = self._generate_trace(input, self.config, trace_configs)

        return traces


    # === Private ===

    def _generate_trace(self, input: Any, config, update_config):

        # get fn
        fn = type(self)._get_type(self.type)

        traces = fn(self, input, **config)

        # N to N update
        if is_array(update_config) and is_array(traces):
            
            if not len(update_config) == len(traces):
                raise RuntimeError('The trace configuration must have same length as generated traces')
            
            for t, c in zip(traces, update_config):
                t.update(**c)

        # one to N update
        elif is_array(traces):

            for t in traces:
                t.update(**update_config)

        # one to one update
        else:
            if is_array(update_config):
                raise RuntimeError('Only one trace but receive mutiple configurations')
            
            traces.update(**update_config)

        return traces

    

    @classmethod
    def _add_type(cls, name: str, func: Callable, force: bool):

        # check if the type name is already registered
        if (name in cls._type_mapping) and (not force):
            raise RuntimeError('Trace type \'{}\' is already registered'.format(name))
        
        cls._type_mapping[name] = func

    @classmethod
    def _get_type(cls, name: Hashable):
        
        if name in cls._alias_mapping:
            name = cls._alias_mapping[name]

        if name not in cls._type_mapping:
            raise RuntimeError('Type \'{}\' is not registered'.format(name))

        return cls._type_mapping[name]

    @classmethod
    def _add_alias(cls, name: str, alias: Hashable, force: bool):

        # check if the type name exists
        if (name not in cls._type_mapping):
            raise RuntimeError('Trace type \'{}\' does not exist, cannot create aliasing'.format(name))

        # NOTE: list is not hashable
        if isinstance(alias, list):
            for a in alias:
                if (a in cls._alias_mapping) and (not force):
                    raise RuntimeError('The aliasing for type \'{}\' is already used by type \'{}\''.format(
                                                                                name, cls._alias_mapping[a]))
                cls._alias_mapping[a] = name

        # other hashable
        else:
            if (alias in cls._alias_mapping) and (not force):
                raise RuntimeError('The aliasing for type \'{}\' is already used by type \'{}\''.format(
                                                                                name, cls._alias_mapping[a]))

            cls._alias_mapping[alias] = name


# === trace functions ===

@Trace.register(name='scatter', alias=['Scatter', go.Scatter])
def _scatter(self, data, x_col=None, y_col=None, **kwargs):
    '''
    Scatter

    Args:
        data: (pd.DataFrame)
        x_col: (int, slice, tuple, str)
        y_col: (int, slice, tuple, str)
    '''

    # check data type
    if not isinstance(data, pd.DataFrame):
        raise ValueError('data must be a pandas.DataFrame')


    # get x column
    if x_col is None:
        # default using index as x series
        x = np.array(data.index)
    elif isinstance(x_col, (int, slice)):
        # integer-location based indexing 
        x = data.iloc[:, x_col].to_numpy()
    elif isinstance(x_col, str):
        # label indexing
        x = data.loc[:, x_col].to_numpy()
    else:
        raise ValueError('Unknown x_col value: {}'.format(x_col))


    # get x column
    if y_col is None:
        # get first numerical columns
        columns = data.select_dtypes(include=np.number).columns
        assert len(columns) > 0, 'No numeric columns found'
        y = data.loc[:, columns[0]]
    elif isinstance(y_col, (int, slice)):
        # integer-location based indexing
        y = data.iloc[:, y_col].to_numpy()
    elif isinstance(y_col, str):
        # labe indexing 
        y = data.loc[:, y_col].to_numpy()
    else:
        raise ValueError('Unknown x_col value: {}'.format(x_col))

    # assert two series have same length
    assert len(x) == len(y), 'X and Y must have same length, got {} and {}'.format(len(x), len(y))

    # create trace
    trace = go.Scatter(x=x, y=y, **kwargs)

    return trace


@Trace.register(name='interval', alias=['Interval', 'confidence', 'Confidence'])
def _interval(self, data, x_col=None, y_col=None, ci: float=0.95, **kwargs):
    '''
    Confidence interval

    References:
        * https://github.com/elsa-lab/plotting-wand/blob/master/plotting_wand/helpers/plotting.py
    '''

    # check data type
    if not isinstance(data, pd.DataFrame):
        raise ValueError('data must be a pandas.DataFrame')

    # print warning if data as NaN value
    if data.isnull().values.any():
        self.LOG.warning('the data has NaN values')

        self.LOG.set_header()
        self.LOG.add_rows(data.isna().any(axis=1).head(n=5), fmt='{}')
        self.LOG.flush('WARN')

    # get default x column
    if x_col is None:
        x_col = data.index
    
    # get default y column (first numeric column)
    if y_col is None:
        columns = data.select_dtypes(include=np.number).columns

        assert len(columns) > 0, 'No numeric columns found'
        y_col = columns[0]

    # group the data by x column
    groups = data.groupby(x_col)

    # get the number of group
    num_groups = len(groups)

    # initialize the x series
    xs = np.zeros(num_groups)

    # initialize y series
    y_lo = np.zeros(num_groups)
    y_hi = np.zeros(num_groups), 
    y_mean = np.zeros(num_groups)

    for idx, (x, group) in enumerate(groups):
        # get y series
        y = group[y_col]

        # get number of y
        num = len(y)

        # calculate mean and ignore NaN
        mean_s = np.nanmean(y)

        # only calculate bounds of confidence interval when the number of Ys are more than 1
        if num > 1:
            std_err = st.sem(y)
            h = std_err * st.t.ppf((1 + ci) / 2, num - 1)
            lo, hi = mean_s - h, mean_s + h
        else:
            lo, hi = mean_s, mean_s

        # add x to the list
        xs[idx] = x

        # add bounds to the lists
        y_lo[idx] = lo
        y_hi[idx] = hi

        # add mean values to the list
        y_mean[idx] = mean_s

    # set common attributes for building traces for confidence interval
    ci_attrs = dict(hoverinfo='skip', line_width=0,
                    mode='lines', showlegend=False)

    # build the lower bounds line
    trace_lo = go.Scatter(x=xs, y=y_lo, fill=None, **ci_attrs, **kwargs)

    # build the upper bounds line which fills the area
    trace_hi = go.Scatter(x=xs, y=y_hi, fill='tonexty', **ci_attrs, **kwargs)

    # build the mean values line
    trace_mean = go.Scatter(x=xs, y=y_mean, mode='lines', **kwargs)

    # return the lower bound, upper bounds and mean values traces
    return [trace_lo, trace_hi, trace_mean]

