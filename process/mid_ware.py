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

from cyaegha.plot.process.base import BaseProcess


class Process(BaseProcess):
    def __init__(self, name, process, unpack_batch=False, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify.
            process: (function) custom processing function.
                signature: (self, input, **kwargs)
            unpack_batch: (bool) whether to unpack input, if input is a list or tuple
        '''

        super(Process, self).__init__(name=name, **kwargs)

        self._process = process
        self._unpack_batch = unpack_batch

    def _setup_module(self,**kwargs):
        pass

    def _forward_module(self, input, **kwargs):
        
        try:
            if self._unpack_batch and is_array(input):
                outputs = []
                for data in input:
                    output = self._process(self, data, **kwargs)
                    outputs.append(output)
            else:
                outputs = self._process(self, input, **kwargs)

        except Exception as e:
            self.LOG.exception(
                error_msg(e, '{ERROR_TYPE}: Failed to process data: {ERROR_MSG}'))
            
        return outputs


class Interpolation(BaseProcess):

    def __init__(self, name, xaxis, start, end, interval=1000, default_values=None, ignore_nan=False, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify.
            xaxis: (str, int) index of axis or column name
            start: (int) start value
            end: (int or None) end value
            interval: (int) interpolate interval
            default_values: (int, float, str or dict) default values/methods for filling NA/NaN that cannot be interpolated.
                int, float: value for filling NaN.
                str: method for filling NaN, must be one of ['backfill', 'bfill', 'pad', 'ffill'], please refer to Doc: pandas.DataFrame.fillna
                dict: values to fill NaN for each column
            ignore_nan: (bool) whether to ignore NaN
        '''
        super(Interpolation, self).__init__(name=name, **kwargs)
        self._xaxis = xaxis
        self._start = start
        self._end = end
        self._interval = interval
        self._default_values = default_values
        self._ignore_nan = ignore_nan

    def _setup_module(self, **kwargs):
        pass

    def _forward_module(self, input, **kwargs):
        '''
        Args:
            input: (pandas.Dataframe or a list of pandas.Dataframe) input data
        '''

        try:
            if is_array(input):
                outputs = []
                for data in input:
                    # processing data
                    processed = self._interpolate(data, self._xaxis, self._start, self._end, self._interval
                                                self._default_values, self._ignore_nan)
                    outputs.append(processed)

            else:
                outputs = self._interpolate(input, self._xaxis, self._start, self._end, self._interval
                                                self._default_values, self._ignore_nan)

        except Exception as e:
            self.LOG.exception(
                error_msg(e, '{ERROR_TYPE}: Failed to interpolate data: {ERROR_MSG}'))

            raise

        return outputs

    # === utilities ===

    @classmethod
    def _interpolate(cls, data, xaxis, start, stop, interval, 
                            default_values, ignore_nan, **kwargs):
        '''
        Reference: 
            https://stackoverflow.com/questions/51838355/applying-interpolation-on-dataframe-based-on-another-dataframe
        '''

        # check data type
        assert isinstance(data, pd.DataFrame), 'data must be a pandas.DataFrame'

        # clone data
        data = data.copy()
        data['_cyaegha_fake'] = 0

        # create ticks
        num = np.floor_divide((stop-start), interval)
        if np.remainder((stop-start), interval) == 0.0:
            num += 1
        x = np.linspace(start, stop, num, endpoint=True, dtype=data.dtypes[xaxis])
        
        # create empty DataFrame with same columns
        processed_data = pd.DataFrame()
        for c in data.columns:
            processed_data[c] = np.nan

        # add ticks
        proc_data[xaxis] = x
        proc_data['_cyaegha_fake'] = 1

        # concat original data & fate data
        proc_data = pd.concat([data, proc_data], ignore_index=True, sort=False).sort_values([xaxis])

        # interpolate
        proc_data = proc_data.set_index(xaxis)
        proc_data = proc_data.interpolate(method='index')

        # fill in uninterpolable NaN
        if isinstance(default_values, str):
            proc_data = proc_data.fillna(method=default_values)
        else:
            proc_data = proc_data.fillna(default_values)

        # reset index, remove original data
        proc_data = proc_data.reset_index()
        proc_data = proc_data.loc[proc_data['_cyaegha_fake'] == 1]
        proc_data = proc_data.drop('_cyaegha_fake', axis=1).reset_index()

        # NaN check
        if not ignore_nan:
            na_count = proc_data.isnull().sum()
            if na_count > 0:
                raise ValueError('There are {} NaNs in the DataFrame'.format(na_count))

        return proc_data


class BatchAverage(BaseProcess):

    def __init__(self, name, xaxis):
        super(BatchAverage, self).__init__(name=name, **kwargs)

        self._xaxis = xaxis

    def _setup_module(self, **kwargs):
        pass

    def _forward_module(self, input, **kwargs):
        '''
        Args:
            input: (list of pandas.DataFrame) input data
        '''

        try:
            if is_array(input):
                output = self._average(input, self._xaxis)
            else:
                output = input

        except Exception as e:
            self.LOG.exception(
                error_msg(e, '{ERROR_TYPE}: Failed to average data: {ERROR_MSG}'))

            raise

        return output

    # === utilities ===

    @classmethod
    def _average(cls, datas, xaxis):
        '''
        Average
        '''

        # type check
        for data in datas:
            assert isinstance(data, pd.DataFrame), 'inputs must be a list of pd.DataFrame, got {}'.format(type(data))

        proc_data = pd.concat(datas, ignore_index=True, sort=False).sort_values([xaxis])
        proc_data = proc_data.groupby(xaxis).mean().reset_index()
        
        return proc_data


class Smoothing(BaseProcess):

    def __init__(self, name, window_size=20, window_type=None, apply_columns=None, except_columns=None, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify
            window_size: (int) size of the moving window
            window_type: (str) type of the smoothing window, please refer to:
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
            apply_columns: (list of (str, int)) columns to apply smoothing
            except_columns: (list of (str, int)) columns not to apply smoothing
        '''
        super(Smoothing, self).__init__(name=name, **kwargs)
        self._window_size = window_size
        self._window_type = window_type
        self._apply_columns = apply_columns
        self._except_columns = except_columns

    def _setup_module(self, **kwargs):
        pass

    def _forward_module(self, input, **kwargs):
        '''
        Args:
            input: (pandas.DataFrame or list of pandas.DataFrame) input data
        '''

        try:
            if is_array(input):
                outputs = []
                for data in input:
                    output = self._smoothing(data, self._window_size, self._window_type, self._apply_columns, self._except_columns)
                    outputs.append(output)
            else:
                outputs = self._smoothing(data, self._window_size, self._window_type, self._apply_columns, self._except_columns)

        except Exception as e:
            self.LOG.exception(
                error_msg(e, '{ERROR_TYPE}: Failed to smooth data: {ERROR_MSG}'))

            raise

        return outputs

    # === utilities ===
    @classmethod
    def _smoothing(cls, data, window_size, window_type, apply_cols, except_cols):

