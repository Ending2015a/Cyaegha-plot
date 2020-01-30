# --- built in ---
import os
import sys
import time
import logging

# --- 3rd party ---
import numpy as np
import pandas as pd
import scipy.stats as st

# --- my module ---
from cyaegha import logger

from cyaegha.common.utils import flatten
from cyaegha.common.utils import is_array
from cyaegha.common.utils import error_msg


from cyaegha.plot.process.base import BaseProcess


__all__ = [
    'Process',
    'Interpolation',
    'BatchAverage',
    'Smoothing',
    'Combine',
    'ConfidenceInterval'
]

class Process(BaseProcess):
    def __init__(self, name, fn, slice_inputs=False, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify.
            fn: (function) custom processing function.
                signature: (self, input, **kwargs)
            slice_inputs: (bool) whether to slice input, if input is a list or tuple
        '''

        super(Process, self).__init__(name=name, slice_inputs=slice_inputs, **kwargs)

        self._fn = fn

    # === override BaseProcess ===
    def _forward_process(self, input, **kwargs):
        '''
        Override BaseProcess._forward_process
        '''
        return self._fn(self, input, **kwargs)


class Interpolation(BaseProcess):
    '''
    Interpolation
    '''

    # === Main interfaces ===

    def __init__(self, name, xaxis, start, end, interval=1000, ignore_nan=False, default_values=None, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify.
            xaxis: (str, int) index of axis or column name
            start: (int) start value
            end: (int or None) end value
            interval: (int) interpolate interval
            ignore_nan: (bool) whether to ignore NaN
            default_values: (int, float, str or dict) default values/methods for filling NA/NaN that cannot be interpolated.
                int, float: value for filling NaN.
                str: method for filling NaN, must be one of ['backfill', 'bfill', 'pad', 'ffill'], please refer to Doc: pandas.DataFrame.fillna
                dict: values to fill NaN for each column
        '''
        super(Interpolation, self).__init__(name=name, slice_inputs=True, **kwargs)
        self._xaxis = xaxis
        self._start = start
        self._end = end
        self._interval = interval
        self._default_values = default_values
        self._ignore_nan = ignore_nan

    # === Sub interfaces ===

    def _forward_process(self, input, **kwargs):
        '''
        Override BaseProcess._forward_process
        '''
        return self._interpolate(input, self._xaxis, self._start, self._end, self._interval,
                                                self._default_values, self._ignore_nan, LOG=self.LOG)

    def _error_message(self):
        '''
        Override BaseProces._error_message
        '''
        return 'Failed to interpolate data'

    # === Functions ===

    @classmethod
    def _interpolate(cls, data, xaxis, start, stop, interval, 
                            default_values, ignore_nan, **kwargs):
        '''
        Reference: 
            https://stackoverflow.com/questions/51838355/applying-interpolation-on-dataframe-based-on-another-dataframe
        '''

        # check data type
        assert isinstance(data, pd.DataFrame), 'data must be a pandas.DataFrame'
        # empty check
        assert not data.empty, 'data is empty'

        # clone data
        data = data.copy()
        data['_cyaegha_fake'] = 0

        # create ticks
        num = np.floor_divide((stop-start), interval)
        if np.remainder((stop-start), interval) == 0.0:
            num += 1
        x = np.linspace(start, stop, num, endpoint=True, dtype=data.dtypes[xaxis])
        
        # create empty DataFrame with same columns
        proc_data = pd.DataFrame()
        for c in data.columns:
            proc_data[c] = np.nan

        # add ticks
        proc_data[xaxis] = x
        proc_data['_cyaegha_fake'] = 1

        # concat original data & fate data
        proc_data = pd.concat([data, proc_data], ignore_index=True, sort=False).sort_values([xaxis])

        # interpolate
        proc_data = proc_data.set_index(xaxis)
        proc_data = proc_data.interpolate(method='index')

        if default_values is not None:

            # fill in uninterpolable NaN
            # if default_values is None:
            #     raise ValueError('default_values must be specified, or set ignore_nan=True')
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
    '''
    BatchAverage
    '''

    # === Main interfaces ===

    def __init__(self, name, xaxis, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify
            xaxis: (int or str) column name or index that average along with.
        '''
        super(BatchAverage, self).__init__(name=name, slice_inputs=False, **kwargs)

        self._xaxis = xaxis

    # === Sub interfaces ===

    def _forward_process(self, input, **kwargs):
        '''
        Override BaseProcess._forward_process
        '''
        if not is_array(input):
            return input
        else:
            return self._average(input, self._xaxis, LOG=self.LOG)

    def _error_message(self):
        '''
        Override BaseProcess._error_message
        '''
        return 'Failed to average data'

    # === Functions ===

    @classmethod
    def _average(cls, datas, xaxis, **kwargs):
        '''
        Average
        '''

        # type check
        for data in datas:
            # type check
            assert isinstance(data, pd.DataFrame), 'inputs must be a list of pd.DataFrame, got {}'.format(type(data))
            # empty check
            assert not data.empty, 'data is empty'

        proc_data = pd.concat(datas, ignore_index=True, sort=False).sort_values([xaxis])
        proc_data = proc_data.groupby(xaxis).mean(skipna=True).reset_index()
        
        return proc_data





class Smoothing(BaseProcess):
    '''
    Smoothing
    '''

    # === Main interfaces ===

    def __init__(self, name, window_size=20, window_type=None, apply_columns=None, exclude_columns=None, **kwargs):
        '''
        Args:
            name: (str or None) process name, used to identify
            window_size: (int) size of the moving window
            window_type: (str) type of the smoothing window, please refer to:
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
            apply_columns: (list of (str, int)) columns to apply smoothing
            exclude_columns: (list of (str, int)) columns not to apply smoothing
        '''
        super(Smoothing, self).__init__(name=name, slice_inputs=True, **kwargs)
        self._window_size = window_size
        self._window_type = window_type
        self._apply_columns = set(apply_columns)
        self._exclude_columns = exclude_columns

    # === Sub interfaces ===
    def _forward_process(self, input, **kwargs):
        '''
        Override BaseProces._forward_process
        '''
        return self._smoothing(input, self._window_size, self._window_type, self._apply_columns, self._exclude_columns, LOG=self.LOG)

    def _error_message(self):
        '''
        Override BaseProcess._error_message
        '''
        return 'Failed to smooth data'


    # === Functions ===
    
    @classmethod
    def _smoothing(cls, data, window_size, window_type, apply_cols, exclude_cols, **kwargs):
        
        # type check
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame, got {}'.format(type(data))
        # empty check
        assert not data.empty, 'data is empty'

        # clone data
        data = data.copy()

        # get numeric columns
        numeric_columns = set(data.select_dtypes(include='number'))

        # set the columns to apply to all columns if it's None
        apply_cols = apply_cols or data.columns

        # find difference between columns and excluded columns
        if exclude_cols is not None:
            apply_cols = apply_cols.difference(exclude_cols)

        # only apply to the numerical columns
        apply_cols = apply_cols & numeric_columns

        # get column data
        proc_data = data[apply_cols]

        # smoothing
        proc_data = proc_data.rolling(window=window_size, min_periods=1, win_type=window_type).mean()

        # update with smoothed data
        data.update(proc_data)

        return data




class Combine(BaseProcess):
    '''
    Combine all pd.DataFrame
    '''

    # === Main interfaces ===
    def __init__(self, name, **kwargs):
        super(Combine, self).__init__(name=name, slice_inputs=False, **kwargs)


    # === Sub interfaces ===
    def _forward_process(self, input, **kwargs):
        '''
        Override BaseProces._forward_process
        '''
        return self._combine(input, LOG=self.LOG)

    def _error_message(self):
        '''
        Override BaseProcess._error_message
        '''
        return 'Failed to combine pd.DataFrame'

    # === Functions ===

    @classmethod
    def _combine(cls, data, **kwargs):
        
        if isinstance(data, pd.DataFrame):
            return data
        elif is_array(data):
            flattened_data = flatten(data)

            assert all(isinstance(d, pd.DataFrame) for d in flattened_data), 'data must be pd.DataFrame or a list of pd.DataFrame'

            data = pd.concat( flattened_data )
            return data
        else:
            raise ValueError('Receive unknown type: {}'.format(type(data)))





class ConfidenceInterval(BaseProcess):
    '''
    Compute confidence interval
    '''

    # === Main interfaces ===

    def __init__(self, name, xaxis, yaxis, ci=0.95, hi_col='high', lo_col='low', mean_col='mean', **kwargs):

        super(ConfidenceInterval, self).__init__(name=name, slice_inputs=True, **kwargs)

        self._xaxis = xaxis
        self._yaxis = yaxis
        self._ci = ci
        self._hi_col = hi_col
        self._lo_col = lo_col
        self._mean_col = mean_col

    # === Sub interfaces ===
    def _forward_process(self, input, **kwargs):
        '''
        Override BaseProces._forward_process
        '''
        return self._interval(input, self._xaxis, self._yaxis, self._ci, self._hi_col, 
                                            self._lo_col, self._mean_col, LOG=self.LOG)

    def _error_message(self):
        '''
        Override BaseProcess._error_message
        '''
        return 'Failed to compute confidence interval'

    # === Functions ===

    @classmethod
    def _interval(cls, data, xaxis, yaxis, ci, hi_col, lo_col, mean_col, LOG=None, **kwargs):

        # type check
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame, got {}'.format(type(data))
        # empty check
        assert not data.empty, 'data is empty'

        # print warning if data as NaN value
        if data.isnull().values.any():
            LOG.warning('the data has NaN values')

            LOG.set_header()
            LOG.add_rows(data.isna().any(axis=1).head(n=5), fmt='{}')
            LOG.flush('WARN')
        
        # get default x column
        if xaxis is None:
            xaxis = data.index
        
        # get default y column (first numeric column)
        if yaxis is None:
            columns = data.select_dtypes(include=np.number).columns

            assert len(columns) > 0, 'No numeric columns found'
            yaxis = columns[0]

        # group the data by x column
        groups = data.groupby(xaxis)

        # get the number of group
        num_groups = len(groups)

        # initialize the x series
        xs = np.zeros(num_groups)

        # initialize h
        hs = np.zeros(num_groups)
        # initialize y series
        y_mean = np.zeros(num_groups)

        for idx, (x, group) in enumerate(groups):
            # get y series
            y = group[yaxis]

            # get number of y
            num = len(y)

            # calculate mean and ignore NaN
            mean_s = np.nanmean(y)

            # only calculate bounds of confidence interval when the number of Ys are more than 1
            if num > 1:
                std_err = st.sem(y)
                hs[idx] = std_err * st.t.ppf((1 + ci) / 2, num - 1)
            else:
                hs[idx] = 0

            # add x to the list
            xs[idx] = x

            # add mean values to the list
            y_mean[idx] = mean_s

        y_lo = y_mean - hs
        y_hi = y_mean + hs

        data = pd.DataFrame({xaxis: xs, 
                            hi_col: y_hi, 
                            lo_col: y_lo, 
                            mean_col: y_mean})

        return data