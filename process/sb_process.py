# --- built in ---
import re
import os
import csv
import sys
import json
import time
import logging

from glob import glob

# --- 3rd party ---
import pandas as pd

import numpy as np
import plotly.graph_objects as go

# --- my module ---
from cyaegha import logger

from cyaegha.common.route import Route
from cyaegha.plot.process.base import BaseProcess

__all__ = [
    'StableBaselinesProcess'
]


X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

# =========
# Copy from hill-a/stable-baselines: 
#        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/bench/monitor.py
# vvvvvvvvv

class LoadMonitorResultsError(Exception):
    """
    Copy from hill-a/stable-baselines: 
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/bench/monitor.py

    Raised when loading the monitor log fails.
    """
    pass


def load_monitors(path):
    """
    Copy from hill-a/stable-baselines: 
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/bench/monitor.py

    Load all Monitor logs from a given directory path matching ``*monitor.csv`` and ``*monitor.json``
    
    Args:
        path: (str) the directory path containing the log file(s)
    
    Returns: 
        (Pandas DataFrame) the logged data
    """
    # get both csv and (old) json files
    monitor_files = (glob(os.path.join(path, "*monitor.json")) + get_monitor_files(path))
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, path))
    data_frames = []
    headers = []
    for file_name in monitor_files:
        with open(file_name, 'rt') as file_handler:
            if file_name.endswith('csv'):
                first_line = file_handler.readline()
                assert first_line[0] == '#'
                header = json.loads(first_line[1:])
                data_frame = pd.read_csv(file_handler, index_col=None)
                headers.append(header)
            elif file_name.endswith('json'):  # Deprecated json format
                episodes = []
                lines = file_handler.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                data_frame = pd.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pd.concat(data_frames)
    data_frame.sort_values('t', inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame


def ts2xy(timesteps, xaxis):
    """
    Copy from hill-a/stable-baselines: 
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/bench/monitor.py

    Decompose a timesteps variable to x ans ys
    Args:
        timesteps: (Pandas DataFrame) the input data
        xaxis: (str) the axis for the x and y output
            (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    
    Returns: 
        (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var

def load_results(dirs, num_timesteps, xaxis):
    """
    Copy from hill-a/stable-baselines: 
        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/bench/monitor.py

    plot the results
    Args:
        dirs: ([str]) the save location of the results to plot
        num_timesteps: (int or None) only plot the points below this value
        xaxis: (str) the axis for the x and y output
            (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    """

    tslist = []
    for folder in dirs:
        timesteps = load_monitors(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]

    return xy_list

# ^^^^^^^^^
# Copy from hill-a/stable-baselines: 
#        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/bench/monitor.py
# =========



class StableBaselinesProcess(BaseProcess):
    '''
    StableBaselinesProcess

    Read files generated by stable-baselines monitor

    input: (list of str, str) folder names
    output: (pandas.Dataframe) loaded results
    '''
    def __init__(self, name, num_timesteps, xaxis='timesteps', **kwargs):
        super(StableBaselinesProcess, self).__init__(name=name, slice_inputs=False, **kwargs)

        self._num_timesteps = num_timesteps
        self._xaxis = xaxis


    def _forward_module(self, input, **kwargs):
        '''
        Args:
            input: (list of str, str) folder names
        
        Returns:
            list of pandas.Dataframe if input is a list of str,
            otherwise, return pandas.Dataframe
        '''

        try:

            # input is a list of str
            if isinstance(input, list):
                results = load_results(input, self._num_timesteps, self._xaxis)

                outputs = []
                for r in results:
                    # convert to data frame
                    df = pd.DataFrame({self._xaxis: r[0].astype(np.int64),
                                       'rewards': r[1].astype(np.float32)})
                    outputs.append(df)
            else:
                # input is str
                assert isinstance(input, str), 'input must be an str or list of strs'

                results = load_results([input], self._num_timesteps, self._xaxis)
                # unpack list, which only contains one element
                result = results[0]

                # convert to DataFrame
                outputs = df.DataFrame({self._xaxis: result[0].astype(np.int64),
                                        'rewards': result[1].astype(np.float32)})
            

        except Exception as e:

            self.LOG.exception(
                error_msg(e, '{ERROR_TYPE}: Failed to load stable baselines monitor: {ERROR_MSG}'))

        return outputs

    def _forward_process(self, input, **kwargs):
        pass
        