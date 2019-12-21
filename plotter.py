import os
import sys
import time
import logging

import importlib
import pandas as pd

import logger
logger.Config.Use(filename='plotting_logger.log', level='DEBUG', colored=True)

from pymap import Map
from color import trace_colors

from matplotlib.colors import to_hex
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from plotting_wand.plotter import plot as plotly_plot
from plotting_wand.helpers.layout import adjust_annotations_font_sizes
from plotting_wand.helpers.layout import adjust_annotations_shifts
from plotting_wand.helpers.plotting import build_confidence_interval_traces

from configs.beamrider_plot import beamrider_plot_factory
from configs.compare_plot import compare_plots_factory
from configs.seaquest_plot import seaquest_plot_factory
from configs.seaquest_skill_plot import seaquest_skill_plot_factory

LOG = logger.getLogger('main')

config = {
    'plot': [
            # beamrider_plot_factory(),
             seaquest_plot_factory()]
}


def get_row_col_by_idx(config, subplot_idx):
    total_rows = config.get('subplots.settings.rows', 1)
    total_cols = config.get('subplots.settings.cols', 1)

    col = int(subplot_idx % total_cols)
    row = int(subplot_idx / total_cols)

    return row+1, col+1




def get_data(config, data_name):
    data = config.get('data.{}.process'.format(data_name), [(None, {'_outputs': None})])[-1][1].get('_outputs', None)
    return data

def plot_trace(config, fig, content, subplot_idx, trace, trace_idx):

    data_name = trace.get('data', None)

    if data_name is None:
        LOG.warning('cannot find data for trace: {}'.format(trace))
        return

    data = get_data(config, data_name)
    #data = config.get('data.{}.process'.format(data_name), [(None, {'_outputs': None})])[-1][1].get('_outputs', None)

    if data is None:
        LOG.warning('cannot find data {} for trace: {}'.format(data_name, trace))


    title = content.get('title', 'Title_{}'.format(subplot_idx))
    show_legend = content.get('show_legend', 'auto')
    plot_type = content.get('type', 'confidence')

    legend_name = trace.get('legend_name', 'unknown_{}'.format(trace_idx))
    color = trace.get('color', 'auto')
    x_column = trace.get('x_column', 0)
    y_column = trace.get('y_column', 1)

    row, col = get_row_col_by_idx(config, subplot_idx)

    if color == 'auto':
        color = None
    if color == 'id':
        color = trace_colors[trace_idx]
    elif isinstance(color, int):
        color = trace_colors[color]


    if show_legend == 'auto':
        show_legend = True #(subplot_idx == 0)

    if plot_type == 'confidence':

        trace_lo, trace_hi, trace_mean = build_confidence_interval_traces(
            data, x_column, y_column, ci=0.6, legendgroup=title,
            line_color=color, name=legend_name)

        trace_mean.update(showlegend=show_legend)

        fig.add_trace(trace_lo, row=row, col=col)
        fig.add_trace(trace_hi, row=row, col=col)
        fig.add_trace(trace_mean, row=row, col=col)


    else:
        LOG.warning('Unknown plot type: {}'.format(plot_type))





def plot(config):
    config = Map(config)

    fig = make_subplots(**config.get('subplots.settings', {}))

    contents = config.get('subplots.contents', [])

    total_subplots = len(contents)
 

    for subplot_idx, content in enumerate(contents):
        #row, col = get_row_col_by_idx(config, subplot_idx)

        LOG.info('plotting subplot {}/{} ...'.format(subplot_idx+1, total_subplots))



        total_traces = len(content.get('traces', []))
        for trace_idx, trace in enumerate(content.get('traces', [])):

            LOG.info('plotting trace {}/{} ...'.format(trace_idx+1, total_traces))

            plot_trace(config, fig, content, subplot_idx, trace, trace_idx)

            data = get_data(config, trace.get('data', None))



    fig.update_layout(**config.get('subplots.layout', {}))

    #adjust_annotations_font_sizes(fig, factor=1.1)
    #adjust_annotations_shifts(fig, x_factor=0.5, y_factor=0.5)

    return fig


    
def prepare_data(config):
    
    for name, process in config.items():
        LOG.debug('processing data: {}'.format(name))

        for proc in process.get('process', []):
            name, io = proc
            res = name.rsplit('.', 1)
            if len(res) > 1:
                LOG.debug('import module: {}'.format(res[0]))
                module = importlib.import_module(res[0])
                LOG.debug('get function: {}'.format(res[1]))
                func = getattr(module, res[1])
            else:
                module = sys.modules[__name__]
                LOG.debug('get function: {}'.format(res[0]))
                func = getattr(module, res[0])
            
            LOG.debug('processing...')


            res = func(**io.get('inputs', {}))

            output_name = io.get('outputs', None)

            if output_name is None:
                io['_outputs'] = res
            elif isinstance(output_name, list):
                io['_outputs'] = dict(zip(output_name, res))
            else:
                io['_outputs'] = res

            LOG.debug('get results: \n{}'.format(res))

            prev_proc = proc


def main(config):
    num_plot = len(config['plot'])


    LOG.info('{} plots'.format(num_plot))


    for idx, each_plot in enumerate(config['plot']):
        
        LOG.info('{}/{} plots'.format(idx+1, num_plot))
        LOG.info('preparing data...')
        prepare_data(each_plot['data'])
        LOG.info('done')

        LOG.info('plotting...')
        fig = plot(each_plot)
        LOG.info('done')

        if each_plot['offline']:
            fig.write_image(each_plot.get('filename', 'output_{}.png'.format(idx)), 
                            width=each_plot.get('width', 1600), 
                            height=each_plot.get('height', 800),
                            scale=2)
        else:
            plotly_plot(figure=fig, renderer='svg')

if __name__ == '__main__':
    main(config)
