import re
import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add abspath(../) to path

from glob import glob
import logger
LOG = logger.getLogger('config_factory', 'INFO')

from matplotlib.colors import to_hex
import plotly.graph_objects as go

def get_process(**kwargs):

    process = kwargs.get('process', 1)

    if isinstance(process, str):
        if process == 'process.load_results':
            process = 1
        elif process == 'process.load_period_eval':
            process = 2
        elif process == 'process.load_option_critic':
            process = 3

    # monitor
    if process == 1:
        root = kwargs.get('root', None)
        sources = kwargs.get('sources', None)
        max_steps = kwargs.get('max_steps', 10000000)
        smooth_window = kwargs.get('smooth_window', 200)
        
        d = {
            'process_name': 'process.load_results',
            'inputs': {
                'root': root,
                'sources': sources,
                'max_steps': max_steps,
                'smooth_window': smooth_window
            }
        }

    # period_eval
    elif process == 2:
        root = kwargs.get('root', None)
        sources = kwargs.get('sources', None)
        max_steps = kwargs.get('max_steps', 10000000)
        smooth_window = kwargs.get('smooth_window', 20)
        
        d = {
            'process_name': 'process.load_period_eval',
            'inputs': {
                'root': root,
                'sources': sources,
                'max_steps': max_steps,
                'smooth_window': smooth_window
            }
        }
    
    # option-critic
    elif process == 3:
        root = kwargs.get('root', None)
        sources = kwargs.get('sources', None)
        max_steps = kwargs.get('max_steps', 10000000)
        smooth_window = kwargs.get('smooth_window', 200)
        
        d = {
            'process_name': 'process.load_option_critic',
            'inputs': {
                'root': root,
                'sources': sources,
                'max_steps': max_steps,
                'smooth_window': smooth_window
            }
        }

    LOG.debug('process: {}'.format(d))

    return d
    


def get_template(legend_pos='top-left', **kwargs):

    legend_y_pos, legend_x_pos = legend_pos.split('-')

    if legend_y_pos == 'top':
        legend_y = 1.0
        legend_yanchor = 'top'
    elif legend_y_pos == 'middle':
        legend_y = 0.5
        legend_yanchor = 'middle'
    elif legend_y_pos == 'bottom':
        legend_y = 0.0
        legend_yanchor = 'bottom'
    else:
        raise RuntimeError('Unknown legend_y_pos: {}'.format(legend_y_pos))

    if legend_x_pos == 'right':
        legend_x = 1.0
        legend_xanchor = 'right'
    elif legend_x_pos == 'center':
        legend_x = 0.5
        legend_xanchor = 'center'
    elif legend_x_pos == 'left':
        legend_x = 0.0
        legend_xanchor = 'left'
    else:
        raise RuntimeError('Unknown legend_x_pos: {}'.format(legend_x_pos))
    

    template = {
        'subplots': {
            'settings': {
                'rows': 1,
                'cols': 1,
                'shared_xaxes': False,
                'shared_yaxes': False,
                'start_cell': 'top-left',
                'print_grid': True,
                'horizontal_spacing': None,
                'vertical_spacing': None,
                #'subplot_titles': ['BeamRider'],
                'column_widths': None,
                'row_heights': None,
                'specs': None,
                'insets': None,
                'column_titles': None,
                'row_titles': None,
                #'x_title': 'Timesteps',
                #'y_title': 'Rewards'
            },
            'layout': {
                'title': {
                    'text': None,
                    'font.size': 48,
                    'xref': 'paper',
                    'x': 0.5,
                    'y': 0.99,
                },
                'template': 'plotly_white',
                'font.size': 32,
                'font.family': '"Times New Roman"',
                'legend': {
                    'font.size': 32,
                    'x': legend_x,
                    'y': legend_y,
                    'xanchor': legend_xanchor,
                    'yanchor': legend_yanchor,
                    'tracegroupgap': 0,
                    'bordercolor': 'lightgray',
                    'borderwidth': 2
                },
                'margin': {
                    'l': 125,
                    'r': 40,
                    't': 55,
                    'b': 100
                },
                'xaxis': {
                    'tick0': 0,
                    'tickformat': 's',
                    'ticks': 'outside',
                    'mirror': True,
                    'linecolor': 'gray',
                    'linewidth': 2
                },
                'yaxis': {
                    'tick0': 0,
                    'tickformat': 's',
                    'ticks': 'outside',
                    'mirror': True,
                    'linecolor': 'gray',
                    'linewidth': 2
                },
                'annotations': [
                    {
                        'x': 0.5,
                        'y': -0.22,
                        'showarrow': False,
                        'text': 'Timesteps',
                        'xref': 'paper',
                        'yref': 'paper',
                        'font': {
                            'size': 36,
                        }
                    },
                    {
                        'x': -0.20,
                        'y': 0.5,
                        'showarrow': False,
                        'textangle': -90,
                        'text': 'Rewards',
                        'xref': 'paper',
                        'yref': 'paper',
                        'font': {
                            'size': 36,
                        }
                    },
                    
                ],
            },
            'contents': [
            ]
        },

        'data': {
        },

        'offline': True,
        'filename': 'Unknown.png',
        'width': 800,
        'height': 600,
        'scale': 4
    }

    return template



def new_content(title='Unknown', type='confidence', traces=None, show_legend='auto', **kwargs):
    content = {
        'title': title,
        'type': type,
        'show_legend': show_legend,
        'traces': []
    }

    if traces is not None:
        if not isinstance(traces, list):
            traces = [traces]

        content['traces'] = traces

    return content


def new_trace(name='Unknown', **kwargs):

    trace = {
        'data': name,
        'legend_name': name,
        'color': 'auto',
        'x_column': 'timesteps',
        'y_column': 'rewards'
    }

    trace.update(kwargs)

    return trace

def new_process(process_name, **kwargs):

    default_process_config = {
        'process.load_results': {
            'inputs': {
                'root': None,
                'sources': None,
                'combine_sources': True,
                'to_dataframe': True,
                'max_steps': 20000000,
                'smoothing': True,
                'interpolate': True,
                'smooth_window': 160
            },
        },
        'process.load_skill_rewards': {
            'inputs': {
                'path': None,
                'to_dataframe': True,
                'smoothing': True,
                'smooth_window': 160,
                'drop_nan': True
            }
        }
    }

    default_config = default_process_config.get(process_name, {'inputs': {}})

    for k, v in default_config.items():
        v.update(kwargs.get(k, {}))

    return (process_name, default_config)


def new_data(name='Unknown'):
    return (
        name, {
            'process': []
        }
    )

def add_content(config, content):
    config['subplots']['contents'].append(content)    


def add_trace(content, trace):
    content['traces'].append(trace)


def add_process(data_config, process):
    data_config[-1]['process'].append(process)


def add_data(config, data_config):
    config['data'][data_config[0]] = data_config[1]

'''
def add_data(config, root, sources=None, name='Unknown', endswith='monitor', **kwargs):

    if not isinstance(sources, list):
        sources = [sources]

    if endswith is not None:
        sources = [ s if s.endswith(endswith) else os.path.join(name, endswith) for s in sources ]

    process = kwargs.get('process', 'process.load_results')
    inputs = kwargs.get('inputs', {})
    


    # add data sources
    config['data'][name] = {
        'process': [
            (
                process,
                {
                    'inputs': {
                        'root': root,
                        'sources': sources,
                        'combine_sources': True,
                        'to_dataframe': True,
                        'max_steps': 20000000,
                        'smoothing': True,
                        'interpolate': True,
                        'smooth_window': 160
                    },
                    #'outputs': [0, 1]
                }
            )
        ]
    }

    config['data'][name]['process'][0][1]['inputs'].update(inputs)
'''

def grab_folders(root, target_folder='**/monitor'):

    glob_path = os.path.join(root, target_folder)
    LOG.debug('Glob path: {}'.format(glob_path))
    data_path = glob(glob_path, recursive=True)
    data_path = [ p.replace(root+'/', '') for p in data_path if not 'delete' in p]

    
    LOG.debug('results:')

    for p in data_path:
        LOG.debug('    {}'.format(p))

    return data_path


def grab_name(target, pattern='(.+)/monitor'):
    p = re.compile(pattern)

    result = p.match(target).group(1)

    LOG.debug('name: {}'.format(result))

    return result




__all__ = [
    get_template.__name__,
    new_content.__name__,
    new_trace.__name__,
    new_process.__name__,
    new_data.__name__,
    add_data.__name__,
    add_process.__name__,
    add_content.__name__,
    add_trace.__name__,
    grab_folders.__name__,
    grab_name.__name__,
    get_process.__name__,
]
