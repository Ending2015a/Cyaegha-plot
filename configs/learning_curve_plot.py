import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_factory import *

from matplotlib.colors import to_hex
import plotly.graph_objects as go



def learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Seaquest/dqn/SeaquestNoFrameskip_macro[3,3]_r0.1_v0_dqn/macro/macro.txt',
                                color=to_hex('xkcd:red'),
                                legend='Ensemble search',
                                title='Seaquest',
                                filename='outputs/seaquest_skill_plot.png',
                                scale=4,
                                ydtick=None,
                                **kwargs):

    template = get_template(**kwargs)

    template['filename'] = filename
    template['scale'] = scale
    template['subplots']['layout']['title']['text'] = title
    #template['subplots']['layout']['xaxis']['range'] = [0, ]
    template['subplots']['layout']['yaxis']['dtick'] = ydtick
    template['subplots']['layout']['annotations'][0]['text'] = 'Generations'

    content = new_content(title=title, show_legend=False)

    add_content(template, content)

    process = new_process(process_name='process.load_skill_rewards',
                         inputs={
                             'path': path,
                             'to_dataframe': True,
                             'smoothing': True,
                             'smooth_window': 40,
                             'drop_nan': True
                         })

    data = new_data(name=legend)

    add_process(data, process)
    add_data(template, data)

    color_ = 'id'
    if color is not None:
        color_ = color

    legend_ = '{}'.format(title)
    if legend is not None:
        legend_ = legend

    trace = new_trace(legend, legend_name=legend_, color=color_, x_column='generations')
    
    add_trace(content, trace)

    
    return template
        

        



'''
beamrider_plot_config = {
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
                'text': 'BeamRider',
                'font.size': 32,
                'xref': 'paper',
                'x': 0.5,
            },
            'template': 'plotly_white',
            'font.size': 22,
            'legend': {
                'font.size': 22,
                'x': 1.01,
                'y': 1.0,
                'tracegroupgap': 0
            },
            'margin': {
                'l': 80,
                'r': 10,
                't': 80,
                'b': 100
            },
            'xaxis': {
                'tick0': 0,
                'tickformat': 's',
                
            },
            'yaxis': {
                'tick0': 0,
                'tickformat': 's',
            },
            'annotations': [
                go.layout.Annotation(
                    x=0.5,
                    y=-0.10,
                    showarrow=False,
                    text='Timesteps',
                    xref='paper',
                    yref='paper',
                    font=dict(size=22)
                ),
                go.layout.Annotation(
                    x=-0.05,
                    y=0.5,
                    showarrow=False,
                    textangle=-90,
                    text='Rewards',
                    xref='paper',
                    yref='paper',
                    font=dict(size=22)
                ),
                
            ],
        },
        'contents': [
            {
                'title': 'BeamRider',
                'type': 'confidence',
                'traces': [
                    {
                        'data': 'BeamRider-ori',
                        'legend_name': 'BeamRider-ori',
                        'color': to_hex('xkcd:blue'),
                        'x_column': 'timesteps',
                        'y_column': 'rewards'
                    },
                    {
                        'data': 'BeamRider-top',
                        'legend_name': 'BeamRider-top',
                        'color': to_hex('xkcd:red'),
                        'x_column': 'timesteps',
                        'y_column': 'rewards'
                    }
                ]
            },
        ]
    },

    'data': {
        'BeamRider-top': {
            'process': [
                (
                    'process.load_results',
                    {
                        'inputs': {
                            'root': 'data/BeamRider/dqn/top1/BeamRiderNoFrameskip-v4',
                            'sources': [
                                '[[0, 1, 1], [3, 2]]/monitor',
                                '[[0, 1, 1], [3, 2]]_0/monitor',
                                '[[0, 1, 1], [3, 2]]_1/monitor'
                            ],
                            'combine_sources': True,
                            'to_dataframe': True,
                            'max_steps': 10000000,
                        }
                    }
                )
            ]
        },
        'BeamRider-ori': {
            'process': [
                (
                    'process.load_results',
                    {
                        'inputs': {
                            'root': 'data/BeamRider/dqn/ori/BeamRiderNoFrameskip-v4',
                            'sources': [
                                '[]/monitor',
                                '[]_0/monitor',
                                '[]_1/monitor'
                            ],
                            'combine_sources': True,
                            'to_dataframe': True,
                            'max_steps': 10000000,
                        }
                    }
                )
            ]
        },
    },

    'offline': True,
    'filename': 'beamrider_plot.png',
    'width': 1600,
    'height': 800
}

'''