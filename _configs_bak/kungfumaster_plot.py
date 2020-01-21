import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_factory import *

from matplotlib.colors import to_hex
import plotly.graph_objects as go



def kungfumaster_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/KungFuMaster',
                              trace_groups=['ori', 'ev', 'dqn/top3'],
                              colors=['green', 'blue', 'red'],
                              legends=['Primitive', 'Evolution', 'Ours'],
                              title='Seaquest',
                              filename='outputs/KungFuMaster/kungfumaster_plot.png',
                              scale=4):

    template = get_template()

    template['filename'] = filename
    #template['width'] = width
    #template['height'] = height
    template['scale'] = scale
    template['subplots']['layout']['title']['text'] = title
    template['subplots']['layout']['xaxis']['range'] = [0, 10000000]
    template['subplots']['layout']['yaxis']['dtick'] = 10000

    content = new_content(title=title)

    add_content(template, content)

    # for each group, grab all folders under this group, then add them to sources
    for idx, group in enumerate(trace_groups):
        group_root = os.path.join(root, group)
        sources = grab_folders(group_root)
        '''
        add_data(config=template,
                 root=group_root,
                 sources=sources, 
                 name=group,
                 process='process.load_results', 
                 inputs={
                     'max_steps': 10000000,
                     'smooth_window': 200
                 })
        '''
        process = new_process(process_name='process.load_results',
                                  inputs={
                                      'root': group_root,
                                      'sources': sources,
                                      'max_steps': 10000000,
                                      'smooth_window': 200
                                  })

        data = new_data(name=group)

        add_process(data, process)
        add_data(template, data)

        color = 'id'
        if colors is not None:
            color = colors[idx]

        legend = '{}-{}'.format(title, group)
        if legends is not None:
            legend = legends[idx]

        trace = new_trace(group, legend_name=legend, color=color, fillopacity=0.2)
        
        add_trace(content, trace)
    
    return template


def kungfumaster_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/KungFuMaster/dqn/a2c',
                           trace_groups=['ori', 'ev', 'top3'],
                           colors=['green', 'blue', 'red'],
                           legends=['Primitive', 'Evolution', 'Ours'],
                           title='Kung-Fu Master',
                           filename='outputs/KungFuMaster/v2/kungfumaster_plot.png',
                           max_steps=10000000,
                           scale=4,
                           **kwargs):
    '''
    kwargs:
        process: (list) process number, 1, 2, or 3, which refer to 'load_results', 'load_period_eval' and 'load_option_critic', respectively
        dash: (list) dash style name, please refer to plotly documentation for more information.
            Sets the dash style of lines. Set to a dash type string ("solid", "dot", "dash", "longdash", "dashdot", or "longdashdot") or a dash length list in px (eg "5px,10px,2px,2px").
    '''

    template = get_template(**kwargs)

    template['filename'] = filename
    #template['width'] = width
    #template['height'] = height
    template['scale'] = scale
    template['subplots']['layout']['title']['text'] = title
    template['subplots']['layout']['xaxis']['range'] = [0, max_steps]
    template['subplots']['layout']['yaxis']['dtick'] = 10000
    template['subplots']['layout']['yaxis']['rangemode'] = 'tozero'

    content = new_content(title=title)

    add_content(template, content)

    # for each group, grab all folders under this group, then add them to sources
    for idx, group in enumerate(trace_groups):
        group_root = os.path.join(root, group)
        

        if 'process' in kwargs:
            if isinstance(kwargs['process'], (list, tuple)):
                process_number = kwargs['process'][idx]
            else:
                process_number = kwargs['process']
        else:
            process_number = 2
            

        if process_number == 1:
            sources = grab_folders(group_root)
        elif process_number == 2:
            sources = grab_folders(group_root, target_folder='**/period_eval')
        elif process_number == 3:
            files = grab_folders(group_root, target_folder='**/training_results_ep_life.csv')
            sources = [grab_name(f, pattern='(.+)/training_results_ep_life.csv') for f in files]

        process = get_process(process=process_number, max_steps=max_steps, sources=sources, root=group_root)

        process = new_process(**process)

        data = new_data(name=group)

        add_process(data, process)
        add_data(template, data)

        color = 'id'
        if colors is not None:
            color = colors[idx]

        legend = '{}-{}'.format(title, group)
        if legends is not None:
            legend = legends[idx]

        # line
        line = {}
        if 'dash' in kwargs:
            dash = kwargs['dash']

            if not isinstance(data, (tuple, list)):
                line['dash'] = dash
            else:
                line['dash'] = dash[idx]

        trace = new_trace(group, legend_name=legend, color=color, fillopacity=0.2, line=line)
        
        add_trace(content, trace)
    
    return template     



# def kungfumaster_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/KungFuMaster/dqn/a2c',
#                            trace_groups=['ori', 'ev', 'top4'],
#                            colors=['green', 'blue', 'red'],
#                            legends=['Primitive', 'Evolution', 'Ours'],
#                            title='Kung-Fu Master',
#                            filename='outputs/KungFuMaster/v2/kungfumaster_plot.png',
#                            scale=4):

#     template = get_template()

#     template['filename'] = filename
#     #template['width'] = width
#     #template['height'] = height
#     template['scale'] = scale
#     template['subplots']['layout']['title']['text'] = title
#     template['subplots']['layout']['xaxis']['range'] = [0, 10000000]
#     template['subplots']['layout']['yaxis']['dtick'] = 10000
#     template['subplots']['layout']['yaxis']['rangemode'] = 'tozero'

#     content = new_content(title=title)

#     add_content(template, content)

#     # for each group, grab all folders under this group, then add them to sources
#     for idx, group in enumerate(trace_groups):
#         group_root = os.path.join(root, group)
#         sources = grab_folders(group_root, target_folder='**/period_eval')
#         '''
#         add_data(config=template,
#                  root=group_root,
#                  sources=sources, 
#                  name=group,
#                  process='process.load_results', 
#                  inputs={
#                      'max_steps': 10000000,
#                      'smooth_window': 200
#                  })
#         '''

#         process = new_process(process_name='process.load_period_eval',
#                                   inputs={
#                                       'root': group_root,
#                                       'sources': sources,
#                                       'max_steps': 10000000,
#                                       'smooth_window': 20,
#                                   })

#         data = new_data(name=group)

#         add_process(data, process)
#         add_data(template, data)

#         color = 'id'
#         if colors is not None:
#             color = colors[idx]

#         legend = '{}-{}'.format(title, group)
#         if legends is not None:
#             legend = legends[idx]

#         trace = new_trace(group, legend_name=legend, color=color, fillopacity=0.2)
        
#         add_trace(content, trace)
    
#     return template     



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