import re
import os
import sys
import time
import logging

from glob import glob
import logger
LOG = logger.getLogger('compare_plot')

from matplotlib.colors import to_hex
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_factory import *
from pymap import Map
root = '/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/BeamRider/dqn'


def compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/Seaquest',
                         groups=['dqn/ev', 'dqn/top1', 'dqn/top2', 'dqn/top3', 'dqn/ori', 'repeat', 'seaquence'],
                         filename='outputs/compare/compare_plot.png',
                         scale=4,
                         ydtick=1000):
    configs = []

    if groups is None:
        groups = ['./']

    for group in groups:

        group_root = os.path.join(root, group)
        folders = grab_folders(group_root)
        
        template = get_template()

        
        _, ext = os.path.splitext(filename)
        postfix = '_{}{}'.format(group.replace('/', '.'), ext)
        comp_filename = filename.replace(ext, postfix)

        template['filename'] = comp_filename
        template['scale'] = scale
        template['subplots']['layout']['title']['text'] = group
        template['subplots']['layout']['xaxis']['range'] = [0, 10000000]
        template['subplots']['layout']['yaxis']['dtick'] = ydtick
        template['subplots']['layout']['yaxis']['rangemode'] = 'tozero'

        content = new_content(group)
        add_content(template, content)

        for idx, folder in enumerate(folders):

            name = grab_name(folder)

            LOG.info("{}th sources".format(idx+1))
            LOG.info("    path: {}".format(os.path.join(group_root, folder)))
            LOG.info("    name: {}".format(name))

            process = new_process(process_name='process.load_results',
                                  inputs={
                                      'root': group_root,
                                      'sources': [folder],
                                      'max_steps': 10000000,
                                      'smooth_window': 200
                                  })

            data = new_data(name=name)

            add_process(data, process)
            add_data(template, data)

            #add_data(template, group_root, [folder], name, inputs={'max_steps': 10000000, 'smooth_window': 200})
            
            trace = new_trace(name, legend_name=name, fillopacity=0.2)
            add_trace(content, trace)

        configs.append(template)

    return configs


def compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/BeamRider/dqn/a2c',
                           groups=['ori', 'ev', 'top3'],
                           filename='outputs/compare/compare_plot.png',
                           max_steps=10000000,
                           scale=4,
                           ydtick=1000,
                           **kwargs):
    '''
    kwargs:
        process: (list) process number, 1, 2, or 3, which refer to 'load_results', 'load_period_eval' and 'load_option_critic', respectively
        dash: (list) dash style name, please refer to plotly documentation for more information.
            Sets the dash style of lines. Set to a dash type string ("solid", "dot", "dash", "longdash", "dashdot", or "longdashdot") or a dash length list in px (eg "5px,10px,2px,2px").
    '''

    template = get_template()

    template['filename'] = filename
    #template['width'] = width
    #template['height'] = height
    template['scale'] = scale
    template['subplots']['layout']['title']['text'] = title
    template['subplots']['layout']['xaxis']['range'] = [0, max_steps]
    template['subplots']['layout']['yaxis']['dtick'] = 1000
    template['subplots']['layout']['yaxis']['rangemode'] = 'tozero'

    content = new_content(title=title)

    add_content(template, content)

    # for each group, grab all folders under this group, then add them to sources
    for idx, group in enumerate(trace_groups):
        group_root = os.path.join(root, group)
        

        if 'process' in kwargs:
            process_number = kwargs['process'][idx]
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


def compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Seaquest/dqn/a2c',
                         groups=['ori', 'top1', 'top2', 'top3', 'top4'],
                         filename='outputs/compare/compare_plot.png',
                         scale=4,
                         ydtick=1000,
                         **kwargs):
    configs = []

    if groups is None:
        groups = ['./']

    for group in groups:

        group_root = os.path.join(root, group)
        folders = grab_folders(group_root, '**/period_eval')

        if len(folders) == 0:
            LOG.error('No folder found')
        
        template = get_template(**kwargs)

        
        _, ext = os.path.splitext(filename)
        postfix = '_{}{}'.format(group.replace('/', '.'), ext)
        comp_filename = filename.replace(ext, postfix)

        template['filename'] = comp_filename
        template['scale'] = scale
        template['subplots']['layout']['title']['text'] = group
        template['subplots']['layout']['xaxis']['range'] = [0, 10000000]
        template['subplots']['layout']['yaxis']['dtick'] = ydtick
        template['subplots']['layout']['yaxis']['rangemode'] = 'tozero'

        content = new_content(group)
        add_content(template, content)

        for idx, folder in enumerate(folders):

            name = grab_name(folder, pattern='(.+)/period_eval')

            LOG.info("{}th sources".format(idx+1))
            LOG.info("    path: {}".format(os.path.join(group_root, folder)))
            LOG.info("    name: {}".format(name))

            process = new_process(process_name='process.load_period_eval',
                                  inputs={
                                      'root': group_root,
                                      'sources': [folder],
                                      'max_steps': 10000000,
                                      'smooth_window': 80,
                                  })

            data = new_data(name=name)

            add_process(data, process)
            add_data(template, data)

            #add_data(template, group_root, [folder], name, inputs={'max_steps': 10000000, 'smooth_window': 200})
            
            trace = new_trace(name, legend_name=name, fillopacity=0.2)
            add_trace(content, trace)

        configs.append(template)

    return configs
