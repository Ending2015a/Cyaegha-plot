import os
import re
import sys
import time
import logging

import logger

LOG = logger.getLogger('config', 'DEBUG')


from configs.beamrider_plot import beamrider_plot_factory, beamrider_plot_factory_v2
from configs.compare_plot import compare_plots_factory, compare_plots_factory_v2
from configs.seaquest_plot import seaquest_plot_factory_v2
from configs.spaceinvaders_plot import spaceinvaders_plot_factory, spaceinvaders_plot_factory_v2
from configs.kungfumaster_plot import kungfumaster_plot_factory, kungfumaster_plot_factory_v2
from configs.enduro_plot import enduro_plot_factory, enduro_plot_factory_v2
from configs.qbert_plot import qbert_plot_factory, qbert_plot_factory_v2
from configs.learning_curve_plot import learning_curve_plot_factory_v2
from configs.breakout_plot import breakout_plot_factory_v2

# old (v1)
'''
config = {
    'plot': [
        beamrider_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider',
                               trace_groups=['ori', 'dqn/top3'],
                               colors=['green', 'red'],
                               legends=['Primitive', 'Ours'],
                               title='Beamrider',
                               filename='outputs/Beamrider/final/beamriderl_split_ori_plot.png'),
        beamrider_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider',
                               trace_groups=['ev', 'dqn/top3'],
                               colors=['blue', 'red'],
                               legends=['Evolution', 'Ours'],
                               title='Beamrider',
                               filename='outputs/Beamrider/final/beamrider_split_ev_plot.png'),
        beamrider_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider',
                               trace_groups=['ev_ensemble', 'dqn/top3'],
                               colors=['blue', 'red'],
                               legends=['Evolution ensemble', 'Ours'],
                               title='Beamrider',
                               filename='outputs/Beamrider/final/beamrider_split_ev_ens_plot.png'),
        beamrider_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider',
                               trace_groups=['dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top3'],
                               colors=['green', 'blue', 'red'],
                               legends=['Macro action $m_2$', 'Macro action $m_1$', 'Ours'],
                               title='Beamrider',
                               filename='outputs/Beamrider/final/beamrider_split_dia_plot.png'),
        beamrider_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider',
                               trace_groups=['sequence', 'repeat', 'dqn/top3'],
                               colors=['green', 'blue', 'red'],
                               legends=['Most frequent action sequence', 'Action repeat', 'Ours'],
                               title='Beamrider',
                               filename='outputs/Beamrider/final/beamrider_split_rep_seq_plot.png'),

        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['ori', 'dqn/top3'],
                              colors=['green', 'red'],
                              legends=['Primitive', 'Ours'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/seaquest_split_ori_plot.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['ev', 'dqn/top3'],
                              colors=['blue', 'red'],
                              legends=['Evolution', 'Ours'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/seaquest_split_ev_plot.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['ev_ensemble', 'dqn/top3'],
                              colors=['blue', 'red'],
                              legends=['Evolution ensemble', 'Ours'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/seaquest_split_ev_ens_plot.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top3'],
                              colors=['brown', 'green', 'blue', 'red'],
                              legends=[r'Macro action $m_3$', r'Macro action $m_2$', r'Macro action $m_1$', 'Ours'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/seaquest_split_dia_plot.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['sequence', 'repeat', 'dqn/top3'],
                              colors=['green', 'blue', 'red'],
                              legends=['Most frequent action sequence', 'Action repeat', 'Ours'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/seaquest_split_rep_seq_plot.png'),
        
        

        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['ori', 'dqn/top1'],
                                   colors=['green', 'red'],
                                   legends=['Primitive', 'Ours'],
                                   title='Space Invaders',
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_split_ori_plot.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['ev', 'dqn/top1'],
                                   colors=['blue', 'red'],
                                   legends=['Evolution', 'Ours'],
                                   title='Space Invaders',
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_split_ev_plot.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['ev_ensemble', 'dqn/top1'],
                                   colors=['blue', 'red'],
                                   legends=['Evolution ensemble', 'Ours'],
                                   title='Space Invaders',
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_split_ev_ens_plot.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top1'],
                                   colors=['brown', 'green', 'blue', 'red'],
                                   legends=[r'Macro action $m_3$', r'Macro action $m_2$', r'Macro action $m_1$', 'Ours'],
                                   title='Space Invaders',
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_split_dia_plot.png'),

        kungfumaster_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/KungFuMaster',
                                  trace_groups=['ori', 'dqn/top2'],
                                  colors=['green', 'red'],
                                  legends=['Primitive', 'Ours'],
                                  title='Kung-Fu Master',
                                  filename='outputs/KungFuMaster/final/kungfumaster_split_ori_plot.png'),
        kungfumaster_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/KungFuMaster',
                                  trace_groups=['ev', 'dqn/top2'],
                                  colors=['blue', 'red'],
                                  legends=['Evolution', 'Ours'],
                                  title='Kung-Fu Master',
                                  filename='outputs/KungFuMaster/final/kungfumaster_split_ev_plot.png'),
        kungfumaster_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/KungFuMaster',
                                  trace_groups=['ev_ensemble', 'dqn/top2'],
                                  colors=['blue', 'red'],
                                  legends=['Evolution ensemble', 'Ours'],
                                  title='Kung-Fu Master',
                                  filename='outputs/KungFuMaster/final/kungfumaster_split_ev_ens_plot.png'),

        
        
    ]
}
'''

'''
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/Seaquest',
                              trace_groups=['ev_ensemble', 'dqn/top3'],
                              colors=['blue', 'red'],
                              legends=['Evolution ensemble', 'Ours'],
                              filename='outputs/Seaquest/seaquest_plot_2.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/Seaquest',
                              trace_groups=['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top3'],
                              colors=['brown', 'green', 'blue', 'red'],
                              legends=['Skill 3', 'Skill 2', 'Skill 1', 'Ours'],
                              filename='outputs/Seaquest/seaquest_plot_3.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/Seaquest',
                              trace_groups=['sequence', 'repeat', 'dqn/top3'],
                              colors=['green', 'blue', 'red'],
                              legends=['Most frequent sequence', 'Most frequent repetition', 'Ours'],
                              filename='outputs/Seaquest/seaquest_plot_4.png'),
        learning_curve_plot_factory(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/BeamRider/dqn/BeamRiderNoFrameskip_macro[3,3]_r0.05_v1_dqn/macro/macro.txt',
                                    title='Beamrider',
                                    filename='outputs/Beamrider/beamrider_learning_curve_plot.png'),
        learning_curve_plot_factory(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Seaquest/dqn/SeaquestNoFrameskip_macro[3,3]_r0.1_v0_dqn/macro/macro.txt',
                                    title='Seaquest',
                                    filename='outputs/Seaquest/seaquest_learning_curve_plot.png'),
        learning_curve_plot_factory(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/SpaceInvaders/dqn/SpaceInvadersNoFrameskip_macro[3,3]_r0.05_v0_dqn/macro/macro.txt',
                                    title='Space Invaders',
                                    filename='outputs/SpaceInvaders/spaceinvaders_learning_curve_plot.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['sequence', 'repeat', 'dqn/top1'],
                                   colors=['green', 'blue', 'red'],
                                   legends=['Most frequent sequence', 'Most frequent repetition', 'Ours'],
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_plot_1.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['ev_ensemble', 'dqn/top1'],
                                   colors=['blue', 'red'],
                                   legends=['Evolution ensemble', 'Ours'],
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_plot_2.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top1'],
                                   colors=['brown', 'green', 'blue', 'red'],
                                   legends=['Skill 3', 'Skill 2', 'Skill 1', 'Ours'],
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_plot_3.png'),
        '''
'''
config = {
    'plot': [

        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['/home/allenchen/Moana_proj/neural_skill_search/log/other/efficient/Seaquest/dqn/top1', '/home/allenchen/Moana_proj/neural_skill_search/log/other/efficient/Seaquest/dqn/top2', '/home/allenchen/Moana_proj/neural_skill_search/log/other/efficient/Seaquest/dqn/top3', 'ori', 'dqn/top3'],
                              colors=['brown', 'yellow', 'blue', 'green', 'red'],
                              legends=['eff-top3', 'eff-top2', 'eff-top1', 'Primitive', 'Ours'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/efficient/seaquest_split_ori_plot.png'),
        ]
}
'''




'''
            compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/BeamRider/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3', 'top4'],
                         filename='outputs/compare/BeamRider/compare_plot.png',
                         scale=4,
                         ydtick=1000) + 
            compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Enduro/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3'],
                         filename='outputs/compare/Enduro/compare_plot.png',
                         scale=4,
                         ydtick=200) + 
            compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/KungFuMaster/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3', 'top4'],
                         filename='outputs/compare/KungFuMaster/compare_plot.png',
                         scale=4,
                         ydtick=10000) + 
            compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Qbert/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3', 'top4'],
                         filename='outputs/compare/Qbert/compare_plot.png',
                         scale=4,
                         ydtick=2000) +
            compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Breakout/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3', 'top4'],
                         filename='outputs/compare/Breakout/compare_plot.png',
                         scale=4,
                         ydtick=100)
}
'''

'''
config = {
    'plot': #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/BeamRider/dqn/a2c',
            #                      groups=['repeat', 'seaquence', 'dqn_diassemble', 'new_ev_ensemble'],
            #                      filename='outputs/compare/BeamRider/v1/compare_plot.png',
            #                      ydtick=1000)+
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/KungFuMaster/dqn/a2c',
            #                      groups=['repeat', 'seaquence'],
            #                      filename='outputs/compare/KungFuMaster/v1/compare_plot.png',
            #                      ydtick=10000)+
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Qbert/dqn/a2c',
            #                      groups=['repeat', 'seaquence', 'dqn_diassemble', 'new_ev_ensemble'],
            #                      filename='outputs/compare/Qbert/v1/compare_plot.png',
            #                      ydtick=2000)+
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Seaquest/dqn/a2c',
            #                      groups=['top1', 'top2', 'top3', 'top4', 'ori', 'ev', 'repeat', 'seaquence'],
            #                      filename='outputs/compare/Seaquest/v1/compare_plot.png',
            #                      ydtick=1000)+
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Breakout/dqn/a2c',
            #                      groups=['top1', 'top2', 'top3', 'top4', 'ori', 'ev', 'repeat', 'seaquence'],
            #                      filename='outputs/compare/Breakout/v1/compare_plot.png',
            #                      ydtick=100)+
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Enduro/dqn/a2c',
            #                      groups=['top1', 'top2', 'top3', 'top4', 'ori', 'ev', 'repeat', 'seaquence', 'dqn_diassemble', 'new_ev_ensemble'],
            #                      filename='outputs/compare/Enduro/v1/compare_plot.png',
            #                      ydtick=200)+
            compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/SpaceInvaders/dqn/a2c',
                                  groups=['top1', 'top2', 'top3', 'top4', 'ori', 'ev', 'repeat', 'seaquence'],
                                  filename='outputs/compare/SpaceInvaders/v1/compare_plot.png',
                                  ydtick=250)

}
'''


'''
config = {
    'plot':[
        beamrider_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider',
                               trace_groups=['dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top3'],
                               colors=['green', 'blue', 'red'],
                               legends=[r'$\text{Macro action } m_2$', r'$\text{Macro action } m_1$', r'$\text{Ours}$'],
                               title='Beamrider',
                               filename='outputs/Beamrider/final/beamrider_split_dia_plot.png'),
        seaquest_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Seaquest',
                              trace_groups=['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top3'],
                              colors=['brown', 'green', 'blue', 'red'],
                              legends=[r'$\text{Macro action } m_3$', r'$\text{Macro action } m_2$', r'$\text{Macro action } m_1$', r'$\text{Ours}$'],
                              title='Seaquest',
                              filename='outputs/Seaquest/final/seaquest_split_dia_plot.png'),
        spaceinvaders_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/SpaceInvaders',
                                   trace_groups=['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first', 'dqn/top1'],
                                   colors=['brown', 'green', 'blue', 'red'],
                                   legends=[r'$\text{Macro action } m_3$', r'$\text{Macro action } m_2$', r'$\text{Macro action } m_1$', r'$\text{Ours}$'],
                                   title='Space Invaders',
                                   filename='outputs/SpaceInvaders/final/spaceinvaders_split_dia_plot.png'),
    ]
}
'''
'''
config = {
    'plot': [
        enduro_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Enduro',
                               trace_groups=['ori', 'dqn/top3'],
                               colors=['green', 'red'],
                               legends=['Primitive', 'Ours'],
                               title='Enduro',
                               filename='outputs/Enduro/final/enduro_split_ori_plot.png'),
        enduro_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Enduro',
                               trace_groups=['ev', 'dqn/top3'],
                               colors=['blue', 'red'],
                               legends=['Evolution', 'Ours'],
                               title='Enduro',
                               filename='outputs/Enduro/final/enduro_split_ev_plot.png'),
        enduro_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Enduro',
                               trace_groups=['sequence', 'repeat', 'dqn/top3'],
                               colors=['green', 'blue', 'red'],
                               legends=['Most frequent action sequence', 'Action repeat', 'Ours'],
                               title='Enduro',
                               filename='outputs/Enduro/final/enduro_split_rep_seq_plot.png'),

        qbert_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Qbert',
                               trace_groups=['ev', 'ori'],
                               colors=['blue', 'red'],
                               legends=['Evolution', 'Ours (Primitive)'],
                               title='Qbert',
                               filename='outputs/Qbert/final/qbert_split_ev_plot.png'),
        qbert_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/Qbert',
                               trace_groups=['sequence', 'repeat', 'ori'],
                               colors=['green', 'blue', 'red'],
                               legends=['Most frequent action sequence', 'Action repeat', 'Ours (Primitive)'],
                               title='Qbert',
                               filename='outputs/Qbert/final/qbert_split_rep_seq_plot.png'),
    ]
}
'''
'''
config = {
    'plot': [
        
        learning_curve_plot_factory(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/KungFuMaster/dqn/KungFuMasterNoFrameskip_macro[3,3]_r0.02_v1_dqn/macro/macro.txt',
                                    title='Kung-Fu Master',
                                    filename='outputs/KungFuMaster/kungfumaster_learning_curve_plot.png'),
        learning_curve_plot_factory(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Enduro/dqn/EnduroNoFrameskip_macro[3,3]_r0.002_v0_dqn/macro/macro.txt',
                                    title='Enduro',
                                    filename='outputs/Enduro/enduro_learning_curve_plot.png'),
        learning_curve_plot_factory(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Qbert/dqn/QbertNoFrameskip_macro[3,3]_r0.05_v0_dqn/macro/macro.txt',
                                    title='Qbert',
                                    filename='outputs/Qbert/qbert_learning_curve_plot.png'),
    ]
}
'''

'''
config = {
    'plot': [
        kungfumaster_plot_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/KungFuMaster',
                                  trace_groups=['sequence', 'repeat', 'dqn/top2'],
                                  colors=['green', 'blue', 'red'],
                                  legends=['Most frequent action sequence', 'Action repeat', 'Ours'],
                                  title='Kung-Fu Master',
                                  filename='outputs/KungFuMaster/final/kungfumaster_split_rep_seq_plot.png'),
    ]
}
'''






# 2019.11.11 (compare v2)
'''
config = {
    'plot': compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Seaquest/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3', 'top4', 'repeat', 'seaquence'],
                         filename='outputs/compare/Seaquest/v2/compare_plot.png',
                         scale=4,
                         ydtick=1000) + 
            compare_plots_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/SpaceInvaders/dqn/a2c',
                         groups=['ev', 'ori', 'top1', 'top2', 'top3', 'top4', 'repeat', 'seaquence'],
                         filename='outputs/compare/SpaceInvaders/v2/compare_plot.png',
                         scale=4,
                         ydtick=250)
}
'''


# 2019.11.11 (v2)
'''
config = {
    'plot': [
        beamrider_plot_factory_v2(),
        kungfumaster_plot_factory_v2(),
        qbert_plot_factory_v2()
    ]
}
'''


# 2019.11.12 (compare v1)
'''
config = {
    'plot': compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/BeamRider/dqn/a2c',
                                  groups=['dqn_diassemble', 'ev_ensemble'],
                                  filename='outputs/compare/BeamRider/v1/compare_plot.png',
                                  ydtick=1000)+
            compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/KungFuMaster/dqn/a2c',
                                  groups=['dqn_diassemble', 'ev_ensemble'],
                                  filename='outputs/compare/KungFuMaster/v1/compare_plot.png',
                                  ydtick=10000)+
            compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Qbert/dqn/a2c',
                                  groups=['dqn_diassemble', 'ev_ensemble'],
                                  filename='outputs/compare/Qbert/v1/compare_plot.png',
                                  ydtick=2000)+
            compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Seaquest/dqn/a2c',
                                  groups=['dqn_diassemble', 'ev_ensemble'],
                                  filename='outputs/compare/Seaquest/v1/compare_plot.png',
                                  ydtick=1000)+
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Breakout/dqn/a2c',
            #                      groups=['top1', 'top2', 'top3', 'top4', 'ori', 'ev', 'repeat', 'seaquence'],
            #                      filename='outputs/compare/Breakout/v1/compare_plot.png',
            #                      ydtick=100)+
            compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Enduro/dqn/a2c',
                                  groups=['top5'],
                                  filename='outputs/compare/Enduro/v1/compare_plot.png',
                                  ydtick=200)
            #compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/SpaceInvaders/dqn/a2c',
            #                      groups=['top1', 'top2', 'top3', 'top4', 'ori', 'ev', 'repeat', 'seaquence'],
            #                      filename='outputs/compare/SpaceInvaders/v1/compare_plot.png',
            #                      ydtick=250)

}
'''

# 2019.11.13 (v2+oc)
'''
config = {
    'plot': [
        beamrider_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/BeamRider',
                                  trace_groups=['oc', 'dqn/a2c/top3'],
                                  colors=['blue', 'red'],
                                  legends=['Option-critic', 'Ours'],
                                  title='Beamrider',
                                  filename='outputs/Beamrider/v2/beamrider_plot_oc.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=[3, 1],
                                  dash=['20px,20px', 'solid']),
        seaquest_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/Seaquest',
                                  trace_groups=['oc', 'dqn/a2c/top4'],
                                  colors=['blue', 'red'],
                                  legends=['Option-critic', 'Ours'],
                                  title='Seaquest',
                                  filename='outputs/Seaquest/v2/seaquest_plot_oc.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=[3, 1],
                                  dash=['20px,20px', 'solid']),
        enduro_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/Enduro',
                                  trace_groups=['oc', 'dqn/a2c/top1'],
                                  colors=['blue', 'red'],
                                  legends=['Option-critic', 'Ours'],
                                  title='Enduro',
                                  filename='outputs/Enduro/v2/enduro_plot_oc.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=[3, 1],
                                  dash=['20px,20px', 'solid']),
        qbert_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/Qbert',
                                  trace_groups=['oc', 'dqn/a2c/top2'],
                                  colors=['blue', 'red'],
                                  legends=['Option-critic', 'Ours'],
                                  title='Qbert',
                                  filename='outputs/Qbert/v2/qbert_plot_oc.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=[3, 1],
                                  dash=['20px,20px', 'solid']),
        spaceinvaders_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/SpaceInvaders',
                                  trace_groups=['oc', 'dqn/a2c/top2'],
                                  colors=['blue', 'red'],
                                  legends=['Option-critic', 'Ours'],
                                  title='Space Invaders',
                                  filename='outputs/SpaceInvaders/v2/spaceinvaders_plot_oc.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=[3, 1],
                                  dash=['20px,20px', 'solid']),
        breakout_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/Breakout',
                                trace_groups=['oc', 'dqn/a2c/top4'],
                                colors=['blue', 'red'],
                                legends=['Option-critic', 'Ours'],
                                title='Breakout',
                                filename='outputs/Breakout/v2/breakout_plot_oc.png',
                                max_steps=10000000,
                                scale=4,
                                process=[3, 1],
                                dash=['20px,20px', 'solid']),
    ]
}'''

# 2019/11/13 (space v2 only)
'''
config = {
    'plot': [
        spaceinvaders_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/SpaceInvaders',
                                  trace_groups=['oc', 'dqn/a2c/top2'],
                                  colors=['blue', 'red'],
                                  legends=['Option-critic', 'Ours'],
                                  title='Space Invaders',
                                  filename='outputs/SpaceInvaders/v2/spaceinvaders_plot_oc.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=[3, 1],
                                  dash=['20px,20px', 'solid']),
    ]
}
'''

# 2019.11.13 (compare v1 breakout, )
'''
config = {
    'plot': compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/Breakout/dqn/a2c',
                                  groups=['dqn_diassemble', 'ev_ensemble'],
                                  filename='outputs/compare/Breakout/v1/compare_plot.png',
                                  ydtick=100)+
            compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3_life/SpaceInvaders/dqn/a2c',
                                  groups=['dqn_diassemble', 'ev_ensemble'],
                                  filename='outputs/compare/SpaceInvaders/v1/compare_plot.png',
                                  ydtick=250)
}
'''

# 2019.11.14 (conf a2c v1)
'''
config = {
    'plot': []
}

envs = ['BeamRider', 'KungFuMaster', 'Qbert', 'Enduro', 'Seaquest', 'SpaceInvaders', 'Breakout']
factories = [beamrider_plot_factory_v2,
             kungfumaster_plot_factory_v2,
             qbert_plot_factory_v2,
             enduro_plot_factory_v2,
             seaquest_plot_factory_v2,
             spaceinvaders_plot_factory_v2,
             breakout_plot_factory_v2]
titles = ['Beamrider', 'Kung-Fu Master', 'Q*bert', 'Enduro', 'Seaquest', 'Space Invaders', 'Breakout']
top_group = ['top2', 'top4', 'top1', 'top1', 'top4', 'top3', 'top4']

trace_groups = [['ori', 'ev'],
               ['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first'],
               ['ev_ensemble'],
               ['seaquence',  'repeat']]
legend_group = [['Primitive', 'Evolution'],
                ['Macro action 3', 'Macro action 2', 'Macro action 1'],
                ['Evolution ensemble'],
                ['Most frequent', 'Action repeat']]

colors = ['brown', 'green', 'blue', 'red']
#markers = ['triangle','square','cross','circle']


for env, factory, title, top in zip(envs, factories, titles, top_group):
    LOG.info('env: {} / factory: {} / title: {} / top: {}'.format(env, factory.__name__, title, top))

    for idx, (trace_group, legend) in enumerate(zip(trace_groups, legend_group)):

        if env == 'Enduro':
            if idx == 2 or idx == 1:
                continue
        if idx != 3:
            continue

        tg = trace_group+[top]
        c = colors[len(colors)-(len(trace_group)+1):]
        lg = legend+['Ours']
        fn = 'outputs/{}/a2c_onelife/{}_plot_vs_{}.png'.format(env, env.lower(), '_'.join(trace_group).replace('/', '_'))
        #d = dash[len(dash)-(len(trace_group)+1):]
        #mk = markers[len(markers)-(len(trace_group)+1):]

        LOG.info('    trace: {}'.format(tg))
        LOG.info('    color: {}'.format(c))
        LOG.info('    legen: {}'.format(lg))
        LOG.info('    filen: {}'.format(fn))
        #LOG.info('    marke: {}'.format(mk))

        config['plot'].append(factory(
            root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3_life/{}/dqn/a2c'.format(env),
            trace_groups=tg,
            colors=c,
            legends=lg,
            title=title,
            filename=fn,
            max_steps=10000000,
            scale=4,
            process=1,
            #marker=mk,
            #dash=d,
        ))
'''


# 2019.11.14 (conf ppo v1)

config = {
    'plot': []
}

envs = ['BeamRider', 'KungFuMaster', 'Enduro', 'Seaquest', 'SpaceInvaders']
factories = [beamrider_plot_factory_v2,
             kungfumaster_plot_factory_v2,
             enduro_plot_factory_v2,
             seaquest_plot_factory_v2,
             spaceinvaders_plot_factory_v2]
titles = ['Beamrider', 'Kung-Fu Master', 'Enduro', 'Seaquest', 'Space Invaders']
top_group = ['dqn/top3', 'dqn/top2', 'dqn/top3', 'dqn/top3', 'dqn/top1']

trace_groups = [['ori'], 
                ['ev'],
                ['dqn_diassemble/third', 'dqn_diassemble/second', 'dqn_diassemble/first'],
                ['ev_ensemble'],
                ['seaquence',  'repeat']]
legend_group = [['Primitive'], 
                ['Evolution'],
                ['Macro action 3', 'Macro action 2', 'Macro action 1'],
                ['Evolution ensemble'],
                ['Most frequent', 'Action repeat']]

colors = ['brown', 'green', 'blue', 'red']
#markers = ['triangle','square','cross','circle']


for env, factory, title, top in zip(envs, factories, titles, top_group):
    LOG.info('env: {} / factory: {} / title: {} / top: {}'.format(env, factory.__name__, title, top))

    for idx, (trace_group, legend) in enumerate(zip(trace_groups, legend_group)):

        #if env != 'BeamRider':
        #    continue

        #if idx > 0:
        #    continue



        if env == 'BeamRider':
            if idx == 2: # dia only 2
                trace_group = trace_group[1:]
                legend = legend[1:]
        elif env == 'Enduro':
            if idx == 2 or idx == 3: #no dia no ev_ensemble
                continue


        # legend position
        if env == 'Seaquest' or env == 'KungFuMaster':
            legend_pos = 'bottom-right'
        else:
            legend_pos = 'top-left'


        tg = trace_group+[top]
        c = colors[len(colors)-(len(trace_group)+1):]
        #c = [colors[1], colors[3]]
        lg = legend+['Ours']
        fn = 'outputs/{}/ppo/{}_plot_vs_{}.png'.format(env, env.lower(), '_'.join(trace_group).replace('/', '_'))
        #d = dash[len(dash)-(len(trace_group)+1):]
        #mk = markers[len(markers)-(len(trace_group)+1):]

        LOG.info('    trace: {}'.format(tg))
        LOG.info('    color: {}'.format(c))
        LOG.info('    legen: {}'.format(lg))
        LOG.info('    filen: {}'.format(fn))
        #LOG.info('    marke: {}'.format(mk))

        config['plot'].append(factory(
            root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/{}'.format(env),
            trace_groups=tg,
            colors=c,
            legends=lg,
            title=title,
            filename=fn,
            max_steps=10000000,
            scale=4,
            process=1,
            legend_pos=legend_pos,
            #marker=mk,
            #dash=d,
        ))



# 2019.11.15 (seaquest v4)
'''
config = {
    'plot': [
        seaquest_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/Seaquest',
                                  trace_groups=['1_3_v2/third', '1_3_v2/second', '1_3_v2/first', 'ori'],
                                  colors=['brown', 'green', 'blue', 'red'],
                                  legends=['Macro action 3 (ours)', 'Macro action 2 (ours)', 'Macro action 1 (ours)', 'Primitive'],
                                  title='Seaquest',
                                  filename='outputs/Seaquest/1_3/spaceinvaders_plot_ori_vs_1_3_v2.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=1,
                                  dtick=500),
        seaquest_plot_factory_v2(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/20M_top3/Seaquest',
                                  trace_groups=['1_3_v2/third', '1_3_v2/second', '1_3_v2/first', '1_3_ensemble_v2'],
                                  colors=['brown', 'green', 'blue', 'red'],
                                  legends=['Macro action 3 (ours)', 'Macro action 2 (ours)', 'Macro action 1 (ours)', 'Ensemble (ours)'],
                                  title='Seaquest',
                                  filename='outputs/Seaquest/1_3/spaceinvaders_plot_1_3_vs_1_3_en_v2.png',
                                  max_steps=10000000,
                                  scale=4,
                                  process=1,
                                  dtick=500),
    ]
}
'''


# 2019.11.15 (learning curve)
'''
config = {
    'plot': [
        learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Seaquest/dqn/SeaquestNoFrameskip_macro[3,3]_r0.1_v0_dqn/macro/macro.txt',
                                    title='Seaquest',
                                    filename='outputs/Seaquest/seaquest_learning_curve_plot.png'),
        learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/BeamRider/dqn/BeamRiderNoFrameskip_macro[3,3]_r0.05_v1_dqn/macro/macro.txt',
                                    title='Beamrider',
                                    filename='outputs/BeamRider/beamrider_learning_curve_plot.png'),
        learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/SpaceInvaders/dqn/SpaceInvadersNoFrameskip_macro[3,3]_r0.05_v0_dqn/macro/macro.txt',
                                    title='Space Invaders',
                                    filename='outputs/SpaceInvaders/spaceinvaders_learning_curve_plot.png'),

        learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/KungFuMaster/dqn/KungFuMasterNoFrameskip_macro[3,3]_r0.02_v1_dqn/macro/macro.txt',
                                    title='Kung-Fu Master',
                                    filename='outputs/KungFuMaster/kungfumaster_learning_curve_plot.png'),
        learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Enduro/dqn/EnduroNoFrameskip_macro[3,3]_r0.002_v0_dqn/macro/macro.txt',
                                    title='Enduro',
                                    filename='outputs/Enduro/enduro_learning_curve_plot.png'),
        learning_curve_plot_factory_v2(path='/home/allenchen/Moana_proj/neural_skill_search/log/controller/Qbert/dqn/QbertNoFrameskip_macro[3,3]_r0.05_v0_dqn/macro/macro.txt',
                                    title='Qbert',
                                    filename='outputs/Qbert/qbert_learning_curve_plot.png'),
    ]
}
'''

# 2019.11.15 (Beamrider comp)
'''
config = {
    'plot': compare_plots_factory(root='/home/allenchen/Moana_proj/neural_skill_search/log/other/final_20M_top3/BeamRider/',
                                  groups=['dqn/top3'],
                                  filename='outputs/compare/BeamRider/ppo/compare_plot_2.png',
                                  ydtick=1000)
}
'''