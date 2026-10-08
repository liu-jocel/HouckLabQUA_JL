import pickle
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q, meta_dict_to_config_6Q
from LabCode2.Config.make_meta_dict import make_meta_dict
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_amp_cal_1Q import AmpCal1Q
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_RB1Q import RB1Q
from LabCode2.Experiments.m_RB1Q_test import RB1Q2
from LabCode2.Experiments.m_RB1Q_baking import RB1Q_baked
from LabCode2.Experiments.m_amp_cal_2Q import AmpCal2Q
import numpy as np
import matplotlib.pyplot as plt

from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from LabCode2.Experiments.m_single_shot_1Q_3level import SingleShot1Q3Level


#
# # Get metaDict
# with open(pathPickle, 'rb') as f:
#     meta_dict = pickle.load(f)
#
meta_dict = make_meta_dict()
outer_folder = meta_dict["outer_folder"]
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'
while True:
    element_1 = 'q4'
    # element_2 = 'q4'
    Q1 = meta_dict[element_1]
    # Q2 = meta_dict[element_2]

    LOs = [Q1['q_LO'], Q1['r_LO']]

    # set_LO_parameters(LOs)

    config = meta_dict_to_config_6Q(meta_dict)

    # Single shot calibration
    shots = 1000

    SS_1 = SingleShot1Q(element_1, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='RB', suffix='Q1')
    # SS_2 = SingleShot1Q(element_1, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='RB', suffix='Q2')
    # SS_3level_1 = SingleShot1Q3Level(element_1, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='RB', suffix='Q1')
    # SS_3level_2 = SingleShot1Q3Level(element_1, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='RB', suffix='Q2')

    calib_params_1 = SS_1.calib_params
    # calib_params_2 = SS_2.calib_params
    # calib_params_3level_1 = SS_3level_1.calib_params
    # calib_params_3level_2 = SS_3level_2.calib_params# [[-8.90437439084053e-05, -1.8659409135580062e-05],
     # [-6.102825328707695e-05, -3.818803653120994e-05],
     # [-1.7578378319740295e-05, -1.8658202141523362e-05]]
    #
    m_vec = [8,] #16,32,64,128,256,512,1024
    num_ks = 1
    num_ns = 1
    # #
    RB_1 = RB1Q(element_1,m_vec,num_ks,num_ns,calib_params_1,path="RB",  outer_folder=outer_folder,meta_dict=meta_dict, interleaved=True)
    RB_1.run_RB()
    # RB_1.run_interleaved_RB()
    RB_1.analyze()
    RB_1.pickle_data()

# SS_1 = SingleShot1Q(element_1, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='RB', suffix='Q4')
# calib_params_1 = SS_1.calib_params
# RB_2 = RB1Q2(element_1,m_vec,num_ks,num_ns,calib_params_1,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
# RB_2.run_RB()
# RB_2.analyze()
# RB_2.pickle_data()
#
# # RB_1.simulate(RB_1.make_prog(simulate=True),['1','5'],20000,[[27000,29000]])
#
#
# # RB_1 = RB1Q('q1',num_ms,num_ks,num_ns,calib_params_1,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
# # RB_1.run_interleaved_RB()
# #
# RB_1 = RB1Q('q1',m_vec_orbit,num_ks,num_ns,calib_params_1,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
# RB_1.acquire_RB(interleaved=False)
# RB_1.analyze()
# RB_2 = RB1Q('q2',m_vec_orbit,num_ks,num_ns,calib_params_2,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
# RB_2.acquire_RB(interleaved=False)
# RB_2.analyze()
# #
# # RB_2_leak = RB1Q('q2',m_vec_orbit,num_ks,num_ns,calib_params_3level,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
# # RB_2_leak.acquire_RB(interleaved=False)
# # RB_2_leak.analyze_leak()
#
# m_vec = [4,8,16,32,64,128,256,512,1024]
# num_ks = 20
# num_ns = 100
#
# # calib_params = [[0,0],[1,1],[2,2]]
#
# pulse_params_1 = {'pulse_amp': meta_dict['q1']['pi_amp'],
#                 'pulse_len': 60,
#                 'DRAG_beta': 0}
#
# pulse_params_2 = {'pulse_amp': meta_dict['q2']['pi_amp'],
#                 'pulse_len': 60,
#                 'DRAG_beta': 0}
#
# RB_bake_1 = RB1Q_baked('q1',m_vec,num_ks,num_ns,calib_params_3level_1,pulse_params_1,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
#
# # RB_bake.amp_cal(0,0.5,20)
# # RB_bake.analyze_amp_cal()
#
# RB_bake_1.run_mk_programs()
# RB_bake_1.analyze()
#
# file_name = RB_bake_1.iname[:-4]+'_RB_data.pkl'
# with open(file_name,'wb') as f:
#     pickle.dump(RB_bake_1,f)
#
# RB_bake_2 = RB1Q_baked('q2',m_vec,num_ks,num_ns,calib_params_3level_2,pulse_params_2,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
#
# # RB_bake.amp_cal(0,0.5,20)
# # RB_bake.analyze_amp_cal()
#
# RB_bake_2.run_mk_programs()
# RB_bake_2.analyze()
#
# file_name = RB_bake_2.iname[:-4]+'_RB_data.pkl'
# with open(file_name,'wb') as f:
#     pickle.dump(RB_bake_2,f)
# #
# # ORBIT
# m_vec_orbit = [2,4, 8,16,32, 64,128,256, 512, 1024]
# num_ks = 1000
# num_ns = 1
# num_freqs = 11
# num_amps = 11
# freq_vec = meta_dict['q2']['q_LO']['LO_freq'] + 0.2e6 * np.linspace(-1, 1, num_freqs)
# amp_vec = meta_dict['q2']['pi_amp'] + 0.04 * np.linspace(-1, 1, num_amps)
# fidels = np.zeros([num_freqs,num_amps])
#
# for i,f in enumerate(freq_vec):
#     for j,a in enumerate(amp_vec):
#         print(i,j)
#         Q2['q_LO']['LO_freq'] = f
#         meta_dict['q2']['pi_amp'] = a
#         meta_dict['q2']['pi2_amp'] = a/2
#
#         LOs = [Q2['q_LO']]
#         set_LO_parameters(LOs)
#
#         RB_orb = RB1Q('q2', m_vec_orbit, num_ks, num_ns, calib_params_2, path="RB", outer_folder=outer_folder,
#                     meta_dict=meta_dict, plot=False)
#         RB_orb.acquire_RB(interleaved=False)
#         RB_orb.analyze()
#
#         fidels[i][j] = RB_orb.gate_fidel
#
#
# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# pcm1 = ax.pcolormesh(amp_vec, freq_vec, fidels)
# fig.colorbar(pcm1, ax=ax)
# ax.set_xlabel('Amplitude (V)')
# ax.set_ylabel('LO Frequency (GHz)')
# plt.savefig(RB_orb.iname[:-8]+'ORBIT', bbox_inches='tight')
#
# data_to_pickle = {'meta_dict': meta_dict,
#                   'freq_vec': freq_vec,
#                   'amp_vec': amp_vec,
#                   'fidels': fidels}
# with open(RB_orb.iname[:-8]+'ORBIT.pkl', 'wb') as f:
#     pickle.dump(data_to_pickle, f)
# #
# # ORBIT tuning
#
# m_vec = [2, 8, 64, 512, 1024]
# num_ks = 10
# num_ns = 1
#
# num_freqs = 3
# num_amps = 11
# freq_0 = meta_dict[element_2]['q_LO']['LO_freq']
# amp_0 = meta_dict[element_2]['pi_amp']
# freq_vec = freq_0 + 0.2e6 * np.linspace(-1, 1, num_freqs)
# amp_vec = amp_0 + 0.02 * np.linspace(-1, 1, num_amps)
# amp_fidels = np.zeros(num_amps)
# freq_fidels = np.zeros(num_freqs)
#
# for j,a in enumerate(amp_vec):
#
#     pulse_params = {'pulse_amp': a,
#                     'pulse_len': 40,
#                     'DRAG_beta': 0}
#
#     RB_orb = RB1Q_baked(element_2, m_vec, num_ks, num_ns, calib_params_3level_2, pulse_params, path="RB",
#                         outer_folder=outer_folder, meta_dict=meta_dict)
#     RB_orb.run_mk_programs()
#     RB_orb.analyze()
#
#     amp_fidels[j] = RB_orb.gate_fidel
#
# pulse_params = {'pulse_amp': amp_0,
#                 'pulse_len': 40,
#                 'DRAG_beta': 0}
#
# for i,f in enumerate(freq_vec):
#
#     Q2['q_LO']['LO_freq'] = f
#     LOs = [Q2['q_LO']]
#     set_LO_parameters(LOs)
#
#     RB_orb = RB1Q_baked(element_2, m_vec, num_ks, num_ns, calib_params_3level_2, pulse_params, path="RB",
#                         outer_folder=outer_folder, meta_dict=meta_dict)
#     RB_orb.run_mk_programs()
#     RB_orb.analyze()
#
#     freq_fidels[i] = RB_orb.gate_fidel
#
#
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# ax[0].plot(amp_vec, amp_fidels)
# ax[1].plot(freq_vec, freq_fidels)
# ax[0].set_xlabel('Amplitude (V)')
# ax[0].set_ylabel('Gate fidelity')
# ax[0].set_xlabel('LO Frequency (GHz)')
# ax[0].set_ylabel('Gate fidelity')
# plt.savefig(RB_orb.iname[:-8]+'ORBIT', bbox_inches='tight')
# #
# data_to_pickle = {'meta_dict': meta_dict,
#                   'freq_vec': freq_vec,
#                   'amp_vec': amp_vec,
#                   'amp_fidels': amp_fidels,
#                   'freq_fidels': freq_fidels}
# with open(RB_orb.iname[:-8]+'ORBIT.pkl', 'wb') as f:
#     pickle.dump(data_to_pickle, f)
# print(f)
#
# Q2['q_LO']['LO_freq'] = f
# LOs = [Q2['q_LO']]
# set_LO_parameters(LOs)

# DRAG orbit
# m_vec = [2, 8, 64, 256, 512]
# num_ks = 100
# num_ns = 10
# num_drags = 21-9
# drag_params = np.linspace(-0.65,1,num_drags)
# leak_pops = np.zeros(num_drags)
# leak_rates = np.zeros(num_drags)
# leak_rate_errs = np.zeros(num_drags)
# meta_dict['qc']['read_amp'] = 0
# meta_dict['q1']['read_amp'] = 0
#
# for i,d in enumerate(drag_params):
#
#     print(i)
#
#     # meta_dict['q2']['DRAG_beta'] = d
#
#     pulse_params = {'pulse_amp': Q2['pi_amp'],
#                     'pulse_len': int(meta_dict['global_params']['gauss_len']),
#                     'DRAG_beta': d}
#
#     RB_orb = RB1Q_baked('q2',m_vec,num_ks,num_ns,calib_params_3level_2,pulse_params,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
#     RB_orb.run_mk_programs()
#     RB_orb.analyze()
#     dat = RB_orb.data
#     with open(RB_orb.iname[:-8] + 'DRAG.pkl', 'wb') as f:
#         pickle.dump(dat, f)
#
#
#     leak_pops[i] = RB_orb.data[2][2][-1]
#     leak_rates[i] = RB_orb.leak_rate
#     leak_rate_errs[i] = RB_orb.leak_rate_err
#
#
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# ax[0].plot(drag_params,leak_pops)
# ax[0].set_xlabel('DRAG parameter')
# ax[0].set_ylabel('2 state pop. at 512 Cliffords')
# ax[1].errorbar(drag_params,leak_rates,yerr=leak_rate_errs)
# ax[1].set_xlabel('DRAG parameter')
# ax[1].set_ylabel('Leakage rate')
# plt.savefig(RB_orb.iname[:-8]+'ORBIT', bbox_inches='tight')
#
# data_to_pickle = {'meta_dict': meta_dict,
#                   'drag_vec': drag_params,
#                   'leak_pops': leak_pops,
#                   'leak_rates': leak_rates,
#                   'leak_rate_errs': leak_rate_errs,}
# with open(RB_orb.iname[:-8]+'ORBIT.pkl', 'wb') as f:
#     pickle.dump(data_to_pickle, f)

# #Q1
# with open(pathPickle, 'rb') as f:
#     meta_dict = pickle.load(f)
#
#
# m_vec = [2, 4, 8, 16, 32, 64, 256, 512, 1024]
# num_ks = 1000
# num_ns = 1
# num_drags = 21
# drag_params = np.linspace(-2,1,num_drags)
# leak_pops = np.zeros(num_drags)
# leak_rates = np.zeros(num_drags)
# leak_rate_errs = np.zeros(num_drags)
# meta_dict['qc']['read_amp'] = 0
# meta_dict['q2']['read_amp'] = 0
#
# for i,d in enumerate(drag_params):
#
#     print(i)
#
#     # meta_dict['q2']['DRAG_beta'] = d
#
#     pulse_params = {'pulse_amp': Q1['pi_amp'],
#                     'pulse_len': int(meta_dict['global_params']['gauss_len']),
#                     'DRAG_beta': d}
#
#     RB_orb = RB1Q_baked('q1',m_vec,num_ks,num_ns,calib_params_3level_1,pulse_params,path="RB", outer_folder=outer_folder,meta_dict=meta_dict)
#     RB_orb.run_mk_programs()
#     RB_orb.analyze()
#
#     leak_pops[i] = RB_orb.data[2][2][-1]
#     leak_rates[i] = RB_orb.leak_rate
#     leak_rate_errs[i] = RB_orb.leak_rate_err
#
#
# fig, ax = plt.subplots(2, 1, constrained_layout=True)
# ax[0].plot(drag_params,leak_pops)
# ax[0].set_xlabel('DRAG parameter')
# ax[0].set_ylabel('2 state pop. at 512 Cliffords')
# ax[1].errorbar(drag_params,leak_rates,yerr=leak_rate_errs)
# ax[1].set_xlabel('DRAG parameter')
# ax[1].set_ylabel('Leakage rate')
# plt.savefig(RB_orb.iname[:-8]+'ORBIT', bbox_inches='tight')
#
# data_to_pickle = {'meta_dict': meta_dict,
#                   'drag_vec': drag_params,
#                   'leak_pops': leak_pops,
#                   'leak_rates': leak_rates,
#                   'leak_rate_errs': leak_rate_errs,}
# with open(RB_orb.iname[:-8]+'ORBIT.pkl', 'wb') as f:
#     pickle.dump(data_to_pickle, f)
