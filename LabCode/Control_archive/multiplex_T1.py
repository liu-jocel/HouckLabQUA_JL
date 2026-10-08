import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_6Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_T1 import T1
from LabCode2.Experiments.m_T1_2q import T1_2Q
from LabCode2.Experiments.m_T1_2q_bg import T1_2Q_shots
from LabCode2.Experiments.m_T1_2q_noPulse import T1_2Qnp
from LabCode2.Experiments.m_T2R_2q import T2R_2Q
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_ZZ import ZZ
from LabCode2.Experiments.m_T2E import T2E
from LabCode2.Experiments.m_transmission import Transmission
from LabCode2.Experiments.m_qubit_spec import QubitSpec
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_Rabi_time import Rabi_time
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q

from qm.qua import *
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Z://t1Team/Data/2024-08-09_cooldown/TATQ01-si_01_qm/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# YOKOGS200.main(0.52)

qubit_list = ['q6', 'q4']#, 'q3', 'q4', 'q5', 'q6']
### for pickling data - cheap hack
pickled_data = []
start = time.time_ns()
pickled_data.append(time.ctime(time.time()))
file_name = outer_folder + 'PairWiseCor/pairwisecor.pkl'
###

# while True:



# for q in qubit_list:

    ### for pickling data - cheap hack

    ###

# Q = meta_dict[q]
# path = '2Q_Char',#qubit_list[0] + '_' + qubit_list[1]
config = meta_dict_to_config_6Q(meta_dict)

Q1 = meta_dict[qubit_list[0]]
Q2 = meta_dict[qubit_list[1]]


LOs = [Q1['q_LO'], Q2['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

# qubits = [Q['q_name']]

do_readout_cal = 1
do_transmission = 0
do_qubit_spec = 0
do_rabi_IQ = 0
do_rabi_SS = 0
do_rabi_time = 0
do_fine_tune = 0
do_T1_2Q = 1
do_T2R = 0
do_T2E = 0
do_ZZ = 0
do_T1_sweep = 0

# Single shot parameters
shots_SS = [100]

# Transmission parameters
# f_min_trans = [30e6, 130e6, 230e6]
# f_max_trans = [70e6, 170e6, 270e6]
f_min_trans = [28.5e6,]
f_max_trans = [31e6,]
f_step_trans = [0.1e6,]
shots_trans = [100,]

# Qubit spectroscopy parameters
# resonator = meta_dict['q1']['r_name']
# freq = 30.1e6
# update_frequency(resonator, freq)
f_min_spec = [-60e6]
f_max_spec = [-40e6]
f_step_spec = [0.1e6]
shots_spec = [200]

# Rabi parameters
# t_min_rabi = [16e-9,]
# t_max_rabi = [4000e-9,]
# num_t_rabi = [451]
a_min_rabi = [0.001]
a_max_rabi = [0.4]
num_a_rabi = [15]
freq_range_rabi = [0e6] # 150
num_freqs_rabi = [1] # 41
shots_rabi = [200]

# T1 parameters
# t_min_t1 = [1e-6]
# t_max_t1 = [4000e-6]  #800 800 80
# num_t_t1 = [30] #30
# shots_t1 = [100] #1000

# T2R parameters
t_min_t2r = [100e-9]
t_max_t2r = [400e-6]
num_t_t2r = [200]
shots_t2r = [500]
detuning_t2r = [.005e6]

# T2E
t_min_t2e = [1e-6]
t_max_t2e = [500e-6]
num_t_t2e = [25]
shots_t2e = [2000]

# ZZ
t_min_ZZ = 100e-9
t_max_ZZ = 0.5e-6
num_t_ZZ = 50
shots_ZZ = 500
detuning_ZZ = 4e6

# shots_SS = {
#     "q1": 1000,
#     "q2": 1000
# }

# pickled_data = []
# start = time.time()
# pickled_data.append(time.ctime(time.time()))
# file_name = outer_folder + 'PairWiseCor/sscalibparams.pkl'
###


# while True:
#     data_dict = {}
#
#     data_dict['timestamp'] = time.time() - start
#
#     shots_SS = 1000  # 1000
#     if do_readout_cal:
#
#         calib_params = {}
#         # for i, q in enumerate(qubit_list):
#         #     # Q = meta_dict[q]
#         #     # LOs = [Q['r_LO']]
#         #     # set_LO_parameters(LOs)
#         #     SS = SingleShot1Q(q, shots_SS, meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char',
#         #                       suffix=q)
#         #     calib_params[q] = SS.calib_params
#
#         # Q = meta_dict[qubit_list[0]]
#         # LOs = [Q['r_LO']]
#         # set_LO_parameters(LOs)
#         SS_2Q = SingleShot2Q(qubit_list[0], qubit_list[1], shots_SS, meta_dict=meta_dict, outer_folder=outer_folder,
#                              path='2Q_Char', suffix='2Q')
#
#         calib_params['2Q'] = SS_2Q.calib_params
#         confusion = SS_2Q.confusion
#         data_dict['fidelity'] = SS_2Q.fidelity
#         data_dict['calib_params'] = SS_2Q.calib_params
#
#         # ## for pickling data - cheap hack
#         # data_dict['Cor_380'] = exp.data
#             ###
#         pickled_data.append(data_dict)
#         with open(file_name, 'wb') as f:
#             pickle.dump(pickled_data, f)
        ##


pickled_data = []
start = time.time()
pickled_data.append(time.ctime(time.time()))
file_name = outer_folder + 'PairWiseCor/test.pkl'

# tau_list = [200e-6, 250e-6, 300e-6, 350e-6, 400e-6, 450e-6, 500e-6, 550e-6, 600e-6, 650e-6, 700e-6, 750e-6, 800e-6, 850e-6, 900e-6, 950e-6, 1000e-6]

tau_list = [1e-6, 2e-6, 4e-6, 8e-6, 16e-6, 32e-6, 64e-6, 128e-6, 256e-6]#, 512e-6, 1024e-6]
# tau_list = [380e-6]#, 380e-6, 380e-6]
for tau in tau_list:
# while True:
    data_dict = {}
    data_dict['timestamp'] = time.time() - start
    data_dict['tau'] = tau

    shots_SS = 1000  # 1000
    if do_readout_cal:

        calib_params = {}
        for i, q in enumerate(qubit_list):
            # Q = meta_dict[q]
            # LOs = [Q['r_LO']]
           #  # set_LO_parameters(LOs)
            SS = SingleShot1Q(q, shots_SS, meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char',
                              suffix=q)
            calib_params[q] = SS.calib_params

        # Q = meta_dict[qubit_list[0]]
        # LOs = [Q['r_LO']]
       #  # set_LO_parameters(LOs)
        SS_2Q = SingleShot2Q(qubit_list[0], qubit_list[1], shots_SS, meta_dict=meta_dict, outer_folder=outer_folder,
                             path='2Q_Char', suffix='2Q')

        calib_params['2Q'] = SS_2Q.calib_params
        confusion = SS_2Q.confusion

        data_dict['fidelity'] = SS_2Q.fidelity
        data_dict['calib_params'] = SS_2Q.calib_params

        # ## for pickling data - cheap hack
        # data_dict['Cor_380'] = exp.data
            ###




    t_min_t1 = [tau]
    t_max_t1 = [tau]  #800 800 80
    num_t_t1 = [1] #30
    shots_t1 = [100000] #1000

    set_LO_parameters([meta_dict[qubit_list[0]]['q_LO']])
    set_LO_parameters([meta_dict[qubit_list[1]]['q_LO']])

    # for i in range(100):
        # data_dict['timestamp'] = time.time_ns() - start
        # print(i, (time.time_ns() - start)*1E-9)
    start_time = time.time_ns()
    if do_T1_2Q:
            # time.sleep(30)

        exp = T1_2Q_shots(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
                 meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix= '_' + str(tau*1e6) + '_us_' +str(shots_t1[0]) +'T2R', fit= 0,
                 save=True, plot = 0, sscor = 1)  # calib_params[i]

        end_time = time.time_ns() - start_time
        print("total time elapsed is", end_time * 1E-9)
        print("Tau is", tau)
        data_dict['bitstrings_t2r'] = exp.data


    # if do_T1_2Q:
    #         # time.sleep(30)
    #
    #     exp = T1_2Qnp(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
    #              meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix=str(tau*1e6) + 's_' + str(shots_t1[0]) + 'shots_nopulse', fit=0,
    #              save=True, plot = 0, sscor = True)  # calib_params[i]
    #
    #     data_dict['bitstrings_bg'] = exp.data
    #
    #
    # if do_T1_2Q:
    #         # time.sleep(30)
    #
    #     exp = T2R_2Q(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
    #              meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix=str(tau*1e6) + 's_' + str(shots_t1[0]) + 'shots_pib2_pulse', fit=0,
    #              save=True, plot = 0, sscor = True)  # calib_params[i]
    #
    #     data_dict['bitstrings_t2r'] = exp.data



    pickled_data.append(data_dict)
    with open(file_name, 'wb') as f:
        pickle.dump(pickled_data, f)
#
# t_min_t1 = [280e-6]
# t_max_t1 = [280e-6]  #800 800 80
# num_t_t1 = [1] #30
# shots_t1 = [1500000] #1000
#
# set_LO_parameters([meta_dict[qubit_list[0]]['q_LO']])
# set_LO_parameters([meta_dict[qubit_list[1]]['q_LO']])
#
# # for i in range(100):
#     # data_dict['timestamp'] = time.time_ns() - start
#     # print(i, (time.time_ns() - start)*1E-9)
# start_time = time.time_ns()
#
# if do_T1_2Q:
#         # time.sleep(30)
#
#     exp = T1_2Q(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
#              meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix="T1_280", fit=0,
#              save=True, plot = 0, sscor = True)  # calib_params[i]
# end_time = time.time_ns() - start_time
# print("total time for 280 us elapsed is:", end_time*1E-9)
#
#
# t_min_t1 = [180e-6]
# t_max_t1 = [180e-6]  #800 800 80
# num_t_t1 = [1] #30
# shots_t1 = [1500000] #1000
#
# set_LO_parameters([meta_dict[qubit_list[0]]['q_LO']])
# set_LO_parameters([meta_dict[qubit_list[1]]['q_LO']])
#
# # for i in range(100):
#     # data_dict['timestamp'] = time.time_ns() - start
#     # print(i, (time.time_ns() - start)*1E-9)
# start_time = time.time_ns()
#
# if do_T1_2Q:
#         # time.sleep(30)
#
#     exp = T1_2Q(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
#              meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix="T1_180", fit=0,
#              save=True, plot = 0, sscor = True)  # calib_params[i]
# end_time = time.time_ns() - start_time
# print("total time for 180 us elapsed is:", end_time*1E-9)
#
# t_min_t1 = [580e-6]
# t_max_t1 = [580e-6]  #800 800 80
# num_t_t1 = [1] #30
# shots_t1 = [1500000] #1000
#
# set_LO_parameters([meta_dict[qubit_list[0]]['q_LO']])
# set_LO_parameters([meta_dict[qubit_list[1]]['q_LO']])
#
# # for i in range(100):
#     # data_dict['timestamp'] = time.time_ns() - start
#     # print(i, (time.time_ns() - start)*1E-9)
# start_time = time.time_ns()
#
# if do_T1_2Q:
#         # time.sleep(30)
#
#     exp = T1_2Q(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
#              meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix="T1_580", fit=0,
#              save=True, plot = 0, sscor = True)  # calib_params[i]
# end_time = time.time_ns() - start_time
# print("total time for 580 us elapsed is:", end_time*1E-9)


# t_min_t1 = [180e-6]
# t_max_t1 = [180e-6]  #800 800 80
# num_t_t1 = [1] #30
# shots_t1 = [1500000] #1000
#
# set_LO_parameters([meta_dict[qubit_list[0]]['q_LO']])
# set_LO_parameters([meta_dict[qubit_list[1]]['q_LO']])
#
# # for i in range(100):
#     # data_dict['timestamp'] = time.time_ns() - start
#     # print(i, (time.time_ns() - start)*1E-9)
# start_time = time.time_ns()
#
# if do_T1_2Q:
#         # time.sleep(30)
#
#     exp = T1_2Q(qubit_list[0],qubit_list[1], t_min_t1[0], t_max_t1[0], num_t_t1[0], shots_t1[0], calib_params['2Q'],
#              meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix="T1", fit=0,
#              save=True, plot = 0, sscor = True)  # calib_params[i]
# end_time = time.time_ns() - start_time
# print("total time for 180 us elapsed is:", end_time*1E-9)

    ### for pickling data - cheap hack
    # data_dict['Cor_380'] = exp.data
    #     ###
    # pickled_data.append(data_dict)
    # with open(file_name, 'wb') as f:
    #     pickle.dump(pickled_data, f)
    ###

# if do_readout_cal:
#     time.sleep(10)
#     calib_params = []
#     for i,q in enumerate(qubits):
#         SS = SingleShot1Q(q, shots_SS[i], meta_dict=meta_dict, outer_folder=outer_folder, path=path,
#                             suffix="read_cal")
#         calib_params.append(SS.calib_params)

if do_transmission:
    for i,q in enumerate(qubits):
        exp = Transmission(q, f_min_trans[i], f_max_trans[i], f_step_trans[i], shots_trans[i], meta_dict=meta_dict, outer_folder=outer_folder,
                           path=path, suffix="trans")

if do_qubit_spec:
    for i,q in enumerate(qubits):
        set_LO_parameters([meta_dict[q]['q_LO']])
        exp = QubitSpec(q, f_min_spec[i], f_max_spec[i], f_step_spec[i], shots_spec[i], meta_dict=meta_dict, outer_folder=outer_folder,
                        path=path, suffix="spec")


if do_rabi_IQ or do_rabi_SS:
    for i,q in enumerate(qubits):
        set_LO_parameters([meta_dict[q]['q_LO']])
        chevron_data = np.zeros([num_freqs_rabi[i],num_a_rabi[i]])
        LO_freqs = meta_dict[q]['q_LO']['LO_freq'] + np.linspace(-freq_range_rabi[i]/2,freq_range_rabi[i]/2,num_freqs_rabi[i])
        start_time = time.time()

        for j in range(num_freqs_rabi[i]):

            print(j, time.time()-start_time)
            meta_dict[q]['q_LO']['LO_freq'] = LO_freqs[j]
            set_LO_parameters([meta_dict[q]['q_LO']])

            if num_freqs_rabi[i] == 1: # Use native plotting in experiment if you just want a line cut
                if do_rabi_IQ:
                    exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], None, meta_dict=meta_dict, outer_folder=outer_folder,
                           path=path, suffix="rabi_IQ")
                elif do_rabi_SS:
                    exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], calib_params[i], meta_dict=meta_dict, outer_folder=outer_folder,
                           path=path, suffix="rabi_SS", ss=True)
            else: # Do plotting here if you want a chevron
                if do_rabi_IQ:
                    exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], None, meta_dict=meta_dict,
                             outer_folder=outer_folder, path=path, suffix='rabi_IQ', plot=False)
                elif do_rabi_SS:
                    exp = Rabi(q, a_min_rabi[i], a_max_rabi[i], num_a_rabi[i], shots_rabi[i], calib_params[i], meta_dict=meta_dict,
                             outer_folder=outer_folder, path=path, suffix='rabi_SS', plot=False, ss=True)

                if j==0:
                    name = exp.iname[:-8]

                chevron_data[j] = exp.data[1]
                if num_freqs_rabi[i] > 1:
                    fig, ax = plt.subplots(1,1,constrained_layout=True)
                    pcm1 = ax.pcolormesh(exp.amp_vec, LO_freqs, chevron_data)
                    fig.colorbar(pcm1, ax=ax)
                    plt.savefig(name+'chevron', bbox_inches='tight')
                    plt.close()

# if do_rabi_time:
#     for i,q in enumerate(qubits):
#         set_LO_parameters([meta_dict[q]['q_LO']])
#         chevron_data = np.zeros([num_freqs_rabi[i],num_t_rabi[i]])
#         LO_freqs = meta_dict[q]['q_LO']['LO_freq'] + np.linspace(-freq_range_rabi[i]/2,freq_range_rabi[i]/2,num_freqs_rabi[i])
#         start_time = time.time()
#
#         for j in range(num_freqs_rabi[i]):
#
#             print(j, time.time()-start_time)
#             meta_dict[q]['q_LO']['LO_freq'] = LO_freqs[j]
#             set_LO_parameters([meta_dict[q]['q_LO']])
#
#             if num_freqs_rabi[i] == 1: # Use native plotting in experiment if you just want a line cut
#                 exp = Rabi_time(q, t_min_rabi[i], t_max_rabi[i], num_t_rabi[i], shots_rabi[i], calib_params[i], meta_dict=meta_dict, outer_folder=outer_folder,
#                        path='1Q_Char', suffix=q)
#             else: # Do plotting here if you want a chevron
#                 exp = Rabi_time(q, t_min_rabi[i], t_max_rabi[i], num_t_rabi[i], shots_rabi[i], calib_params[i], meta_dict=meta_dict,
#                          outer_folder=outer_folder, path='1Q_Char', suffix='', plot=False)
#                 if j==0:
#                     name = exp.iname[:-8]
#
#                 chevron_data[j] = exp.data[1]
#                 if num_freqs_rabi[i] > 1:
#                     fig, ax = plt.subplots(1,1,constrained_layout=True)
#                     pcm1 = ax.pcolormesh(exp.t_vec, LO_freqs, chevron_data)
#                     fig.colorbar(pcm1, ax=ax)
#                     plt.savefig(name+'chevron', bbox_inches='tight')
#                     plt.close()
#         with open(exp.pname, 'wb') as f:
#                 pickle.dump(chevron_data, f)
#




if do_T2R:
    for i,q in enumerate(qubits):
        set_LO_parameters([meta_dict[q]['q_LO']])
        exp = T2R(q, t_min_t2r[i], t_max_t2r[i], num_t_t2r[i], shots_t2r[i], detuning_t2r[i], calib_params[i],
                  meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T2R",freq_cal=False, save=True)


if do_T2E:
    for i,q in enumerate(qubits):
        set_LO_parameters([meta_dict[q]['q_LO']])
        exp = T2E(q, t_min_t2e[i], t_max_t2e[i], num_t_t2e[i], shots_t2e[i], calib_params[i],
                          meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T2E", save=True)
#         ### for pickling data - cheap hack
#         data_dict[q + '_T2E'] = exp.data
# pickled_data.append(data_dict)
# with open(file_name, 'wb') as f:
#     pickle.dump(pickled_data, f)
# ###

if do_ZZ:
    for i,q in enumerate(qubits):
        exp = ZZ(qubits[i],qubits[(i+1)%2],t_min_ZZ,t_max_ZZ,num_t_ZZ,shots_ZZ,detuning_ZZ,calib_params[i],
                 outer_folder=outer_folder,path=path, suffix="ZZ", meta_dict=meta_dict, plot=True, fit=True, save=True)

t1_amps = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
if do_T1_sweep:
    for a in t1_amps:
        meta_dict['c1']['pulse_amp'] = a
        for i,q in enumerate(qubits):
            set_LO_parameters([meta_dict[q]['q_LO']])
            exp = T1(q, t_min_t1[i], t_max_t1[i], num_t_t1[i], shots_t1[i], calib_params[i],
                     meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix="T1"+f"_amp_{a}", save=True)
