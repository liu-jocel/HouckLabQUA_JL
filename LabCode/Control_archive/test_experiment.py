import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_T1 import T1
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_ZZ import ZZ
from LabCode2.Experiments.m_T2E import T2E
from LabCode2.Experiments.m_transmission import Transmission
from LabCode2.Experiments.m_qubit_spec import QubitSpec
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ6B2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# YOKOGS200.main(0.52)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
element = Q2
config = meta_dict_to_config_2Q(meta_dict)

LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

# f_min = 90e6
# f_max = 110e6
# f_step = 0.1e6
# shots = 100
# #
# # exp = Transmission('q1', f_min, f_max, f_step, shots, meta_dict=meta_dict,outer_folder=outer_folder, path='Transmission', suffix='')
# exp = Transmission('q2', f_min, f_max, f_step, shots, meta_dict=meta_dict,outer_folder=outer_folder, path='Transmission', suffix='')


# #
# meta_dict[element['q_name']]['r_IF'] = 102.5e6 #exp.min_f
# config = meta_dict_to_config_2Q(meta_dict)

# yokovec = np.arange(0,2,0.005)
# yokovec = np.around(yokovec,decimals=4)
# num_yokos = len(yokovec)
# freqs = np.zeros(num_yokos)

# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# ax.plot(yokovec, freqs)
# ax.set_xlabel('Yoko voltage (V)')
# ax.set_ylabel('Resonator IF (Hz)')
# ax.set_title('R1 Transmission vs. Flux')
# plt.savefig('Y:/Charlie/Data/TwoQubit/TQ5A2/flux.png', bbox_inches="tight")
# plt.close(fig)



# Qubit spec
#
# f_min = 50e6
# f_max = 150e6
# f_step = 1e6
# shots = 100
#
# exp = QubitSpec('q1', f_min, f_max, f_step, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='Spectroscopy', suffix='')
# exp = QubitSpec('q2', f_min, f_max, f_step, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='Spectroscopy', suffix='')

# print("Max F = ", exp.max_f)
# meta_dict[element['q_name']]['q_IF'] = 101.2e6
# meta_dict[element['q_name']]['q_IF'] = exp.max_f
# config = meta_dict_to_config_2Q(meta_dict)


# Rabi
#
a_min = 0.02
a_max = 0.5
num_a = 50
shots = 500
# #
# exp = Rabi('q1', a_min, a_max, num_a, shots, None, meta_dict=meta_dict, outer_folder=outer_folder, path='Rabi', suffix='')
exp = Rabi('q2', a_min, a_max, num_a, shots, None, meta_dict=meta_dict, outer_folder=outer_folder, path='Rabi', suffix='')

###################################################
# Chevron
###################################################
# shots = 100
#
# # SS_1 = SingleShot1Q('q1', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q1')
# SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')
#
# # calib_params_1 = SS_1.calib_params
# calib_params_2 = SS_2.calib_params
# num_freqs = 21
# LO_freqs_1 = Q1['q_LO']['LO_freq'] + np.linspace(-3e6,3e6,num_freqs)
# LO_freqs_2 = Q2['q_LO']['LO_freq'] + np.linspace(-50e6,50e6,num_freqs)
#
# chevron_data_1 = np.zeros([num_freqs,num_a])
# chevron_data_2 = np.zeros([num_freqs,num_a])
#
# start_time = time.time()
#
# for i in range(num_freqs):
#
#     print(i, time.time()-start_time)
#     # Q1['q_LO']['LO_freq'] = LO_freqs_1[i]
#     Q2['q_LO']['LO_freq'] = LO_freqs_2[i]
#     set_LO_parameters([Q1['q_LO'],Q2['q_LO']])
#
#     # exp_1 = Rabi('q1', a_min, a_max, num_a, shots, None, meta_dict=meta_dict, outer_folder=outer_folder, path='Rabi', suffix='',
#     #              plot=False)
#     exp_2 = Rabi('q2', a_min, a_max, num_a, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path='Rabi', suffix='',
#                plot=False, ss=True)
#
#     # chevron_data_1[i] = exp_1.data[1]
#     chevron_data_2[i] = exp_2.data[1]
#
# fig, ax = plt.subplots(1,2,constrained_layout=True)
# # pcm1 = ax[0].pcolormesh(exp_1.amp_vec, LO_freqs_1, chevron_data_1)
# pcm1 = ax[1].pcolormesh(exp_2.amp_vec, LO_freqs_2, chevron_data_2)
# fig.colorbar(pcm1, ax=ax)
# plt.savefig(exp_2.iname[:-8]+'chevron', bbox_inches='tight')
###################################################

# Single shot calibration
#
shots = 1000

# SS_1 = SingleShot1Q('q1', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q1')
SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')

# calib_params_1 = SS_1.calib_params
calib_params_2 = SS_2.calib_params

# T1
t_min = 1e-6
t_max = 200e-6
num_t = 30
shots = 1000

# T1_1 = T1('q1', t_min, t_max, num_t, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q1')
T1_2 = T1('q2', t_min, t_max, num_t, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q2')

# T2R
t_min = 100e-9
t_max = 0.5e-6
t_max_2 = 0.5e-6
num_t = 50
shots = 1000
detuning = 4e6

# T2R_1 = T2R('q1', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q1',freq_cal=True)
T2R_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q2',freq_cal=True)

# Q1['q_LO']['LO_freq'] = Q1['q_LO']['LO_freq'] - T2R_1.freq_correction
Q2['q_LO']['LO_freq'] = Q2['q_LO']['LO_freq'] - T2R_2.freq_correction


LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO']]
# set_LO_parameters(LOs)

# ZZ_1 = ZZ('q1', 'q2', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q1')
ZZ_2 = ZZ('q2', 'q1', t_min, t_max, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q2')

t_min = 100e-9
t_max = 40e-6
t_max_2 = 40e-6
num_t = 40
shots = 1000
detuning = 0.2e6

# T2R_1 = T2R('q1', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q1',freq_cal=False)
T2R_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q2',freq_cal=False)

# T2E
t_min = 1e-6
t_max = 200e-6
t_max_2 = 200e-6
num_t = 40
shots = 1000

# T2E_1 = T2E('q1', t_min, t_max, num_t, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q1')
T2E_2 = T2E('q2', t_min, t_max_2, num_t, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix='Q2')



start_time = time.time()
times = []
T1_1s = []
T1_2s = []
T2R_1s = []
T2R_2s = []
T2E_1s = []
T2E_2s = []
freq_1 = []
freq_2 = []


for i in range(1000):

    try:

        t_min = 1e-6
        t_max = 200e-6
        num_t = 30
        shots = 1000

        T1_1 = T1('q1', t_min, t_max, num_t, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                  path='1Q_Char', suffix='Q1', fit=False, plot=False)
        T1_2 = T1('q2', t_min, t_max, num_t, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
                  path='1Q_Char', suffix='Q2', fit=False, plot=False)

        # freq cal
        t_min = 100e-9
        t_max = 0.5e-6
        t_max_2 = 0.5e-6
        num_t = 50
        shots = 1000
        detuning = 4e6

        FC_1 = T2R('q1', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict,
                    outer_folder=outer_folder, path='1Q_Char', suffix='Q1', freq_cal=True, fit=False, plot=False)
        FC_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict,
                    outer_folder=outer_folder, path='1Q_Char', suffix='Q2', freq_cal=True, fit=False, plot=False)

        ZZ_1 = ZZ('q1', 'q2', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict,
                  outer_folder=outer_folder, path='1Q_Char', suffix='Q1', fit=False, plot=False)
        ZZ_2 = ZZ('q2', 'q1', t_min, t_max, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict,
                  outer_folder=outer_folder, path='1Q_Char', suffix='Q2', fit=False, plot=False)

        t_min = 100e-9
        t_max = 40e-6
        t_max_2 = 40e-6
        num_t = 40
        shots = 1000
        detuning = 0.2e6

        T2R_1 = T2R('q1', t_min, t_max, num_t, shots, detuning, calib_params_1, meta_dict=meta_dict,
                    outer_folder=outer_folder, path='1Q_Char', suffix='Q1', freq_cal=False, fit=False, plot=False)
        T2R_2 = T2R('q2', t_min, t_max_2, num_t, shots, detuning, calib_params_2, meta_dict=meta_dict,
                    outer_folder=outer_folder, path='1Q_Char', suffix='Q2', freq_cal=False, fit=False, plot=False)

        # T2E
        t_min = 1e-6
        t_max = 200e-6
        t_max_2 = 200e-6
        num_t = 40
        shots = 1000

        T2E_1 = T2E('q1', t_min, t_max, num_t, shots, calib_params_1, meta_dict=meta_dict, outer_folder=outer_folder,
                    path='1Q_Char', suffix='Q1', fit=False, plot=False)
        T2E_2 = T2E('q2', t_min, t_max_2, num_t, shots, calib_params_2, meta_dict=meta_dict, outer_folder=outer_folder,
                    path='1Q_Char', suffix='Q2', fit=False, plot=False)

        times.append(time.time()-start_time)
        T1_1s.append(T1_1.data)
        T1_2s.append(T1_2.data)
        T2R_1s.append(T2R_1.data)
        T2R_2s.append(T2R_2.data)
        T2E_1s.append(T2E_1.data)
        T2E_2s.append(T2E_2.data)
        freq_1.append(FC_1.data)
        freq_2.append(FC_2.data)

        data = {'times': times,
                'T1_1s': T1_1s,
                'T1_2s': T1_2s,
                'T2R_1s': T2R_1s,
                'T2R_2s': T2R_2s,
                'T2E_1s': T2E_1s,
                'T2E_2s': T2E_2s,
                'freq_1': freq_1,
                'freq_2': freq_2}

        with open(outer_folder + 'stability/Yoko1p120_1.pkl','wb') as f:
            pickle.dump(data,f)

    except:
        print('Error in sweep, moving on')