import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_6Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_T1 import T1
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_ZZ import ZZ
from LabCode2.Experiments.m_T2E import T2E
from LabCode2.Experiments.m_transmission import Transmission
from LabCode2.Experiments.m_qubit_spec import QubitSpec
from LabCode2.Experiments.m_Rabi import Rabi
# from LabCode2.Experiments.m_Rabi_time import Rabi_1020
from LabCode2.Experiments.m_Rabi_fState import Rabi_fState
from LabCode2.Experiments.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Z://t1Team/Data/2024-12-22_cooldown/TATQ01_Topsil_01/OPX/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# YOKOGS200.main(0.52)

Q = meta_dict['q4']
config = meta_dict_to_config_6Q(meta_dict)

LOs = [Q['q_LO'], Q['r_LO']]

elements = ['q4']  #'q1'
element = "q4"
# set_LO_parameters(LOs)

# Rabi
#
t_min = 16e-9
t_max = 100e-9
num_t = 20
a_min = 0.02
a_max = 0.5
num_a = 40
shots = 500


freq_12= -119e6

# rabi = Rabi_fState(element, a_min, a_max, num_a, shots, calib_params, outer_folder=outer_folder, path='Rabi', meta_dict=meta_dict,ss=True,confusion=confusion)
# #
# rabi = Rabi_1020(element, freq_12, t_min, t_max, num_t, shots, None, outer_folder=outer_folder, path='Rabi', meta_dict=meta_dict)
# rabi = Rabi_1020(element, freq_12, t_min, t_max, num_t, shots, None, outer_folder=outer_folder, path='Rabi', meta_dict=meta_dict)
#
rabi = Rabi_fState(element, a_min, a_max, num_a, shots, None, ss=False, outer_folder=outer_folder, path='Rabi', meta_dict=meta_dict)
# rabi = Rabi_fState(element, a_min, a_max, num_a, shots, None, outer_folder=outer_folder, path='Rabi', meta_dict=meta_dict)

##################################################
# Chevron
##################################################
for element in elements:
    with open(pathPickle, 'rb') as f:
        meta_dict = pickle.load(f)

    freq_12 = meta_dict[element]['f_state_IF']
    SS = SingleShot1Q3Level(element, shots, outer_folder=outer_folder, meta_dict=meta_dict,path='Rabi')
    calib_params = SS.calib_params
    confusion = SS.confusion
    num_freqs = 41
    IF_freqs_12 = freq_12 + np.linspace(-4e6,4e6,num_freqs)

    chevron_data_2 = np.zeros([num_freqs,num_a])

    start_time = time.time()

    for i, f in enumerate(IF_freqs_12):
        meta_dict[element]['f_state_IF'] = f
        print(i, time.time()-start_time)

        exp = Rabi_fState(element, a_min, a_max, num_a, shots, calib_params, outer_folder=outer_folder, path='Rabi', meta_dict=meta_dict,ss=False,confusion=confusion, plot = False)
        chevron_data_2[i] = exp.data[1]

        fig, ax = plt.subplots(1,constrained_layout=True)
        pcm1 = ax.pcolormesh(exp.amp_vec, IF_freqs_12, chevron_data_2)
        fig.colorbar(pcm1, ax=ax)
        plt.savefig(exp.iname[:-8]+'chevron', bbox_inches='tight')
##################################################

# meta_dict['q2']['read_amp'] = 0.4
#
# SS = SingleShot1Q3Level('q2', shots, outer_folder=outer_folder, meta_dict=meta_dict)

# Single shot calibration
#
# num_f = 11
# num_p = 10
# f_vec = np.linspace(-4e6, 4e6, num_f)
# f_0_1 = Q1['r_IF']
# f_0_2 = Q2['r_IF']
# p_vec = np.linspace(0.02, 0.1, num_p)
# fidels_1 = np.zeros([num_f,num_p])
# fidels_2 = np.zeros([num_f,num_p])
#
# for j, p in enumerate(p_vec):
#     meta_dict['q1']['read_amp'] = p
#     meta_dict['q2']['read_amp'] = p
#     for i, f in enumerate(f_vec):
#         print(i,j)
#         meta_dict['q1']['r_IF'] = f_0_1 + f
#         meta_dict['q2']['r_IF'] = f_0_2 + f
#
#         SS_1 = SingleShot1Q3Level('q1', shots, outer_folder=outer_folder, meta_dict=meta_dict, plot=False)
#         SS_2 = SingleShot1Q3Level('q2', shots, outer_folder=outer_folder, meta_dict=meta_dict, plot=False)
#
#         fidels_1[i][j] = SS_1.max_F
#         fidels_2[i][j] = SS_2.max_F
#
#         if i == 0:
#             if j == 0:
#                 name = SS_1.iname[:-6]+'qutrit'
#
#         vmin = np.min([fidels_1, fidels_2])
#         vmax = np.max([fidels_1, fidels_2])
#
#         fig, ax = plt.subplots(1, 2, constrained_layout=True)
#         pcm1 = ax[0].pcolormesh(p_vec, f_vec+f_0_1, fidels_1, vmin=vmin, vmax=vmax)
#         pcm2 = ax[1].pcolormesh(p_vec, f_vec+f_0_2, fidels_2, vmin=vmin, vmax=vmax)
#         ax[0].set_xlabel('Resonator 1 Amplitude (V)')
#         ax[1].set_xlabel('Resonator 2 Amplitude (V)')
#         ax[0].set_ylabel('Resonator 1 IF (Hz)')
#         ax[1].set_ylabel('Resonator 2 IF (Hz)')
#         ax[0].set_title('Q1')
#         ax[1].set_title('Q2')
#         fig.colorbar(pcm1, ax=ax)
#         plt.savefig(name, bbox_inches='tight')
#         plt.close()
# #
#
