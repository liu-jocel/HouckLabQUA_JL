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
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from LabCode2.Experiments.m_SWAP_osc_time import SWAP_Osc
import Drivers.YOKOGS200 as YOKOGS200
import math

def nearest_kHz(f):
    return int(math.ceil(f/1e3))*1e3

outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
config = meta_dict_to_config_2Q(meta_dict)

LOs = [Q1['q_LO'],Q2['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

qubits = ['q1','q2']

shots_SS = [1000,1000]

calib_params = []

for i,q in enumerate(qubits):
    SS = SingleShot1Q(q, shots_SS[i], meta_dict=meta_dict, outer_folder=outer_folder, path='lamb',
                        suffix=q)
    calib_params.append(SS.calib_params)

SS_2Q = SingleShot2Q('q1', 'q2', shots_SS[0], meta_dict=meta_dict, outer_folder=outer_folder, path='lamb', suffix='2Q')

calib_params_2Q = SS_2Q.calib_params

t_min = 100e-9
t_max = 0.5e-6
num_t = 50
shots = 500
detuning = 4e6

# iSWAP parameters
do_iSWAP = 1
t_min_iSWAP = 16e-9
t_max_iSWAP = 4000e-9
num_t_iSWAP = 80
freq_range_iSWAP = 3e6
num_freqs_iSWAP = 21
shots_iSWAP = 100

yoko = -5.00

data_name = outer_folder + 'lamb/data_m5p00.pkl'

def get_iSWAP_data(t_min_iSWAP,t_max_iSWAP,num_t_iSWAP,freq_center_iSWAP,freq_range_iSWAP,num_freqs_iSWAP,shots_iSWAP):

    chevron_data = np.zeros([num_freqs_iSWAP, 4, num_t_iSWAP])
    IF_freqs = freq_center_iSWAP + np.linspace(-freq_range_iSWAP / 2, freq_range_iSWAP / 2, num_freqs_iSWAP)
    start_time = time.time()

    for j in range(num_freqs_iSWAP):

        print(j, time.time() - start_time)
        meta_dict['c1']['c_IF'] = IF_freqs[j]

        if num_freqs_iSWAP == 1:  # Use native plotting in experiment if you just want a line cut

            exp = SWAP_Osc('q1', 'q2', 'c1', t_min_iSWAP, t_max_iSWAP, num_t_iSWAP, shots_iSWAP, calib_params_2Q,
                           meta_dict=meta_dict, outer_folder=outer_folder, path='lamb', suffix='iSWAP')

        else:  # Do plotting here if you want a chevron

            exp = SWAP_Osc('q1', 'q2', 'c1', t_min_iSWAP, t_max_iSWAP, num_t_iSWAP, shots_iSWAP, calib_params_2Q,
                           meta_dict=meta_dict, outer_folder=outer_folder, path='lamb', suffix='iSWAP',
                           plot=False)

            if j == 0:
                name = exp.iname[:-9]
                pname = exp.pname

            chevron_data[j] = exp.data[1]

            if num_freqs_iSWAP > 1:
                vmin = np.min(chevron_data / exp.shots)
                vmax = np.max(chevron_data / exp.shots)
                fig, ax = plt.subplots(2, 2, constrained_layout=True)
                pcm1 = ax[0][0].pcolormesh(exp.t_vec, IF_freqs, chevron_data[:, 0, :] / exp.shots, vmin=vmin,
                                           vmax=vmax)
                pcm2 = ax[0][1].pcolormesh(exp.t_vec, IF_freqs, chevron_data[:, 1, :] / exp.shots, vmin=vmin,
                                           vmax=vmax)
                pcm3 = ax[1][0].pcolormesh(exp.t_vec, IF_freqs, chevron_data[:, 2, :] / exp.shots, vmin=vmin,
                                           vmax=vmax)
                pcm4 = ax[1][1].pcolormesh(exp.t_vec, IF_freqs, chevron_data[:, 3, :] / exp.shots, vmin=vmin,
                                           vmax=vmax)
                fig.colorbar(pcm1, ax=ax)
                ax[0][0].set_title('00')
                ax[0][1].set_title('01')
                ax[1][0].set_title('10')
                ax[1][1].set_title('11')

                ax[1][0].set_ylabel('Coupler Frequency (Hz)')
                ax[0][0].set_ylabel('Coupler Frequency (Hz)')
                ax[1][0].set_xlabel('Pulse time (s)')
                ax[1][1].set_xlabel('Pulse time (s)')
                plt.suptitle('SWAP Chevron')

                plt.savefig(name + 'chevron', bbox_inches='tight')
                plt.close()

    return [exp.t_vec, IF_freqs, chevron_data]



YOKOGS200.main(yoko)

freq_cal_1 = T2R('q1', t_min, t_max, num_t, shots, detuning, calib_params[0], meta_dict=meta_dict,
                 outer_folder=outer_folder,
                 path='lamb', suffix='Q1_f', freq_cal=True)
freq_cal_2 = T2R('q2', t_min, t_max, num_t, shots, detuning, calib_params[1], meta_dict=meta_dict,
                 outer_folder=outer_folder,
                 path='lamb', suffix='Q2_f', freq_cal=True)

meta_dict['q1']['q_LO']['LO_freq'] = nearest_kHz(meta_dict['q1']['q_LO']['LO_freq'] - freq_cal_1.freq_correction)
meta_dict['q2']['q_LO']['LO_freq'] = nearest_kHz(meta_dict['q2']['q_LO']['LO_freq'] - freq_cal_2.freq_correction)

LOs = [meta_dict['q1']['q_LO'], meta_dict['q2']['q_LO']]
# set_LO_parameters(LOs)

freq_1 = meta_dict['q1']['q_LO']['LO_freq'] + meta_dict['q1']['q_IF']
freq_2 = meta_dict['q2']['q_LO']['LO_freq'] + meta_dict['q2']['q_IF']


freq_center_iSWAP = np.abs(freq_1-freq_2)


# data_4us = get_iSWAP_data(16e-9, 4000e-9, 80, freq_center_iSWAP, 3e6, 21, 100)
# data_1us = get_iSWAP_data(16e-9, 1000e-9, 80, freq_center_iSWAP, 6e6, 21, 100)
# data_p5us = get_iSWAP_data(16e-9, 500e-9, 80, freq_center_iSWAP, 10e6, 21, 100)

data_dict = {'f1':freq_1,'f2':freq_2}


with open(data_name,'wb') as f:
    pickle.dump(data_dict,f)

outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

with open(pathPickle, 'wb') as f:
    pickle.dump(meta_dict,f)
