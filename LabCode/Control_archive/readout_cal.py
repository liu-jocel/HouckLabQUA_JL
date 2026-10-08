import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode.Config.config_helpers import meta_dict_to_config_6Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Experiments.m_T1 import T1
from LabCode.Experiments.m_T2R import T2R
from LabCode.Experiments.m_transmission import Transmission
from LabCode.Experiments.m_qubit_spec import QubitSpec
from LabCode.Experiments.m_Rabi import Rabi
from LabCode.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.m_single_shot_2Q import SingleShot2Q
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Z://t1Team/Data/2024-12-22_cooldown/TATQ01_Topsil_01/OPX/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# YOKOGS200.main(0.52)
q='q2'
Q = meta_dict[q]
path = q
config = meta_dict_to_config_6Q(meta_dict)

# Q['q_LO']['LO_freq'] = 7.781e9  # TWPA PUMP
# Q['q_LO']['LO_pow'] = 20.1

LOs = [Q['q_LO'],Q['r_LO']]

# set_LO_parameters(LOs)
shots = 500

SS = SingleShot1Q(q, shots, meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix='SS_cal')

print(SS.confusion)

# Single shot calibration
#
num_f = 7
num_p = 11
f_vec = np.linspace(-0.5e6, 0.5e6, num_f)
f_0_1 = Q['r_IF']
p_0_1 = 0.06 #Q['read_amp']
p_vec = np.linspace(0.01, 0.06, num_p) #0.4
fidels_1 = np.zeros([num_f,num_p])

for j, p in enumerate(p_vec):
    for i, f in enumerate(f_vec):
        print(j,i)

        meta_dict[q]['read_amp'] = p

        meta_dict[q]['r_IF'] = f_0_1 + f

        SS_1 = SingleShot1Q(q, shots, meta_dict=meta_dict, outer_folder=outer_folder, path=path, suffix='Read_cal', plot=False)

        if i == 0:
            if j == 0:
                name = SS_1.iname[:-6]+'2Q'

        fidels_1[i][j] = SS_1.max_F

        vmin = np.max([np.min([fidels_1]),0.5])
        vmax = np.max([fidels_1])

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        pcm1 = ax.pcolormesh(p_vec, f_vec + f_0_1, fidels_1, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Readout amplitude (V)')
        ax.set_ylabel('Resonator IF (Hz)')
        ax.set_title(q)
        fig.colorbar(pcm1, ax=ax)
        plt.savefig(name + '2QC', bbox_inches='tight')
        plt.close()


#
# fidels_1 = np.zeros([num_f,num_f])
# fidels_2 = np.zeros([num_f,num_f])
#
# meta_dict[q]['read_amp'] = p_0_1
# meta_dict['q2']['read_amp'] = p_0_2
#
# for i, f1 in enumerate(f_vec):
#     meta_dict[q]['r_IF'] = f_0_1 + f1
#
#
#     for j, f2 in enumerate(f_vec):
#         print(i,j)
#         meta_dict['q2']['r_IF'] = f_0_2 + f2
#
#         SS = SingleShot2Q(q, 'q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='2Q', plot=False)
#
#         fidels_1[i][j] = SS.max_F[0]
#         fidels_2[i][j] = SS.max_F[1]
#
#         if i == 0:
#             if j == 0:
#                 name = SS.iname[:-6]+'2Q'
#
#         vmin = np.min([fidels_1, fidels_2])
#         vmax = np.max([fidels_1, fidels_2])
#
#         fig, ax = plt.subplots(1, 2, constrained_layout=True)
#         pcm1 = ax[0].pcolormesh(f_vec+f_0_2, f_vec+f_0_1, fidels_1, vmin=vmin, vmax=vmax)
#         pcm2 = ax[1].pcolormesh(f_vec+f_0_2, f_vec+f_0_1, fidels_2, vmin=vmin, vmax=vmax)
#         ax[0].set_xlabel('Resonator 2 IF (Hz)')
#         ax[1].set_xlabel('Resonator 2 IF (Hz)')
#         ax[0].set_ylabel('Resonator 1 IF (Hz)')
#         ax[1].set_ylabel('Resonator 1 IF (Hz)')
#         ax[0].set_title('Q1')
#         ax[1].set_title('Q2')
#         fig.colorbar(pcm1, ax=ax)
#         plt.savefig(name, bbox_inches='tight')
#         plt.close()