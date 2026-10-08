import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_single_shot_1Q2L import SingleShot1Q2L
from LabCode2.Experiments.m_single_shot_1Q2L_logistic_regression import SingleShot1Q2L_LR
from LabCode2.Experiments.m_single_shot_2Q2L import SingleShot2Q2L
from LabCode2.Experiments.m_single_shot_2Q3L import SingleShot2Q3L
from LabCode2.Experiments.m_single_shot_1Q_3level import SingleShot1Q3Level
import Drivers.YOKOGS200 as YOKOGS200

outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# YOKOGS200.main(0.52)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
Qc = meta_dict['qc']
config = meta_dict_to_config_2Q(meta_dict)

LOs = [Q1['q_LO'],Q2['q_LO'],Qc['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

qubits = ["q1", "q2"]

# Readout Tuning Parameters
num_f = 11#11
num_p = 10 #10
f_vec = np.linspace(-1e6, 1e6, num_f) #4e6
f_0_1 = Q1['r_IF']
f_0_2 = Q2['r_IF']
f_0_c = Qc['r_IF']
center_freqs = {
    "q1": f_0_1,
    "q2": f_0_2,
    "qc": f_0_c
}
p_0_1 = Q1['read_amp']
p_0_2 = Q2['read_amp']
p_0_c = Qc['read_amp']
p_vec = np.linspace(0.0, 0.055, num_p)


do_1Q2L_readout_cal = 1
do_1Q3L_readout_cal = 0
do_2Q2L_readout_cal = 1
do_2Q3L_readout_cal = 0
do_3Q2L_readout_cal = 0
do_3Q3L_readout_cal = 0

do_1Q2L_readout_tune = 0
do_1Q3L_readout_tune = 0
do_2Q2L_readout_tune = 1
do_2Q3L_readout_tune = 0
do_3Q2L_readout_tune= 0
do_3Q3L_readout_tune = 0

do_discrim_readout = 0

# Single shot parameters
shots_SS = {
    "q1" : 1000,
    "q2": 1000,
    "qc": 1000
}


if do_1Q2L_readout_cal:
    for i in range(100):
        calib_params = []
        for i,q in enumerate(qubits):
            SS1Q2L = SingleShot1Q2L(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                                suffix=q)
            calib_params.append(SS1Q2L.calib_params)
        print(calib_params)

if do_1Q3L_readout_cal:
    calib_params = []
    for i,q in enumerate(qubits):
        SS1Q3L = SingleShot1Q3Level(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                            suffix=q)
        calib_params.append(SS1Q3L.calib_params)
    print(calib_params)

if do_2Q2L_readout_cal:
    calib_params = []
    SS2Q2L = SingleShot2Q2L(qubits[0], qubits[1], shots_SS["q1"], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                            suffix="2Q2L")
    calib_params.append(SS2Q2L.calib_params)
    print(calib_params)

if do_2Q3L_readout_cal:
    calib_params = []
    SS2Q3L = SingleShot2Q3L(qubits[0], qubits[1], shots_SS["q1"], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                            suffix="2Q3L")
    calib_params.append(SS2Q3L.calib_params)
    print(calib_params)

if do_1Q2L_readout_tune:
    for i, q in enumerate(qubits):
        SS1Q2L = SingleShot1Q2L(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                                suffix=q)

        fidelity = np.zeros([num_f, num_p])

        for j, p in enumerate(p_vec):
            for k, f in enumerate(f_vec):
                print(j, k)

                meta_dict[q]['read_amp'] = p

                meta_dict[q]['r_IF'] = center_freqs[q] + f

                SS1Q2L = SingleShot1Q2L(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder,
                                        path='SingleShotCal', plot=True, suffix=q)
                if k == 0:
                    if j == 0:
                        name = SS1Q2L.iname[:-6] + '2Q'

                fidelity[k][j] = SS1Q2L.max_F

                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                pcm1 = ax.pcolormesh(p_vec, f_vec + center_freqs[q], fidelity)
                ax.set_xlabel('Readout amplitude (V)')
                ax.set_ylabel('Resonator IF (Hz)')
                ax.set_title(q)
                fig.colorbar(pcm1, ax=ax)
                plt.savefig(name + '2Q', bbox_inches='tight')
                plt.close()


if do_1Q3L_readout_tune:
    for i, q in enumerate(qubits):
        SS1Q3L = SingleShot1Q3Level(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                                suffix=q)

        fidelity = np.zeros([num_f, num_p])

        for j, p in enumerate(p_vec):
            for k, f in enumerate(f_vec):
                print(j, k)

                meta_dict[q]['read_amp'] = p

                meta_dict[q]['r_IF'] = center_freqs[q] + f

                SS1Q3L = SingleShot1Q3Level(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder,
                                        path='SingleShotCal', plot=False, suffix=q)
                if k == 0:
                    if j == 0:
                        name = SS1Q3L.iname[:-6] + '2Q'

                fidelity[k][j] = SS1Q3L.max_F

                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                pcm1 = ax.pcolormesh(p_vec, f_vec + center_freqs[q], fidelity)
                ax.set_xlabel('Readout amplitude (V)')
                ax.set_ylabel('Resonator IF (Hz)')
                ax.set_title(q)
                fig.colorbar(pcm1, ax=ax)
                plt.savefig(name + '2Q', bbox_inches='tight')
                plt.close()

if do_discrim_readout:
    for i, q in enumerate(qubits):
        SS1Q2L = SingleShot1Q2L(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                                suffix=q)

        fidelity = np.zeros([num_f, num_p])

        for j, p in enumerate(p_vec):
            for k, f in enumerate(f_vec):
                print(j, k)

                meta_dict[q]['read_amp'] = p

                meta_dict[q]['r_IF'] = center_freqs[q] + f

                SS1Q2L = SingleShot1Q2L(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder,
                                        path='SingleShotCal', plot=False, suffix=q)
                if k == 0:
                    if j == 0:
                        name = SS1Q2L.iname[:-6] + '2Q'

                fidelity[k][j] = SS1Q2L.max_F

                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                pcm1 = ax.pcolormesh(p_vec, f_vec + center_freqs[q], fidelity)
                ax.set_xlabel('Readout amplitude (V)')
                ax.set_ylabel('Resonator IF (Hz)')
                ax.set_title(q)
                fig.colorbar(pcm1, ax=ax)
                plt.savefig(name + '2Q', bbox_inches='tight')
                plt.close()

#
# if do_1Q2L_readout_cal_lr:
#     for i,q in enumerate(qubits):
#         SS1Q2L = SingleShot1Q2L(q, shots_SS[i], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
#                             suffix=q)
