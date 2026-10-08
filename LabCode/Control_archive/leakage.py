import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_T1_leakage import T1_leakage
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

LOs = [Q1['q_LO'],Q2['q_LO'], Q2['r_LO']]

# set_LO_parameters(LOs)

qubits = ['q1','q2']

do_readout_cal = 1

do_T1 = 1

# Single shot parameters
shots_SS = [1000,1000,1000]

# T1 parameters
t_min_t1 = [1e-6, 1e-6]
t_max_t1 = [400e-6, 400e-6]
num_t_t1 = [40, 40]
shots_t1 = [5000, 5000]


if do_readout_cal:
    calib_params = []
    confusions = []
    for i,q in enumerate(qubits):
        SS = SingleShot1Q3Level(q, shots_SS[i], meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char',
                            suffix=q)
        calib_params.append(SS.calib_params)
        confusions.append(SS.confusion)



if do_T1:
    for i,q in enumerate(qubits):
        set_LO_parameters([meta_dict[q]['q_LO']])
        exp = T1_leakage(q, t_min_t1[i], t_max_t1[i], num_t_t1[i], shots_t1[i], calib_params[i], confusion=confusions[i],
                 meta_dict=meta_dict, outer_folder=outer_folder, path='1Q_Char', suffix=q, save=True, fit=False)


