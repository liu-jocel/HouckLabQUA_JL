import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_temperature_boltzmann import TemperatureBoltzmann
from LabCode2.Experiments.m_temperature_RPM import TemperatureRPM
from LabCode2.Experiments.m_Rabi_fState import Rabi_fState
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

qubits = ['q1', 'q2']

# Readout Tuning Parameters
num_f = 11
num_p = 10
f_vec = np.linspace(-4e6, 4e6, num_f)
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
p_vec = np.linspace(0.02, 0.4, num_p)

do_1Q_readout_cal = 1
do_temp_boltzmann = 1
do_temp_RPM = 1

# Single shot parameters
shots_SS = {
    "q1" : 1000,
    "q2": 1000,
    "qc": 500
}

anharms = {
    "q1": 250,
    "q2": 250,
}

a_min = -0.3
a_max = 0.3
num_a = 80

if do_1Q_readout_cal:
    calib_params = dict()
    for i,q in enumerate(qubits):
        SS1Q = SingleShot1Q(q, shots_SS[q], meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal',
                            suffix=q)
        calib_params[q] = SS1Q.calib_params
    print(calib_params)

if do_temp_boltzmann:
    for i,q in enumerate(qubits):
        temp_boltzmann = TemperatureBoltzmann(q, shots_SS[q], calib_params[q], anharms[q], meta_dict=meta_dict, outer_folder=outer_folder, path='Temperature',
                            suffix=q, plot=True, save=True)
        print(temp_boltzmann)

if do_temp_RPM:
    for i,q in enumerate(qubits):
        temp_RPM = TemperatureRPM(q, a_min, a_max, num_a, shots_SS[q], calib_params[q], anharms[q], confusion=SS1Q.confusion, meta_dict=meta_dict, outer_folder=outer_folder, path='Temperature',
                            suffix=q, plot=True, save=True)