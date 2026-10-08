import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from qm.qua import *
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from LabCode2.Experiments.m_state_tomo import StateTomography
from LabCode2.Helpers.iSWAP_tomography import plot_statetomo
from LabCode2.Helpers.gate_macros import CZ
import Drivers.YOKOGS200 as YOKOGS200

import qiskit
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import StateTomography as StateTomographyQiskit
from qiskit_experiments.library import LocalReadoutError, CorrelatedReadoutError

from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeParis
from qiskit.quantum_info import DensityMatrix

backend = AerSimulator.from_backend(FakeParis())

outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

# Define macro for state tomography, Qiskit circuit also required

circ = qiskit.QuantumCircuit(2)
circ.s(0)
circ.sx(0)
circ.sdg(0)
circ.s(1)
circ.sx(1)
circ.sdg(1)
circ.cz(0,1)
circ.z(0)
circ.z(1)
circ.s(0)
circ.sx(0)
circ.sdg(0)

qiskit_dm = DensityMatrix(circ)

tomo_qiskit = StateTomographyQiskit(circ)
analysis_qiskit = tomo_qiskit.analysis
tomo_init_data = tomo_qiskit.run(backend, seed_simulation=100).block_for_results()

readout_qiskit = CorrelatedReadoutError([0,1])
readout_analysis_qiskit = readout_qiskit.analysis
readout_init_data = readout_qiskit.run(backend)
data_holder_meta = readout_init_data.metadata

def circuit_macro(element_1, element_2, coupler, a):

    # # pass
    # play('X90', element_2['q_name'])
    # align()
    # # reset_phase(coupler['c_name'])
    # play('iSWAP_const', coupler['c_name'])
    align()
    play('Y90', element_1['q_name'])
    play('Y90', element_2['q_name'])
    align()

    CZ(element_1,element_2,coupler,-0.72,-0.02)
    align()

    frame_rotation_2pi(0.5, element_1['q_name'])
    frame_rotation_2pi(0.5, element_2['q_name'])

    align()
    play('Y90', element_1['q_name'])


# SHOTS = 1024
# qubits = [0,1]
# num_qubits = len(qubits)

#
# result = exp.run(backend)
# mitigator = result.analysis_results(0).value



# YOKOGS200.main(0.52)

Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
coupler = meta_dict['c1']
element_1 = Q1
element_2 = Q2
config = meta_dict_to_config_2Q(meta_dict)

LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO'], coupler['c_LO']]

# set_LO_parameters(LOs)

# Single shot calibration
#
shots = 1000
SS = SingleShot2Q(meta_dict['q1'], meta_dict['q2'], shots, cfg=config, outer_folder=outer_folder, path='SingleShotCal', suffix='2Q')
calib_params = SS.calib_params

confusion = SS.confusion

data_holder = []
expected_outcome = ['00','01','10','11']
data_holder_qiskit = ExperimentData(metadata=data_holder_meta)
for i in range(0,len(expected_outcome)):
    readout_dict = {'counts': {'00':int(confusion[0][i]*shots), '01':int(confusion[1][i]*shots),
                               '10':int(confusion[2][i]*shots),'11':int(confusion[3][i]*shots)},
                    'metadata': {"state_label": expected_outcome[i]}, 'shots':shots}
    data_holder.append(readout_dict)
data_holder_qiskit.add_data(data_holder)
fitted_data = readout_qiskit.analysis.run(data_holder_qiskit)

mitigator = fitted_data.analysis_results(0).value

# data_holder = []
# outcomes = ['00','01','10','11']
#
# for i in range(4):
#     readout_dict = {'counts': {'00':int(SS.confusion[0][i]*shots), '01':int(SS.confusion[1][i]*shots),'10':int(SS.confusion[2][i]*shots),'11':int(SS.confusion[3][i]*shots)},
#                         'metadata': {"state_label": outcomes[i], 'shots':shots}}
#     data_holder.append(readout_dict)
#
# data_holder_qiskit = ExperimentData()
# data_holder_qiskit.add_data(data_holder)
#
# # fitted_data = readout_analysis_qiskit.run(readout_init_data)
# fitted_data = readout_analysis_qiskit.run(data_holder_qiskit)

shots = 1000
macro_args = 1
tomo = StateTomography(element_1, element_2, coupler, circuit_macro, macro_args, shots, calib_params, cfg=config, outer_folder=outer_folder, path='StateTomography')#, confusion=SS.confusion)

# Simulate control waveforms
# reset_time = 1e-6
# element_1['reset_ticks'] = int(reset_time/4e-9)
#
# tomo = StateTomography(element_1, element_2, coupler, circuit_macro, macro_args, shots, calib_params, cfg=config, outer_folder=outer_folder, path='StateTomography', suffix='sim', confusion=SS.confusion, run=False)
#
# tomo.simulate(['1','2','3','4','5','6','7','8','9','10'],10000,[[1400,1500],[4800,4900],[8400,8500],[11900,12000]])



# Qiskit data simulation
def analyze(tomo):
    raw_data = tomo.data

    data_holder = []

    for i in range(9):
        prob_dict = {'00':raw_data[i][0]/shots, '01':raw_data[i][1]/shots,'10':raw_data[i][2]/shots,'11':raw_data[i][3]/shots}

        quasi_probs = mitigator.quasi_probabilities(prob_dict)

        mitigated_probs = quasi_probs.nearest_probability_distribution().binary_probabilities()

        counts_dict = {}

        for k,v in mitigated_probs.items():
            counts_dict[k] = int(v*shots)

        data_holder.append(
            {'counts': counts_dict,
             'metadata': tomo_init_data.data(i)['metadata']})

        # data_holder.append({'counts': {'00':raw_data[i][0], '01':raw_data[i][1],'10':raw_data[i][2],'11':raw_data[i][3]},
        #                     'metadata':tomo_init_data.data(i)['metadata']})

    data_holder_qiskit = ExperimentData()
    data_holder_qiskit.add_data(data_holder)

    fitted_data = analysis_qiskit.run(data_holder_qiskit)

    # for r in tomo_init_data.analysis_results():
    #     print(r)

    for r in fitted_data.analysis_results():
        print(r)

    state = fitted_data.analysis_results('state').value
    print(state)
    print(state.purity())

    plot_statetomo(tomo,state,qiskit_dm)

analyze(tomo)
#
# IFs = np.linspace(10e6,200e6,20)
#
# res_freq = coupler['c_LO']['LO_freq'] + coupler['c_IF']
#
# for i in range(20):
#
#     coupler['c_LO']['LO_freq'] = res_freq - IFs[i]
#     set_LO_parameters([coupler['c_LO']])
#
#     meta_dict['c1']['c_IF'] = IFs[i]
#     config = meta_dict_to_config_2Q(meta_dict)
#
#     tomo = StateTomography(element_1, element_2, coupler, circuit_macro, macro_args, shots, calib_params, cfg=config,
#                            outer_folder=outer_folder, path='StateTomography', confusion=SS.confusion)
#
#     analyze(tomo)
#
