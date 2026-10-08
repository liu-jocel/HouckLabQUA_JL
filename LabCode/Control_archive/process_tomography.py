import pickle
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') ### Using the 'Agg' backend so that won't have memory issues saving figures
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from qm.qua import *
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_state_tomo import StateTomography
from LabCode2.Experiments.m_SWAP_osc_time import SWAP_Osc as SWAP_Osc_time
from LabCode2.Experiments.m_SWAP_osc_phase import SWAP_Osc_phase
from LabCode2.Helpers.gate_macros import CZ
from LabCode2.Helpers.tomography_tools import optimize_Z_rotations
import Drivers.YOKOGS200 as YOKOGS200

import qiskit
import qiskit.quantum_info as qi
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import ProcessTomography as ProcessTomographyQiskit
from qiskit_experiments.library import LocalReadoutError, CorrelatedReadoutError

from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeParis
from LabCode2.Helpers.iSWAP_tomography import plot_processtomo
from LabCode2.Helpers.gate_macros import flat_top_iSWAP, flat_top_iSWAP_Z_corr
from LabCode2.Helpers.gate_macros import flat_top_CZ

backend = AerSimulator.from_backend(FakeParis())

outer_folder = 'Y:/Charlie/Data/TwoQubit/NSG6/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'

# Get metaDict
with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)


Q1 = meta_dict['q1']
Q2 = meta_dict['q2']
coupler = meta_dict['c1']
element_1 = Q1
element_2 = Q2

#coupler experimental params
# meta_dict['c1']['pulse_amp'] = 0.07
meta_dict['c1']['c_IF'] = np.abs((meta_dict['q1']['q_LO']['LO_freq']+meta_dict['q1']['q_IF'])-(meta_dict['q2']['q_LO']['LO_freq']+meta_dict['q2']['q_IF']))
# meta_dict['c1']['c_LO']['LO_freq'] = 1.9335e9-meta_dict['c1']['c_IF'] #1.9335e9-meta_dict['c1']['c_IF'] #1.8335e9
# meta_dict['c1']['c_LO']['LO_pow'] = 19 #1.9335e9-meta_dict['c1']['c_IF'] #1.8335e9
# meta_dict['global_params']['iSWAP_len'] = 1000
# config = meta_dict_to_config_2Q(meta_dict)
# LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO']]
# set_LO_parameters(LOs)

# Single shot calibration

shots = 1000

calib_params_1Q = []
for i,q in enumerate(['q1','q2']):
    SS = SingleShot1Q(q, shots, meta_dict=meta_dict, outer_folder=outer_folder, path='ProcessTomo',
                        suffix=q)
    calib_params_1Q.append(SS.calib_params)

SS = SingleShot2Q('q1', 'q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='ProcessTomo', suffix='2Q')
calib_params = SS.calib_params
confusion = SS.confusion

#iswap linecut
t_min = 16e-9
t_max = 200e-9
num_t = 40
shots = 400

# phase calibration parameters
num_phi_phase = 20
shots_phase = 500
phi_corr = [0.0, 0.0] #[0.74, 0.65], good: iSWAP[0.59, 0.63] [0.96,0.89] XX[[0.6,0.66]]

time_cal = SWAP_Osc_time('q1', 'q2', 'c1', t_min, t_max, num_t, shots, calib_params, meta_dict=meta_dict,
                                 outer_folder=outer_folder, path='ProcessTomo', suffix='iSWAP', confusion = confusion, fit=True)

exp = SWAP_Osc_phase('q1', 'q2', 'c1', num_phi_phase, phi_corr[0], shots_phase,
                             calib_params_1Q[0], meta_dict=meta_dict,
                             outer_folder=outer_folder, path='ProcessTomo', suffix='phase', plot=True)

# phi_corr[0] = exp.phi_corr
#
# exp = SWAP_Osc_phase('q1', 'q2', 'c1', num_phi_phase, phi_corr[0], shots_phase,
#                              calib_params_1Q[0], meta_dict=meta_dict,
#                              outer_folder=outer_folder, path='ProcessTomo', suffix='phase', plot=True)

exp = SWAP_Osc_phase('q2', 'q1', 'c1', num_phi_phase, phi_corr[1], shots_phase,
                             calib_params_1Q[1], meta_dict=meta_dict,
                             outer_folder=outer_folder, path='ProcessTomo', suffix='phase', plot=True)

# phi_corr[1] = exp.phi_corr
#
# exp = SWAP_Osc_phase('q2', 'q1', 'c1', num_phi_phase, phi_corr[1], shots_phase,
#                              calib_params_1Q[1], meta_dict=meta_dict,
#                              outer_folder=outer_folder, path='ProcessTomo', suffix='phase', plot=True)



#
# ### State tomography versus coupler drive duration t
# def circuit_macro(element_1, element_2, coupler, t):
#     align()
#     play('X180', element_1)
#     align()
#     flat_top_CZ(coupler, t)
#
# # The different drive duration that we previously measured in SWAP_Osc_time
# times = time_cal.clk_vec
# purities = np.zeros(num_t)
#
# for j in range(num_t):
#     tomo = StateTomography('q1', 'q2', 'c1', circuit_macro, times[j], duration_ns=times[j]*4,
#     shots, calib_params, meta_dict=meta_dict,
#                            outer_folder=outer_folder, path='StateTomography', confusion=SS.confusion)
#     purities[j] = tomo.data[1].purity().real
#     print(j, purities)
#
# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# ax.plot(time_cal.t_vec * 1e9, purities, 'o', label='State purity')
# ax.set_xlabel('Time (ns)')
# ax.set_ylabel('State purity')
# ax.set_title('SWAP Oscillation Purity')
# ax.legend()
# plt.savefig(tomo.iname[:-4] + '_purity.png', bbox_inches="tight")
# plt.close(fig)

#coupler experimental params for process tomo
# meta_dict['global_params']['iSWAP_len'] = 504 #252*2 #252
coupler_duration_ns = meta_dict['global_params']['iSWAP_len']
# config = meta_dict_to_config_2Q(meta_dict)
# LOs = [Q1['q_LO'], Q1['r_LO'],Q2['q_LO'], Q2['r_LO'], coupler['c_LO']]
# set_LO_parameters(LOs)

for j in range(1):
    ## Dummy parameters for qiskit process tomo fit
    circ = qiskit.QuantumCircuit(2)
    # circ.cz(1,0)
    circ.sxdg(0)
    circ.sxdg(0)
    tomo_qiskit = ProcessTomographyQiskit(circ)
    analysis_qiskit = tomo_qiskit.analysis
    tomo_init_data = tomo_qiskit.run(backend, seed_simulation=100).block_for_results()


    # Process tomo circuit macro
    def circuit_macro(element_1, element_2, coupler, preps):
        # prep pulses
        # wait_for_trigger(element_1)
        # wait_for_trigger(element_2)
        # wait_for_trigger(coupler)
        # align()
        play(preps[0], element_1)
        play(preps[1], element_2)
        align()
        # play('X180','q1')
        # play('X180','q2')
        # play('iSWAP_gauss', coupler)
        # flat_top_iSWAP(coupler) # the next one was ther
        flat_top_iSWAP_Z_corr(element_1,element_2,coupler,phi_corr[0],phi_corr[1])


    shots = 500
    preps = ['Zero', 'X180', 'Y90', 'Xm90']
    prep_list = [[preps[i], preps[j]] for i in range(4) for j in range(4)]
    process_tomo_counts = []

    ############################################################
    ### Acquire process tomo data
    ############################################################
    for i,p in enumerate(prep_list):
        print(f'Prep {i+1}/16')
        if i==4 or i==1:
            tomo = StateTomography('q1', 'q2', 'c1', circuit_macro, p, duration_ns=coupler_duration_ns,
                                   shots=shots, calib_params=calib_params, meta_dict=meta_dict,
                                   outer_folder=outer_folder, path='ProcessTomo', plot=True, confusion=SS.confusion)
            process_tomo_counts.append(tomo.data[0])
        else:
            tomo = StateTomography('q1', 'q2', 'c1', circuit_macro, p,  duration_ns=coupler_duration_ns,
                                   shots=shots, calib_params=calib_params, meta_dict=meta_dict,
                                   outer_folder=outer_folder, path='ProcessTomo', plot=True, confusion=SS.confusion)
            process_tomo_counts.append(tomo.data[0])

    ####
    ### Arranging the counts (not actually raw data, it has been corrected in StateTomography)
    ####
    raw_data = np.zeros([144, 4])
    for i in range(16):
        raw_data[9*i:9*i+9, :] = process_tomo_counts[i]

    # Make qiskit data object that can be used to fit the process tomo
    data_holder = []
    for i in range(144):
        prob_dict = {'00': raw_data[i][0] / shots, '01': raw_data[i][1] / shots, '10': raw_data[i][2] / shots,
                     '11': raw_data[i][3] / shots}
        # quasi_probs = mitigator.quasi_probabilities(prob_dict)
        # mitigated_probs = quasi_probs.nearest_probability_distribution().binary_probabilities(2)
        counts_dict = {}
        for k, v in prob_dict.items():
            counts_dict[k] = int(v*shots)#int(v * shots) ## No Qiskit SPAM
        data_holder.append({'counts': counts_dict,
                            'metadata':tomo_init_data.data(i)['metadata']})
    data_holder_qiskit = ExperimentData()
    data_holder_qiskit.add_data(data_holder)
    fitted_data = analysis_qiskit.run(data_holder_qiskit)

    for r in fitted_data.analysis_results():
        print(r)

    state = fitted_data.analysis_results('state').value

    # Plot process tomo
    plot_processtomo(tomo,state,circ.to_gate())




# Save process tomo data
pathPickle = tomo.iname.replace(".png", ".pkl")
with open(pathPickle, 'wb') as f:
    pickle.dump(fitted_data, f)

############################################################
### Stuff specific to process tomo for the CZ gate
#############################################################
circ1 = qiskit.QuantumCircuit(2)
# matrix_xx = [[0, 0, 0, 1],
#           [0, 0, 1, 0],
#           [0, 1, 0, 0],
#           [1, 0, 0, 0]]
# circ1.unitary(matrix_xx, [0, 1])
circ1.iswap(0,1)

choi_target = qi.Choi(circ1)

fidels = optimize_Z_rotations(state,choi_target)
print(np.max(fidels))
# print(np.argmax(fidels, keepdims = True))
print(np.unravel_index(np.argmax(fidels),fidels.shape))

############################################################
### Extra stuff that doesn't matter for us and was originally commented out
############################################################
#
# state = fitted_data.analysis_results('state').value
# print(state)
# print(state.purity())
# IFs = np.linspace(10e6,100e6,10)
#
# res_freq = coupler['c_LO']['LO_freq'] + coupler['c_IF']
#
# for i in range(10):
#
#     coupler['c_LO']['LO_freq'] = res_freq - IFs[i]
#     set_LO_parameters([coupler['c_LO']])
#
#     meta_dict['c1']['c_IF'] = IFs[i]
#     config = meta_dict_to_config_2Q(meta_dict)
#     process_tomo_counts = []
#
#     for i, p in enumerate(prep_list):
#         print(f'Prep {i + 1}/16')
#
#         tomo = StateTomography(element_1, element_2, coupler, circuit_macro, p, shots, calib_params, cfg=config,
#                                outer_folder=outer_folder, path='StateTomography', confusion=SS.confusion)
#
#         process_tomo_counts.append(tomo.data)
#
#     raw_data = np.zeros([144, 4])
#
#     for i in range(16):
#         raw_data[9 * i:9 * i + 9, :] = process_tomo_counts[i]
#
#     # Qiskit data simulation
#
#     data_holder = []
#
#     for i in range(144):
#         data_holder.append(
#             {'counts': {'00': raw_data[i][0], '01': raw_data[i][1], '10': raw_data[i][2], '11': raw_data[i][3]},
#              'metadata': tomo_init_data.data(i)['metadata']})
#
#     data_holder_qiskit = ExperimentData()
#     data_holder_qiskit.add_data(data_holder)
#
#     fitted_data = analysis_qiskit.run(data_holder_qiskit)
#
#     # for r in tomo_init_data.analysis_results():
#     #     print(r)
#
#     for r in fitted_data.analysis_results():
#         print(r)
#
#     state = fitted_data.analysis_results('state').value
#
#     plot_processtomo(tomo, state, circ.to_gate())