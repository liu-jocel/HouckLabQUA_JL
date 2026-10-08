import pickle

import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from qm.qua import *
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from LabCode2.Experiments.m_unitary_tomo import UniTomography
from LabCode2.Helpers.iSWAP_tomography import plot_statetomo
from LabCode2.Helpers.gate_macros import CZ,flat_top_iSWAP
from LabCode2.Helpers.qua_helpers import time_vec_to_clk_vec_wait
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

do_amp_tomo = 1
do_time_tomo = 0


def circuit_macro(element_1, element_2, coupler, a): #circuit macro-old

    # # pass
    # play('X90', element_2['q_name'])
    # align()
    # # reset_phase(coupler['c_name'])
    # play('iSWAP_const', coupler['c_name'])
    # align()
    # play('Y90', element_1['q_name'])
    # play('Y90', element_2['q_name'])
    align()
    # print("Nothing played")
    # frame_rotation_2pi(0.5, element_1['q_name'])  # it's virtual Z
    # CZ(element_1,element_2,coupler,-0.72,-0.02) # it's an iSWAP gate
    #  time_vec_to_clk_vec_wait
    # flat_top_CZ(self.coupler, clk) # used to iterate over different times

    flat_top_iSWAP(coupler) #this works

    align()


    # frame_rotation_2pi(0.5, element_2['q_name'])
    #
    # align()
    # play('Y90', element_1['q_name'])




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
#amplitude unitary
if do_amp_tomo:
    uni_data = {}
    for pulse_amp in [0.45]:
        #, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]
        meta_dict["c1"]["pulse_amp"] = pulse_amp
        shots = 1000
        SS = SingleShot2Q('q1', 'q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix='2Q')
        calib_params = SS.calib_params
        confusion = SS.confusion
        shots = 1000
        macro_args = 1
        gate_len = meta_dict["global_params"]["iSWAP_len"]
        exp = UniTomography(element_1, element_2, coupler, circuit_macro, macro_args, gate_len,shots, calib_params, meta_dict = meta_dict ,cfg=config, outer_folder=outer_folder, path='UnitaryTomography', confusion=SS.confusion)
        exp.pickle_data()
        uni_data[pulse_amp] = exp.data

    data = {'uni_data': uni_data}
    with open(outer_folder+'UnitaryTomography/amp_tomo_1.pkl','wb') as f:
        pickle.dump(data,f)

    amplitudes_used = data['uni_data'].keys()
    u11 = []
    u12 = []
    u21 = []
    u22 = []
    u33 = []
    theta = []
    phi = []
    delta_p = []
    delta_m = []
    delta_moff = []

    for key in amplitudes_used:
        u11.append(data['uni_data'][key][1]['mat_el']['u11'])
        u12.append(data['uni_data'][key][1]['mat_el']['u12'])
        u21.append(data['uni_data'][key][1]['mat_el']['u21'])
        u22.append(data['uni_data'][key][1]['mat_el']['u22'])
        u33.append(data['uni_data'][key][1]['mat_el']['u33'])
        theta.append(data['uni_data'][key][1]['angles']['theta'])
        phi.append(data['uni_data'][key][1]['angles']['phi'])
        delta_p.append(data['uni_data'][key][1]['angles']['delta_p'])
        delta_m.append(data['uni_data'][key][1]['angles']['delta_m'])
        delta_moff.append(data['uni_data'][key][1]['angles']['delta_moff'])

    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes_used, np.abs(u11), marker='D', label='u11')
    plt.plot(amplitudes_used, np.abs(u12), marker='D', label='u12')
    plt.plot(amplitudes_used, np.abs(u21), marker='D', label='u21')
    plt.plot(amplitudes_used, np.abs(u22), marker='D', label='u22')
    plt.plot(amplitudes_used, np.abs(u33), marker='D', label='u33')

    plt.xlabel('Amplitude')
    plt.ylabel('Mat_el')
    plt.title('Mat_el as Functions of Amplitude')
    plt.legend()
    plt.grid(True)

    plt.savefig(exp.iname, bbox_inches="tight")

    # plot angles as a function of amplitude
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes_used, theta, marker='o', label='Theta')
    plt.plot(amplitudes_used, phi, marker='x', label='Phi')
    plt.plot(amplitudes_used, delta_p, marker='s', label='Delta Plus')
    plt.plot(amplitudes_used, delta_m, marker='^', label='Delta Minus')
    plt.plot(amplitudes_used, delta_moff, marker='d',
             label='Delta Minus_off')  # Something wrong with the code, let's check

    plt.xlabel('Amplitude')
    plt.ylabel('Angle')
    plt.title('Angles Values as Functions of Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig((exp.iname).rstrip(".png") + "_angles.png", bbox_inches="tight")

#time unitary
if do_time_tomo:
    uni_data = {}
    for pulse_time in [40, 44, 48, 52, 56, 60, 64, 68, 72]: # there's an error here that I need to check
        meta_dict["global_params"]["iSWAP_len"] = pulse_time
        meta_dict["c1"]["pulse_amp"] = 0.45
        gate_len = pulse_time
        shots = 1000
        SS = SingleShot2Q('q1', 'q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='2Q_Char', suffix='2Q')
        calib_params = SS.calib_params
        confusion = SS.confusion
        shots = 1000
        macro_args = 1
        exp = UniTomography(element_1, element_2, coupler, circuit_macro, macro_args, gate_len, shots, calib_params, meta_dict = meta_dict ,cfg=config, outer_folder=outer_folder, path='UnitaryTomography', confusion=SS.confusion)
        exp.pickle_data()
        uni_data[pulse_time] = exp.data

    data = {'uni_data': uni_data}
    with open(outer_folder+'UnitaryTomography/time_tomo_1.pkl','wb') as f:
        pickle.dump(data,f)

    amplitudes_used = data['uni_data'].keys()
    u11 = []
    u12 = []
    u21 = []
    u22 = []
    u33 = []
    theta = []
    phi = []
    delta_p = []
    delta_m = []
    delta_moff = []

    for key in amplitudes_used:
        u11.append(data['uni_data'][key][1]['mat_el']['u11'])
        u12.append(data['uni_data'][key][1]['mat_el']['u12'])
        u21.append(data['uni_data'][key][1]['mat_el']['u21'])
        u22.append(data['uni_data'][key][1]['mat_el']['u22'])
        u33.append(data['uni_data'][key][1]['mat_el']['u33'])
        theta.append(data['uni_data'][key][1]['angles']['theta'])
        phi.append(data['uni_data'][key][1]['angles']['phi'])
        delta_p.append(data['uni_data'][key][1]['angles']['delta_p'])
        delta_m.append(data['uni_data'][key][1]['angles']['delta_m'])
        delta_moff.append(data['uni_data'][key][1]['angles']['delta_moff'])

    # plot matrix elements
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes_used, np.abs(u11), marker='D', label='u11')
    plt.plot(amplitudes_used, np.abs(u12), marker='D', label='u12')
    plt.plot(amplitudes_used, np.abs(u21), marker='D', label='u21')
    plt.plot(amplitudes_used, np.abs(u22), marker='D', label='u22')
    plt.plot(amplitudes_used, np.abs(u33), marker='D', label='u33')

    plt.xlabel('Time(ns)')
    plt.ylabel('Mat_el')
    plt.title('Mat_el as Functions of Pulse Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(exp.iname, bbox_inches="tight")

    # plot angles as a function of amplitude
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes_used, theta, marker='o', label='Theta')
    plt.plot(amplitudes_used, phi, marker='x', label='Phi')
    plt.plot(amplitudes_used, delta_p, marker='s', label='Delta Plus')
    plt.plot(amplitudes_used, delta_m, marker='^', label='Delta Minus')
    plt.plot(amplitudes_used, delta_moff, marker='d',
             label='Delta Minus_off')  # Something wrong with the code, let's check

    plt.xlabel('Time(ns)')
    plt.ylabel('Angle')
    plt.title('Angles Values as Functions of Pulse Length')
    plt.legend()
    plt.grid(True)
    plt.savefig((exp.iname).rstrip(".png") + "_angles.png", bbox_inches="tight")

# Simulate control waveforms
# reset_time = 1e-6
# element_1['reset_ticks'] = int(reset_time/4e-9)
#
# tomo = StateTomography(element_1, element_2, coupler, circuit_macro, macro_args, shots, calib_params, cfg=config, outer_folder=outer_folder, path='StateTomography', suffix='sim', confusion=SS.confusion, run=False)
#
# tomo.simulate(['1','2','3','4','5','6','7','8','9','10'],10000,[[1400,1500],[4800,4900],[8400,8500],[11900,12000]])




# def analyze(tomo): #don't delete the commented out matrix elements u's. still have to verify what works. Right now the uncommented seems to work
    # raw_data = tomo.data
    #
    # # data_holder = []
    # #
    # # u11_xp =  (raw_data[0][0]/shots + raw_data[0][1]/shots - raw_data[0][2]/shots  - raw_data[0][3]/shots)
    # # u11_yp =  (raw_data[1][0] / shots + raw_data[1][1] / shots - raw_data[1][2] / shots - raw_data[1][3] / shots)
    # # u11 = u11_xp +1j*u11_yp
    # #
    # # u12_xp = (raw_data[2][0] / shots + raw_data[2][1] / shots - raw_data[2][2] / shots - raw_data[2][3] / shots)
    # # u12_yp = (raw_data[3][0] / shots + raw_data[3][1] / shots - raw_data[3][2] / shots - raw_data[3][3] / shots)
    # # u12 = u12_xp + 1j * u12_yp
    # #
    # # u22_xp = (raw_data[2][0] / shots - raw_data[2][1] / shots + raw_data[2][2] / shots - raw_data[2][3] / shots)
    # # u22_yp = (raw_data[3][0] / shots - raw_data[3][1] / shots + raw_data[3][2] / shots - raw_data[3][3] / shots)
    # # u22 = u22_xp + 1j * u22_yp
    # #
    # # u21_xp = (raw_data[0][0] / shots - raw_data[0][1] / shots + raw_data[0][2] / shots - raw_data[0][3] / shots)
    # # u21_yp = (raw_data[1][0] / shots - raw_data[1][1] / shots + raw_data[1][2] / shots - raw_data[1][3] / shots)
    # # u21 = u21_xp + 1j * u21_yp
    # #
    # # u12e_xp = (raw_data[4][0] / shots + raw_data[4][1] / shots - raw_data[4][2] / shots - raw_data[4][3] / shots)
    # # u12e_yp = (raw_data[5][0] / shots + raw_data[5][1] / shots - raw_data[5][2] / shots - raw_data[5][3] / shots)
    # # u12e = u12e_xp + 1j * u12e_yp
    # #
    # # u22e_xp = (raw_data[4][0] / shots - raw_data[4][1] / shots + raw_data[4][2] / shots - raw_data[4][3] / shots)
    # # u22e_yp = (raw_data[5][0] / shots - raw_data[5][1] / shots + raw_data[5][2] / shots - raw_data[5][3] / shots)
    # # u22e = u22e_xp + 1j * u22e_yp
    # #
    # # if np.abs(u11) < np.abs(u21):
    # #     u33 = u12e/np.conj(u21)
    # # else:
    # #     u33 = u22e/np.conj(u11)
    #
    # # u11_xp = 0.5 * (raw_data[0][0] / shots - raw_data[0][1] / shots + raw_data[0][2] / shots - raw_data[0][3] / shots)
    # # u11_yp = 0.5 * (raw_data[1][0] / shots - raw_data[1][1] / shots + raw_data[1][2] / shots - raw_data[1][3] / shots)
    # # u11 = u11_xp + 1j * u11_yp
    # #
    # # u12_xp = 0.5 * (raw_data[2][0] / shots - raw_data[2][1] / shots + raw_data[2][2] / shots - raw_data[2][3] / shots)
    # # u12_yp = 0.5 * (raw_data[3][0] / shots - raw_data[3][1] / shots + raw_data[3][2] / shots - raw_data[3][3] / shots)
    # # u12 = u12_xp + 1j * u12_yp
    # #
    # # u22_xp = 0.5 * (raw_data[2][0] / shots + raw_data[2][1] / shots - raw_data[2][2] / shots - raw_data[2][3] / shots)
    # # u22_yp = 0.5 * (raw_data[3][0] / shots + raw_data[3][1] / shots - raw_data[3][2] / shots - raw_data[3][3] / shots)
    # # u22 = u22_xp + 1j * u22_yp
    # #
    # # u21_xp = 0.5 * (raw_data[0][0] / shots + raw_data[0][1] / shots - raw_data[0][2] / shots - raw_data[0][3] / shots)
    # # u21_yp = 0.5 * (raw_data[1][0] / shots + raw_data[1][1] / shots - raw_data[1][2] / shots - raw_data[1][3] / shots)
    # # u21 = u21_xp + 1j * u21_yp
    # #
    # # u12e_xp = 0.5 * (raw_data[4][0] / shots - raw_data[4][1] / shots + raw_data[4][2] / shots - raw_data[4][3] / shots)
    # # u12e_yp = 0.5 * (raw_data[5][0] / shots - raw_data[5][1] / shots + raw_data[5][2] / shots - raw_data[5][3] / shots)
    # # u12e = u12e_xp + 1j * u12e_yp
    # #
    # # u22e_xp = 0.5 * (raw_data[4][0] / shots + raw_data[4][1] / shots - raw_data[4][2] / shots - raw_data[4][3] / shots)
    # # u22e_yp = 0.5 * (raw_data[5][0] / shots + raw_data[5][1] / shots - raw_data[5][2] / shots - raw_data[5][3] / shots)
    # # u22e = u22e_xp + 1j * u22e_yp
    # #
    # # if np.abs(u11) < np.abs(u21):
    # #     u33 = u12e / np.conj(u21)
    # # else:
    # #     u33 = u22e / np.conj(u11)
    #
    # u11_xp =  (raw_data[0][0] / shots - raw_data[0][1] / shots + raw_data[0][2] / shots - raw_data[0][3] / shots)
    # u11_yp =  (raw_data[1][0] / shots - raw_data[1][1] / shots + raw_data[1][2] / shots - raw_data[1][3] / shots)
    # u11 = u11_xp + 1j * u11_yp
    #
    # u12_xp =  (raw_data[2][0] / shots - raw_data[2][1] / shots + raw_data[2][2] / shots - raw_data[2][3] / shots)
    # u12_yp =  (raw_data[3][0] / shots - raw_data[3][1] / shots + raw_data[3][2] / shots - raw_data[3][3] / shots)
    # u12 = u12_xp + 1j * u12_yp
    #
    # u22_xp = (raw_data[2][0] / shots + raw_data[2][1] / shots - raw_data[2][2] / shots - raw_data[2][3] / shots)
    # u22_yp = (raw_data[3][0] / shots + raw_data[3][1] / shots - raw_data[3][2] / shots - raw_data[3][3] / shots)
    # u22 = u22_xp + 1j * u22_yp
    #
    # u21_xp = (raw_data[0][0] / shots + raw_data[0][1] / shots - raw_data[0][2] / shots - raw_data[0][3] / shots)
    # u21_yp = (raw_data[1][0] / shots + raw_data[1][1] / shots - raw_data[1][2] / shots - raw_data[1][3] / shots)
    # u21 = u21_xp + 1j * u21_yp
    #
    # u12e_xp = (raw_data[4][0] / shots - raw_data[4][1] / shots + raw_data[4][2] / shots - raw_data[4][3] / shots)
    # u12e_yp = (raw_data[5][0] / shots - raw_data[5][1] / shots + raw_data[5][2] / shots - raw_data[5][3] / shots)
    # u12e = u12e_xp + 1j * u12e_yp
    #
    # u22e_xp = (raw_data[4][0] / shots + raw_data[4][1] / shots - raw_data[4][2] / shots - raw_data[4][3] / shots)
    # u22e_yp = (raw_data[5][0] / shots + raw_data[5][1] / shots - raw_data[5][2] / shots - raw_data[5][3] / shots)
    # u22e = u22e_xp + 1j * u22e_yp
    #
    # if np.abs(u11) < np.abs(u21):
    #     u33 = u12e / np.conj(u21)
    # else:
    #     u33 = u22e / np.conj(u11)
    #
    #
    #
    #
    #    # prob_dict = {'00':raw_data[i][0]/shots, '01':raw_data[i][1]/shots,'10':raw_data[i][2]/shots,'11':raw_data[i][3]/shots}
    #
    #
    #
    # # for r in tomo_init_data.analysis_results():
    # #     print(r)
    #
    # print("u11 is ", u11, np.abs(u11))
    # print("u12 is ", u12, np.abs(u12))
    # print("u21 is ", u21, np.abs(u21))
    # print("u22 is ", u22, np.abs(u22))
    # print("u33 is ", u33, np.abs(u33))
    #
    # #theta
    # theta = np.arctan(np.abs(u12)/np.abs(u11))
    #
    # if np.abs(u21) > np.abs(u11):
    #     delta_p = np.angle(-u11 * u21)
    # else:
    #     delta_p = np.angle(-u11 * u22)
    #
    # if np.abs(u21) > np.abs(u11):
    #     phi = delta_p - np.angle(-u12e * u21)
    # else:
    #     phi = np.angle(u22)  - np.angle(u22e)
    #
    # delta_m = 2 * np.angle(u11) - delta_p
    # psi_10 = 1e-9*(meta_dict['q2']['q_LO']['LO_freq'] + meta_dict['q2']['q_IF'] - meta_dict['q1']['q_LO']['LO_freq'] - meta_dict['q1']['q_IF']) * meta_dict['global_params']['iSWAP_len']
    # psi_10 = 0
    # delta_moff = -2 * (np.angle(-u12/1j) + psi_10 ) + delta_p
    #
    # print("\ntheta is ", theta, np.abs(theta))
    # print("phi is ", phi, np.abs(phi))
    # print("delta_p is ", delta_p, np.abs(delta_p))
    # print("delta_m is ", delta_m, np.abs(delta_m))
    # print("delta_moff is ", delta_moff, np.abs(delta_moff))
    #
    # uni_tomo = {'mat_el': {'u11': u11, 'u12': u12, 'u21': u21, 'u22': u22, 'u33': u33},
    #             'angles': {'theta': theta, 'phi': phi, 'delta_p': delta_p, 'delta_m': delta_m, 'delta_moff': delta_moff}
    #
    #             }
    #
    #
    # with open(outer_folder + 'uni_tomo.pkl', 'wb') as f:
    #     pickle.dump(uni_tomo, f)


# analyze(tomo)
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
