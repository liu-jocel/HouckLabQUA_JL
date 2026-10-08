import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from LabCode2.Config.config_helpers import meta_dict_to_config_2Q
from LabCode2.PythonDrivers.control_8257 import set_LO_parameters
from LabCode2.CoreLib.experiment import ExperimentClass
from LabCode2.Experiments.m_T1 import T1
from LabCode2.Experiments.m_T2R import T2R
from LabCode2.Experiments.m_T2E import T2E
from LabCode2.Experiments.m_transmission import Transmission
from LabCode2.Experiments.m_qubit_spec import QubitSpec
from LabCode2.Experiments.m_Rabi import Rabi
from LabCode2.Experiments.m_Rabi_2Q import Rabi_2Q
from LabCode2.Experiments.m_SWAP_osc_amp import SWAP_Osc
from LabCode2.Experiments.m_SWAP_osc_time import SWAP_Osc as SWAP_Osc_time
from LabCode2.Experiments.m_SWAP_osc_phase import SWAP_Osc_phase
from LabCode2.Experiments.m_single_shot_1Q import SingleShot1Q
from LabCode2.Experiments.m_single_shot_2Q import SingleShot2Q
from LabCode2.Experiments.m_state_tomo import StateTomography
from LabCode2.Helpers.gate_macros import flat_top_CZ
import Drivers.YOKOGS200 as YOKOGS200
from qm.qua import *
import datetime
import qiskit
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import StateTomography as StateTomographyQiskit

from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeParis

backend = AerSimulator.from_backend(FakeParis())

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ6B2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'


# amps = [0.07,0.08,0.09,0.093,0.094,0.095]
amps = [0.07]
LO_freqs = [1.8335e9]
# LO_freqs = 1.8335e9 + np.linspace(-0.5e6,0.5e6,11)
for i, a in enumerate(amps):
    for l in LO_freqs:

        # Get metaDict
        with open(pathPickle, 'rb') as f:
            meta_dict = pickle.load(f)

        Q1 = meta_dict['q1']
        Q2 = meta_dict['q2']
        coupler = meta_dict['c1']
        coupler['c_LO']['LO_freq'] = l#1.8335e9  # 1.8335e9 - 0e6
        meta_dict['c1']['c_IF'] = 100e6

        LOs = [Q1['q_LO'], Q1['r_LO'], Q2['q_LO'], Q2['r_LO'], coupler['c_LO']]

        # set_LO_parameters(LOs)

        meta_dict['c1']['pulse_amp'] = a #0.033 #0.2
        meta_dict['c1']['c_LO']['LO_freq'] = coupler['c_LO']['LO_freq']

        with open(pathPickle, 'wb') as f:
            pickle.dump(meta_dict,f)

        #
        shots = 1000
        #
        SS_1 = SingleShot1Q('q1', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q1')
        SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')

        SS = SingleShot2Q('q1', 'q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='2Q')

        calib_params_1 = SS_1.calib_params
        calib_params_2 = SS_2.calib_params
        calib_params = SS.calib_params

        a_min = 0.02
        a_max = 0.5
        num_a = 30
        shots = 500

        # exp = Rabi_2Q(element_1, element_2, a_min, a_max, num_a, shots, calib_params, cfg=config, outer_folder=outer_folder, path='Rabi', suffix='', confusion = SS.confusion)

        # exp = SWAP_Osc(element_1, element_2, coupler, a_min, a_max, num_a, shots, calib_params, cfg=config, outer_folder=outer_folder, path='SWAP', suffix='', confusion = SS.confusion)

        t_min = 16e-9
        t_max = 600e-9
        num_t = 80
        shots = 400

        time_cal = SWAP_Osc_time('q1', 'q2', 'c1', t_min, t_max, num_t, shots, calib_params, meta_dict=meta_dict,
                                 outer_folder=outer_folder, path='SWAP', suffix='', confusion = SS.confusion, fit=True)
        #
        # num_phis = 50
        # phase_cal_Q1 = SWAP_Osc_phase(element_1, element_2, element_1, element_2, coupler, num_phis, shots, calib_params_1, cfg=config, outer_folder=outer_folder, path='SWAP', suffix='', confusion = SS.confusion)
        # phase_cal_Q2 = SWAP_Osc_phase(element_1, element_2, element_2, element_1, coupler, num_phis, shots, calib_params_2, cfg=config, outer_folder=outer_folder, path='SWAP', suffix='', confusion = SS.confusion)


            # # # SWAP Chevron
            # num_detune = 81 #401 # 2 points per MHz
            # detuning_vec = np.linspace(-20e6,20e6,num_detune)
            # # detuning_vec = np.linspace(-100e6,100e6,num_detune)
            # f_0 = meta_dict['c1']['c_LO']['LO_freq']
            # data_list = []
            #
            # for i,d in enumerate(detuning_vec):
            #
            #     coupler['c_LO']['LO_freq'] = f_0 + d
            #
            #     LOs = [Q1['q_LO'], Q1['r_LO'], Q2['q_LO'], Q2['r_LO'], coupler['c_LO']]
            #
           #  #     set_LO_parameters(LOs)
            #
            #     exp = SWAP_Osc_time('q1', 'q2', 'c1', t_min, t_max, num_t, shots, calib_params, meta_dict=meta_dict,
            #                    outer_folder=outer_folder, path='SWAP', suffix='',plot=False)
            #
            #     data_list.append(exp.data)
            #
            #     print(i)
            #
            #
            # data_00 = np.zeros([num_t, num_detune])
            # data_01 = np.zeros([num_t, num_detune])
            # data_10 = np.zeros([num_t, num_detune])
            # data_11 = np.zeros([num_t, num_detune])
            #
            #
            # for i in range(num_t):
            #     for j in range(num_detune):
            #         data_00[i][j] = data_list[j][1][0][i]/shots
            #         data_01[i][j] = data_list[j][1][1][i]/shots
            #         data_10[i][j] = data_list[j][1][2][i]/shots
            #         data_11[i][j] = data_list[j][1][3][i]/shots
            #
            # vmin = np.min([data_00,data_01,data_10,data_11])
            # vmax = np.max([data_00,data_01,data_10,data_11])
            #
            # fig, ax = plt.subplots(2,2,constrained_layout=True)
            #
            # coupler_freqs = f_0+detuning_vec+meta_dict['c1']['c_IF']
            # pcm1 = ax[0][0].pcolormesh(coupler_freqs, exp.t_vec, data_00, vmin=vmin, vmax=vmax)
            # pcm2 = ax[0][1].pcolormesh(coupler_freqs, exp.t_vec, data_01, vmin=vmin, vmax=vmax)
            # pcm3 = ax[1][0].pcolormesh(coupler_freqs, exp.t_vec, data_10, vmin=vmin, vmax=vmax)
            # pcm4 = ax[1][1].pcolormesh(coupler_freqs, exp.t_vec, data_11, vmin=vmin, vmax=vmax)
            #
            # ax[0][0].set_title('00')
            # ax[0][1].set_title('01')
            # ax[1][0].set_title('10')
            # ax[1][1].set_title('11')
            #
            # ax[1][0].set_xlabel('Coupler Frequency (Hz)')
            # ax[1][1].set_xlabel('Coupler Frequency (Hz)')
            # ax[1][0].set_ylabel('Pulse time (s)')
            # ax[0][0].set_ylabel('Pulse time (s)')
            # plt.suptitle('SWAP Chevron')
            #
            # fig.colorbar(pcm1, ax=ax)
            # plt.savefig(exp.iname, bbox_inches='tight')
            #
            # chevron_data = [coupler_freqs,exp.t_vec,data_00,data_01,data_10,data_11]
            #
            # datetimestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            # with open(exp.iname.rsplit('\\', 1)[0] + "\\" + datetimestring + '_amp_' + str(a) + '_chevron_data.pkl','wb') as f:
            #     pickle.dump(chevron_data,f)

        # State tomography calibration

        # coupler['c_LO']['LO_freq'] = f_0
        #
        # LOs = [Q1['q_LO'], Q1['r_LO'], Q2['q_LO'], Q2['r_LO'], coupler['c_LO']]
        #
       #  # set_LO_parameters(LOs)

        circ = qiskit.QuantumCircuit(2)
        circ.x(0)
        circ.x(1)

        tomo_qiskit = StateTomographyQiskit(circ)
        analysis_qiskit = tomo_qiskit.analysis
        tomo_init_data = tomo_qiskit.run(backend, seed_simulation=100).block_for_results()

        def circuit_macro(element_1, element_2, coupler, t):

            # pass
            # reset_phase(element_1['q_name'])
            # reset_phase(element_2['q_name'])
            # reset_phase(coupler['c_name'])
            # reset_frame(element_2['q_name'],element_1['q_name'],coupler['c_name'])
            align()
            play('X180', element_2)
            align()
            flat_top_CZ(coupler, t)
            # play('iSWAP_const',coupler['c_name'],duration=t)


        times = time_cal.clk_vec
        purities = np.zeros(num_t)

        for j in range(num_t):
            tomo = StateTomography('q1', 'q2', 'c1', circuit_macro, times[j], shots, calib_params, meta_dict=meta_dict,
                                   outer_folder=outer_folder, path='StateTomography', confusion=SS.confusion)

            # Qiskit data simulation

            # raw_data = tomo.data
            #
            # data_holder = []
            #
            # for i in range(9):
            #     data_holder.append(
            #         {'counts': {'00': raw_data[i][0], '01': raw_data[i][1], '10': raw_data[i][2], '11': raw_data[i][3]},
            #          'metadata': tomo_init_data.data(i)['metadata']})
            #
            # data_holder_qiskit = ExperimentData()
            # data_holder_qiskit.add_data(data_holder)
            #
            # fitted_data = analysis_qiskit.run(data_holder_qiskit)
            #
            # purities[j] = fitted_data.analysis_results('state').value.purity()
            purities[j] = tomo.data[1].purity()
            print(j,purities)

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(time_cal.t_vec*1e9, purities, 'o', label='State purity')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('State purity')
        ax.set_title('SWAP Oscillation Purity')
        ax.legend()
        plt.savefig(tomo.iname[:-4]+'_purity.png', bbox_inches="tight")
        plt.close(fig)