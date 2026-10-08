import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') ### Using the 'Agg' backend so that won't have memory issues saving figures
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
import qiskit

backend = AerSimulator.from_backend(FakeParis())

outer_folder = 'Y:/Charlie/Data/TwoQubit/TQ6B2/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'


amps = [0.07] # [0.07,0.08,0.09,
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
        # SS_1 = SingleShot1Q('q1', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q1')
        # SS_2 = SingleShot1Q('q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='Q2')

        SS = SingleShot2Q('q1', 'q2', shots, meta_dict=meta_dict, outer_folder=outer_folder, path='SingleShotCal', suffix='2Q')

        # calib_params_1 = SS_1.calib_params
        # calib_params_2 = SS_2.calib_params
        calib_params = SS.calib_params

        a_min = 0.02
        a_max = 0.5
        num_a = 30
        shots = 500

        # exp = Rabi_2Q(element_1, element_2, a_min, a_max, num_a, shots, calib_params, cfg=config, outer_folder=outer_folder, path='Rabi', suffix='', confusion = SS.confusion)

        # exp = SWAP_Osc(element_1, element_2, coupler, a_min, a_max, num_a, shots, calib_params, cfg=config, outer_folder=outer_folder, path='SWAP', suffix='', confusion = SS.confusion)

        t_min = 16e-9
        t_max = 600e-9
        num_t = 20
        shots = 400

        time_cal = SWAP_Osc_time('q1', 'q2', 'c1', t_min, t_max, num_t, shots, calib_params, meta_dict=meta_dict,
                                 outer_folder=outer_folder, path='SWAP', suffix='', confusion = SS.confusion, fit=True)

        # circ = qiskit.QuantumCircuit(2)
        # circ.x(0)
        # circ.x(1)
        # tomo_qiskit = StateTomographyQiskit(circ)
        # analysis_qiskit = tomo_qiskit.analysis
        # tomo_init_data = tomo_qiskit.run(backend, seed_simulation=100).block_for_results()

        def circuit_macro(element_1, element_2, coupler, t):
            align()
            play('X180', element_1)
            align()
            flat_top_CZ(coupler, t)

        times = time_cal.clk_vec
        purities = np.zeros(num_t)

        for j in range(num_t):
            tomo = StateTomography('q1', 'q2', 'c1', circuit_macro, times[j], shots, calib_params, meta_dict=meta_dict,
                                   outer_folder=outer_folder, path='StateTomography', confusion=SS.confusion)

            purities[j] = tomo.data[1].purity().real
            print(j,purities)

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.plot(time_cal.t_vec*1e9, purities, 'o', label='State purity')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('State purity')
        ax.set_title('SWAP Oscillation Purity')
        ax.legend()
        plt.savefig(tomo.iname[:-4]+'_purity.png', bbox_inches="tight")
        plt.close(fig)