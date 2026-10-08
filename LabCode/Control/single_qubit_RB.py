import matplotlib
import numpy as np
matplotlib.use('QtAgg')
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2R import T2R
from LabCode.Experiments.Coherence.m_T2E import T2E
from LabCode.Experiments.RB.m_RB_1Q import RB1Q
from LabCode.Experiments.DRAG.m_DRAG_orbit import DRAG_Orbit
#from LabCode.Experiments.RB.m_RB_1Q_Interleaved import RB1Q_Interleaved, calculate_interleaved_gate_fidelity
from LabCode.Experiments.Gates.m_bootstrap_tomography import BootstrapTomography
from LabCode.Experiments.RB.m_RB_purity import RBPurity1Q
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses up to DRAG.
"""

experiment_dict = {
    "Infinite_Loop": False,
    "qubits": {
        "q6": {
            "experiments": {
                "SS_Cal": {
                    "run": True,  # Must run to preform SS experiments
                    "shots": 1000,
                },
                "T1": {
                    "run": False,
                    "shots": 500,
                    "num_pulses": [1],
                    "t_min": 400 * u.ns,
                    "t_max": 5000 * u.us,
                    "t_points": 51,
                    "object": T1,
                },
                "T2R": {
                    "run": False,
                    "shots": 500,
                    "detuning": 0.2 * u.MHz,
                    "t_min": 16 * u.ns,
                    "t_max": 800 * u.us,
                    "t_step": 500 * u.ns,
                    "object": T2R,
                },
                "T2E": {
                    "run": False,
                    "shots": 500,
                    "t_min": 0.5 * u.us,
                    "t_max": 5000 * u.us,
                    "t_points": 51,
                    "object": T2E,
                },
                "RB1Q": {
                    "run": True,
                    "simulate": True,
                    "shots": 100,
                    "num_circuits": 150,
                    "circuit_lenghts": [
                        #2,
                        #4,
                        8,
                        #16,
                        32,
                        #64,
                        128,
                        256,
                        #300,
                        #400,
                        500,
                        #600,
                        700,
                        #800,
                        900,
                        #1000,
                        1200,
                        1500,
                        2000,
                        3000,
                        4000,
                        5000,
                        6000,
                        7000,
                    ],
                    "object": RB1Q,
                },
                "RBPurity1Q": {
                    "run": False,
                    "shots": 1,
                    "num_circuits": 1,
                    "circuit_lengths": [1000],
                    "object": RBPurity1Q,
                },
                "RB1Q_Interleaved": {
                    "run": False,
                    "shots": 1,
                    "num_circuits": 1000,
                    "circuit_lenghts": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1028],
                },
                "DRAG_ORBIT": {
                    "run": False,
                    "shots": 100,
                    "num_circuits": 100,
                    "circuit_lenghts": [2, 16, 64, 128, 500, 1000],
                    "alpha_min": 0,  # scaling factor
                    "alpha_max": 1,  # scaling factor
                    "alpha_step": 0.01,
                },
                "BootstrapTomography": {
                    "run": False,
                    "shots": 1500,
                    #'num_circuits': 1,
                    "object": BootstrapTomography,
                },
                # 'BootstrapTomographySweep': {
                #     'run': False,
                #     'shots': 500,
                #     'detuning_span': 5 * u.MHz,
                #     'detuning_points': 13,
                #     'object': BootstrapTomographySweep,
                # },
            },
        },
    },
}

print("Running the Following Experiments:")
print("####################################################")
print(f"Infinite Loop: {experiment_dict['Infinite_Loop']}")
print("####################################################")
for qubit, qubit_dict in experiment_dict['qubits'].items():
    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment_params['run']:
            print(f"{qubit} {experiment}")
print("####################################################")
while True:
    for qubit, qubit_dict in experiment_dict['qubits'].items():
        meta_dict = make_meta_dict(qubit)[qubit]

        experiment_dict['outer_folder'] = meta_dict['global_params']['outer_folder']
        config = meta_dict_to_config_1Q(meta_dict)
        LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
        print(LOs)
        # set_LO_parameters(LOs)
        for experiment, experiment_params in qubit_dict['experiments'].items():
            if experiment == "SS_Cal" and experiment_params['run']:
                print(f"SS Cal estimated time: {experiment_params['shots'] * meta_dict['global_params']['reset_time']/u.s/60} Minutes.")
                SS = SingleShot1Q(qubit, experiment_params['shots'], meta_dict=meta_dict,
                                  outer_folder=experiment_dict['outer_folder'], path=qubit,
                                    suffix=experiment, config=config)
                qubit_dict['calib_params'] = SS.calib_params
                meta_dict['read_theta'] = SS.calib_params["read_theta"]
                config = meta_dict_to_config_1Q(meta_dict)
                print(SS.calib_params)
                print(f"SS Cal for {qubit} complete")

            if experiment == "T1" and experiment_params['run']:
                t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']),
                                    experiment_params['t_points'])
                if experiment_params['num_pulses']:
                    for num_pusle in experiment_params['num_pulses']:
                        exp = T1(qubit, t_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
                                 meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                 path=qubit, suffix=experiment, config=config, num_pulses=num_pusle)
                else:
                    exp = T1(qubit, t_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
                             meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                             path=qubit, suffix=experiment, config=config)
                print(f"T1 {qubit} complete")

            if experiment == "T2R" and experiment_params['run']:
                t_vec = np.arange(experiment_params['t_min'], experiment_params['t_max'], experiment_params['t_step'])
                exp = T2R(qubit, t_vec, experiment_params['shots'], experiment_params['detuning'],
                          calib_params=qubit_dict['calib_params'], meta_dict=meta_dict,
                          outer_folder=experiment_dict['outer_folder'],
                          path=qubit, suffix=experiment, config=config)
                print(f"T2R {qubit} complete")

            if experiment == "T2E" and experiment_params['run']:
                t_vec = np.logspace(np.log10(experiment_params['t_min']), np.log10(experiment_params['t_max']),
                                    experiment_params['t_points'])
                exp = T2E(qubit, t_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
                          meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                          path=qubit, suffix=experiment, config=config)
                print(f"T2E {qubit} complete")

            if experiment == "RB1Q" and experiment_params['run']:
                total_cycles = len(experiment_params['circuit_lenghts']) * experiment_params['num_circuits'] *experiment_params['shots']
                exp = RB1Q(qubit, experiment_params['circuit_lenghts'], experiment_params['num_circuits'],
                                   experiment_params['shots'], qubit_dict['calib_params'],
                                   meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                   path=qubit, suffix=experiment + "_all90", config=config,
                                   simulate=experiment_params['simulate'])
                _rb_fid = exp.data.get('gate_fidelity', float('nan'))
                if _rb_fid > 0.99999:
                    print(f"*** {qubit} RB gate fidelity {_rb_fid:.8f} > 0.99999 — stopping loop ***")
                    experiment_dict['Infinite_Loop'] = False
            if experiment == "RB1Q_Interleaved" and experiment_params['run']:
                total_cycles = len(experiment_params['circuit_lenghts']) * experiment_params['num_circuits'] *experiment_params['shots']
                print(f"RB1Q est. runtime: {2* meta_dict['global_params']['reset_time']/u.s/60 * total_cycles } Minutes.")
                rb_base = RB1Q(qubit, experiment_params['circuit_lenghts'], experiment_params['num_circuits'],
                                   experiment_params['shots'], qubit_dict['calib_params'],
                                   meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                   path=qubit, suffix=experiment,)
                # rb_int = RB1Q_Interleaved(qubit, experiment_params['interleaved_gate'], experiment_params['circuit_lenghts'],
                #                           experiment_params['num_circuits'], experiment_params['shots'],
                #                           qubit_dict['calib_params'], meta_dict=meta_dict,
                #                           outer_folder=experiment_dict['outer_folder'], path=qubit, suffix=experiment,)
                # print(f'Interleaved gate fidelity: {calculate_interleaved_gate_fidelity(2, rb_base, rb_int)}')

            if experiment == "DRAG_ORBIT" and experiment_params['run']:
                alpha_vec = np.arange(experiment_params['alpha_min'], experiment_params['alpha_max'], experiment_params['alpha_step'])
                total_cycles = experiment_params['num_circuits'] * experiment_params['shots'] * len(alpha_vec) * len(experiment_params['circuit_lenghts'])
                # print(f"RB1Q est. runtime: {full_meta_dict['global_params']['reset_time']/u.s/60 * total_cycles } Minutes.")
                fidels = []
                for a in alpha_vec:
                    meta_dict['DRAG_beta'] = a
                    config = meta_dict_to_config_1Q(meta_dict)
                    print(meta_dict['DRAG_beta'])
                    drag_res  = DRAG_Orbit(qubit, experiment_params['circuit_lenghts'], experiment_params['num_circuits'],
                                   experiment_params['shots'], qubit_dict['calib_params'],
                                   meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                   path=qubit, suffix=experiment, config=config)
                    fidels.append(drag_res.data['gate_fidelity'])

            if experiment == "BootstrapTomography" and experiment_params['run']:
                exp = BootstrapTomography(qubit, experiment_params['shots'],
                                        qubit_dict['calib_params'],    # or pass a custom list of pulse tuples
                                        meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                        path=qubit, suffix=experiment, config=config)

            if experiment == "RBPurity1Q" and experiment_params['run']:
                exp = RBPurity1Q(qubit, experiment_params['circuit_lengths'],
                                 experiment_params['num_circuits'], experiment_params['shots'],
                                 qubit_dict['calib_params'],
                                 meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                 path=qubit, suffix=experiment, config=config)

            # if experiment == "BootstrapTomographySweep" and experiment_params['run']:
            #     detuning_vec = np.linspace(
            #         -experiment_params['detuning_span'] // 2,
            #          experiment_params['detuning_span'] // 2,
            #          experiment_params['detuning_points'], dtype=int)
            #     exp = BootstrapTomographySweep(qubit, experiment_params['shots'],
            #                                    qubit_dict['calib_params'], detuning_vec,
            #                                    meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
            #                                    path=qubit, suffix=experiment, config=config)

    if not experiment_dict['Infinite_Loop']:
        break