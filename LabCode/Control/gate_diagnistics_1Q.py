import numpy as np
import matplotlib
import copy
# CHANGE THIS DEPENDING ON WHICH META DICT TO MAKE
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Gates.m_ramsey_frequency import RamseyFrequency
from LabCode.Experiments.Gates.m_phase_cal_XY import Phase_Calibration_XY
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.Gates.m_all_XY_detuning import AllXYDetuning
from LabCode.Experiments.Gates.m_all_XY import AllXY
from LabCode.Experiments.Gates.m_gate_trajectory import GateTrajectory
from LabCode.Experiments.Gates.m_bootstrap_tomography import BootstrapTomography
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

experiment_dict = {
    'qubits':{
        'q6': {
            "experiments": {
                'SS_Cal': {
                    'run': True,  # Must run to preform SS experiments
                    'shots': 1000,
                    'number_pi_pulses': 1,
                    'ground_threshold': 0.7,
                    'object': SingleShot1Q,
                },
                'Ramsey_Frequency': {
                    'run': False,
                    'shots': 400,
                    'detuning': 1 * u.MHz,
                    't_min': 16 * u.ns,
                    't_max': 2 * u.us,
                    't_step': 32 * u.ns,
                    'object': RamseyFrequency,
                },
                'Gate_Trajectory':{
                    'run': False,
                    'shots': 10000,
                    'num_t': 41,
                    'pulse': "Y90",
                    'object': GateTrajectory
                },
                'Phase_Calibration_XY': {
                    'run': False,
                    'shots': 1000,
                    'pulse_sequence':  [("X90", "I"),
                                        ("Y90", "I"),
                                        ("X90", "Y90"),
                                        ("Y90", "X90"),],
                                        # ("X90", "Y90", "Y90"),
                                        # ("Y90", "X90", "X90"),
                                        # ("X90", "X90", "Y90"),
                                        # ("Y90", "Y90", "X90"),
                                        # ("X90", "X90", "X90"),
                                        # ("Y90", "Y90", "Y90"),],
                    'phase_min': -.10,
                    'phase_max': 0.10,
                    'num_phases': 31,
                    'num_repeats': 1,
                    'object': Phase_Calibration_XY

                },
                'Frequency_Calibration_XY': {
                    'run': True,
                    'shots': 300,
                    'pulse_sequence': [("X90", "I"),
                                        ("Y90", "I"),
                                        ("X90", "Y90"),
                                        ("Y90", "X90")],
                    'freq_span': 3 * u.MHz,
                    'freq_points': 201,
                    'num_repeats': 1,
                    'object': Frequency_Calibration_XY
                },
                'All_XY_Frequency': {
                    'run': False,
                    'shots': 3000,
                    'freq_span': 2 * u.MHz,
                    'freq_points': 11,
                    'num_circuits': 1,
                    'object': AllXY,
                },
                'All_XY': {
                    'run':False,
                    'shots': 1000,
                    'num_circuits': 1,
                    'object': AllXY,
                },
                'BootstrapTomography': {
                    'run':False,
                    'shots': 500,
                    #'num_circuits': 1,
                    'object': BootstrapTomography,
                },
            },
        },
    },
}



print("Running the Following Experiments:")
print("####################################################")
for qubit, qubit_dict in experiment_dict['qubits'].items():
    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment_params['run']:
            print(f"{qubit} {experiment}")
print("####################################################")

for qubit, qubit_dict in experiment_dict['qubits'].items():

    meta_dict = make_meta_dict(qubit)[qubit]

    experiment_dict['outer_folder'] = meta_dict['global_params']['outer_folder']
    config = meta_dict_to_config_1Q(meta_dict)
    LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
    print(LOs)
    # set_LO_parameters(LOs)

    for experiment, experiment_params in qubit_dict['experiments'].items():
        if experiment == "SS_Cal" and experiment_params['run']:
            fidelity = 0
            while fidelity < 0.1:
                SS = SingleShot1Q(qubit, experiment_params['shots'], meta_dict=meta_dict,
                                  outer_folder=experiment_dict['outer_folder'], path=qubit,
                                  suffix=experiment, repeats=experiment_params['number_pi_pulses'],
                                  config=config, confidence_threshold=experiment_params['ground_threshold'])
                qubit_dict['calib_params'] = SS.calib_params
                fidelity = SS.max_F
                meta_dict['read_theta'] = SS.calib_params["read_theta"]
                config = meta_dict_to_config_1Q(meta_dict)
                print(SS.calib_params)
                print(f"SS Cal for {qubit} complete")
        if experiment == "Ramsey_Frequency" and experiment_params['run']:
            t_vec = np.arange(experiment_params['t_min'], experiment_params['t_max'], experiment_params['t_step'])
            exp = RamseyFrequency(qubit, t_vec, experiment_params['shots'], experiment_params['detuning'],
                      calib_params=qubit_dict['calib_params'], meta_dict=meta_dict,
                      outer_folder=experiment_dict['outer_folder'],
                      path=qubit, suffix=experiment, config=config)
            print(f"T2R {qubit} complete")

        if experiment == "Gate_Trajectory" and experiment_params['run']:
            exp = GateTrajectory(qubit, experiment_params['pulse'], experiment_params['num_t'], experiment_params['shots'],
                                  calib_params=qubit_dict['calib_params'], meta_dict=meta_dict,
                                  outer_folder=experiment_dict['outer_folder'],
                                  path=qubit, suffix=experiment, config=config)
            print(f"T2R {qubit} complete")

        if experiment == "Phase_Calibration_XY" and experiment_params['run']:
            phase_vec = np.linspace(experiment_params['phase_min'], experiment_params['phase_max'],
                                    experiment_params['num_phases'])

            exp = Phase_Calibration_XY(qubit, phase_vec, experiment_params['pulse_sequence'],
                             experiment_params['shots'], qubit_dict['calib_params'],
                               meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                               path=qubit, suffix=experiment, config=config)

        if experiment == "Frequency_Calibration_XY" and experiment_params['run']:
            freq_vec = np.linspace(-experiment_params['freq_span'] / 2, experiment_params['freq_span'] / 2,
                                   experiment_params['freq_points'], dtype=int)

            exp = Frequency_Calibration_XY(qubit, freq_vec, experiment_params['pulse_sequence'],
                                       experiment_params['shots'], qubit_dict['calib_params'],
                                       num_repeats=experiment_params['num_repeats'],
                                       meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                       path=qubit, suffix=experiment, config=config)
        if experiment == "All_XY_Frequency" and experiment_params['run']:
            freq_vec = np.linspace(-experiment_params['freq_span'] / 2, experiment_params['freq_span'] / 2,
                                   experiment_params['freq_points'], dtype=int)

            exp = AllXYDetuning(qubit, freq_vec, experiment_params['shots'], qubit_dict['calib_params'],
                                       num_repeats=experiment_params['num_circuits'],
                                       meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                                       path=qubit, suffix=experiment, config=config)
        if experiment == "All_XY" and experiment_params['run']:
            exp = AllXY(qubit, experiment_params['shots'],
                        qubit_dict['calib_params'],
                               num_pulses=experiment_params['num_circuits'], meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                               path=qubit, suffix=experiment, config=config)
        if experiment == "BootstrapTomography" and experiment_params['run']:
            exp = BootstrapTomography(qubit, experiment_params['shots'],
                        qubit_dict['calib_params'],    # or pass a custom list of pulse tuples
                               meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
                               path=qubit, suffix=experiment, config=config)