import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict_CR
from LabCode.Config.config_helpers import meta_dict_to_config_CR
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_2Q import SingleShot2Q
from LabCode.Experiments.CR.m_CR_time import TimeRabiCR
from LabCode.Experiments.CR.m_CR_Hamiltonian_Tomography import HamiltonianTomographyCR
from LabCode.Experiments.CR.m_CR_amplitude_calibration import AmpCalibrationCR
from LabCode.Experiments.CR.m_CR_phase_calibration import PhaseCalibrationCR
from LabCode.Experiments.CR.m_CR_cancel_amplitude_calibration import CancelAmpCalibrationCR
from LabCode.Experiments.CR.m_CR_cancel_phase_calibration import CancelPhaseCalibrationCR
from LabCode.Experiments.CR.m_CNOT import CNOT
from LabCode.Experiments.CR.m_CR_ramsey import RamseyCR
from LabCode.Experiments.Readout_Cal.m_active_reset_2Q import ActiveResetCal2Q
from LabCode.Experiments.CR.m_ZZ import ZZ
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate CR pulses
"""

meta_dict = make_meta_dict_CR()
outer_folder = meta_dict["outer_folder"]
config = meta_dict_to_config_CR(meta_dict)

# Make sure LO's for each qubit are the same!!
LOs = [meta_dict['q1']['q_LO'], meta_dict['q1']['r_LO']]
# set_LO_parameters(LOs)

experiment_dict = {
    'meta_dict': meta_dict,
    'outer_folder': outer_folder,
    "experiments": {
        "SS_Cal_2Q":{
            'run': True,  # Must run to preform SS experiments
            'shots': 1000,
            'ground_state_threshold': 0.7,
            'object': SingleShot2Q,
        },
        "Readout_Cal_2Q": {
            'run': False,
            'shots': 200,
            "a_min_1": 0.01,
            "a_max_1": 0.02,
            "num_a_1": 11,
            "a_min_2": 0.01,
            "a_max_2": 0.020,
            "num_a_2": 11,
        },
        "Active_Reset_Cal_2Q":{
            'run': False,
            "shots": 100,
            "t_min": 16 * u.ns,
            "t_max": 5 * u.us,
            't_step': 100 * u.ns,
        },
        "ZZ": {
            'run': False,
            "shots": 500,
            "t_min": 16 * u.ns,
            "t_max": 10 * u.us,
            'num_t': 50,
            'detuning': 2 * u.MHz,
            "object": ZZ,
        },
        "CR_Time_Rabi": {
            "run": False,
            "echo": True,
            "cancel": False,
            "shots": 300,
            "t_min": 8 * u.ns,
            "t_max": 2000 * u.ns,
            't_step': 64 * u.ns, # Make sure divisible by 8 for echo
            "object": TimeRabiCR,
        },
        "CR_Hamiltonian_Tomography":{
            "run": True,
            "echo": True,
            "cancel": True,
            "shots": 1000,
            "t_min": 8 * u.ns,
            "t_max": 1000 * u.ns,
            't_step': 16 * u.ns, # Make sure divisible by 8 for echo
            "object": HamiltonianTomographyCR,

        },
        "CR_Amp_Cal":{
            "run": False,
            "echo": True,
            "cancel": True,
            "shots": 200,
            "pulse_reps": 1,
            "a_min": 0.2,
            "a_max": 0.45,
            "num_a": 6,
            "t_min": 8 * u.ns,
            "t_max": 2000 * u.ns,
            't_step': 64 * u.ns,  # Make sure divisible by 8 for echo
            "object": AmpCalibrationCR,
        },
        "CR_Phase_Cal":{
            "run": False,
            "echo": True,
            "cancel": False,
            "shots": 200,
            "pulse_reps": 1,
            "p_min": -0.5,
            "p_max": 0,
            "num_p": 8,
            "t_min": 8 * u.ns,
            "t_max": 1000 * u.ns,
            't_step': 16 * u.ns,  # Make sure divisible by 8 for echo
            "object": PhaseCalibrationCR,
        },
        "CR_Cancel_Amp_Cal":{
            "run": False,
            "echo": True,
            "cancel": True,
            "shots": 300,
            "pulse_reps": 1,
            "a_min": 0.01,
            "a_max": 0.1,
            "num_a": 10,
            "t_min": 8 * u.ns,
            "t_max": 1000 * u.ns,
            't_step': 16 * u.ns,  # Make sure divisible by 8 for echo
            "object": CancelAmpCalibrationCR,
        },
        "CR_Cancel_Phase_Cal":{
            "run": False,
            "echo": True,
            "cancel": True,
            "shots": 200,
            "pulse_reps": 0,
            "p_min": 5.7,
            "p_max": 5.75,
            "num_p": 4,
            "t_min": 8 * u.ns,
            "t_max": 4000 * u.ns,
            't_step': 64 * u.ns,  # Make sure divisible by 8 for echo
            "object": CancelPhaseCalibrationCR,
        },
        "CR_Ramsey":{
            "run": False,
            "shots": 300,
            "p_min": 0,
            "p_max": 2 * np.pi,
            "num_p": 21,
            "object": RamseyCR,
        },
        "CNOT":{
            "run": True,
            "shots": 1000,
            "object": CNOT,
        },
    },
}


print("Running the Following Experiments:")
print("####################################################")
for experiment, experiment_params in experiment_dict['experiments'].items():
    if experiment_params['run']:
        print(f"{experiment}")
print("####################################################")

PATH = "CR"
for experiment, experiment_params in experiment_dict['experiments'].items():
    if experiment == "SS_Cal_2Q" and experiment_params['run']:
        SS = None
        trace = 0
        while trace/4 < 0.82:
            SS = SingleShot2Q("q1", "q2", experiment_params['shots'], meta_dict=meta_dict,
                              outer_folder=experiment_dict['outer_folder'], path=PATH,
                              suffix=experiment, config=config, confidence_threshold=experiment_params['ground_state_threshold'])
            print(SS.confusion)
            trace = np.trace(np.array(SS.confusion))
            print(trace/4)
        SS.pickle_data()
        experiment_dict['calib_params'] = SS.calib_params
        experiment_dict['confusion_matrix'] = SS.confusion
        meta_dict["q1"]['read_theta'] = SS.calib_params["read_theta_q1"]
        meta_dict["q2"]['read_theta'] = SS.calib_params["read_theta_q2"]
        config = meta_dict_to_config_CR(meta_dict)
        print("SS Cal Complete")
    if experiment == "Readout_Cal_2Q" and experiment_params['run']:
        amp_vec_1 = np.linspace(experiment_params['a_min_1'], experiment_params['a_max_1'], experiment_params['num_a_1'])
        amp_vec_2 = np.linspace(experiment_params['a_min_2'], experiment_params['a_max_2'], experiment_params['num_a_2'])
        print(
            f'Readout_Cal_2Q est. completion time: {len(amp_vec_1) * len(amp_vec_2) * experiment_params["shots"] * meta_dict["global_params"]["reset_time"] / u.s / 60} Minutes')

        fidels = np.zeros([len(amp_vec_1), len(amp_vec_2)])
        for j, a2 in enumerate(amp_vec_2):
            meta_dict["q2"]['read_amp'] = a2
            for i, a1 in enumerate(amp_vec_1):
                meta_dict["q1"]['read_amp'] = a1
                config = meta_dict_to_config_CR(meta_dict)
                SS = SingleShot2Q("q1", "q2", experiment_params['shots'], meta_dict=meta_dict,
                              outer_folder=experiment_dict['outer_folder'], path=PATH,
                              suffix=experiment, config=config, plot=False)
                if i == 0:
                    if j == 0:
                        name = SS.iname[:-6] + '2Q'
                average_f = np.trace(SS.confusion)/4
                fidels[i][j] = average_f
                if average_f > 0.85:
                    print(average_f)
                    print(f" Amp_1: {a1}")
                    print(f" Amp_2: {a2}")

                vmin = np.max([np.min([fidels]), 0.5])
                vmax = np.max([fidels])

                a1_idx_max, a2_idx_max = np.unravel_index(np.argmax(fidels), fidels.shape)

                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                pcm1 = ax.imshow(fidels, vmin=vmin, vmax=vmax, origin='lower')
                # plt.gca().invert_yaxis()
                ax.set_xticks(range(len(amp_vec_1)))
                ax.set_yticks(range(len(amp_vec_2)))
                ax.set_xticklabels(np.round(amp_vec_2, decimals=4))
                ax.set_yticklabels(np.round(amp_vec_1, decimals=4))


                ax.set_xlabel('Readout amplitude q1 (V)')
                ax.set_ylabel('Readout amplitude (V)')
                ax.set_title(
                    f"Readout Cal 2Q, Amp1: {amp_vec_1[a1_idx_max]} Amp2: {amp_vec_2[a2_idx_max]} ")
                fig.colorbar(pcm1, ax=ax)
                plt.savefig(name + '2QC', bbox_inches='tight')
                plt.close()

        print(f"Optimal amps, A1: {amp_vec_1[a1_idx_max]}, A2: {amp_vec_2[a2_idx_max]}")

    if experiment == "Active_Reset_Cal_2Q" and experiment_params['run']:
        t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                          dtype=int)
        exp = ActiveResetCal2Q(t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                         experiment_dict['confusion_matrix'], meta_dict=meta_dict, config=config,
                         outer_folder=experiment_dict['outer_folder'], path=PATH, suffix=experiment,)


    if experiment == "ZZ" and experiment_params['run']:
        SS = ZZ("q1", "q2", experiment_params["t_min"], experiment_params['t_max'], experiment_params['num_t'],
                experiment_params['shots'], experiment_params['detuning'], experiment_dict['calib_params'], meta_dict=meta_dict, config=config, outer_folder=experiment_dict['outer_folder'], path=PATH,
                          suffix=experiment)

    if experiment == "CR_Time_Rabi" and experiment_params['run']:
        if experiment_params['echo']:
            t_vec = np.arange(experiment_params["t_min"]/2, experiment_params['t_max']/2, experiment_params['t_step']/2,
                              dtype=int)
        else:
            t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                              dtype=int)
        exp = TimeRabiCR(t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                         experiment_dict['confusion_matrix'], meta_dict=meta_dict, config=config,
                         echo=experiment_params['echo'], cancel=experiment_params['cancel'],
                         outer_folder=experiment_dict['outer_folder'], path=PATH, suffix=experiment,)

    if experiment == "CR_Hamiltonian_Tomography" and experiment_params['run']:
        print(f"Running Hamilton Tomography, Echo: { experiment_params['echo']}, Cancel: { experiment_params['cancel']}")
        if experiment_params['echo']:
            t_vec = np.arange(experiment_params["t_min"] / 2, experiment_params['t_max'] / 2,
                              experiment_params['t_step'] / 2, dtype=int)
        else:
            t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                              dtype=int)
        exp = HamiltonianTomographyCR(t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                         experiment_dict['confusion_matrix'], meta_dict=meta_dict, config=config,
                         echo=experiment_params['echo'], cancel=experiment_params['cancel'],
                         outer_folder=experiment_dict['outer_folder'], )


    if experiment == "CR_Amp_Cal" and experiment_params['run']:
        # uses echo experiment
        if experiment_params['echo']:
            t_vec = np.arange(experiment_params["t_min"] / 2, experiment_params['t_max'] / 2,
                              experiment_params['t_step'] / 2,
                              dtype=int)
        else:
            t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                              dtype=int)
        amp_vec = np.linspace(experiment_params['a_min'], experiment_params['a_max'], experiment_params['num_a'])

        exp = AmpCalibrationCR(amp_vec, t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                               experiment_dict['confusion_matrix'], echo=experiment_params['echo'],
                               cancel=experiment_params['cancel'],meta_dict=meta_dict, config=config,
                               outer_folder=experiment_dict['outer_folder'], path=PATH,
                               suffix=experiment,)

    if experiment == "CR_Phase_Cal" and experiment_params['run']:
        # uses echo experiment
        if experiment_params['echo']:
            t_vec = np.arange(experiment_params["t_min"] / 2, experiment_params['t_max'] / 2,
                              experiment_params['t_step'] / 2,
                              dtype=int)
        else:
            t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                              dtype=int)
        p_vec = np.linspace(experiment_params['p_min'], experiment_params['p_max'], experiment_params['num_p'])

        exp = PhaseCalibrationCR(p_vec, t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                               experiment_dict['confusion_matrix'], echo=experiment_params['echo'],
                               cancel=experiment_params['cancel'],meta_dict=meta_dict, config=config,
                               outer_folder=experiment_dict['outer_folder'], path=PATH,
                               suffix=experiment,)

    if experiment == "CR_Cancel_Amp_Cal" and experiment_params['run']:
        if experiment_params['echo']:
            t_vec = np.arange(experiment_params["t_min"] / 2, experiment_params['t_max'] / 2,
                              experiment_params['t_step'] / 2,
                              dtype=int)
        else:
            t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                              dtype=int)
        amp_vec = np.linspace(experiment_params['a_min'], experiment_params['a_max'], experiment_params['num_a'])
        exp = CancelAmpCalibrationCR(amp_vec, t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                         experiment_dict['confusion_matrix'], echo=experiment_params['echo'],
                               cancel=experiment_params['cancel'],meta_dict=meta_dict, config=config,
                               outer_folder=experiment_dict['outer_folder'], path=PATH,
                               suffix=experiment,)

    if experiment == "CR_Cancel_Phase_Cal" and experiment_params['run']:
        if experiment_params['echo']:
            t_vec = np.arange(experiment_params["t_min"] / 2, experiment_params['t_max'] / 2,
                              experiment_params['t_step'] / 2,
                              dtype=int)
        else:
            t_vec = np.arange(experiment_params["t_min"], experiment_params['t_max'], experiment_params['t_step'],
                              dtype=int)
        p_vec = np.linspace(experiment_params['p_min'], experiment_params['p_max'], experiment_params['num_p'])

        exp = CancelPhaseCalibrationCR(p_vec, t_vec, experiment_params['shots'], experiment_dict['calib_params'],
                               experiment_dict['confusion_matrix'],  echo=experiment_params['echo'],
                               cancel=experiment_params['cancel'],meta_dict=meta_dict, config=config,
                               outer_folder=experiment_dict['outer_folder'], path=PATH,
                               suffix=experiment,)

    if experiment == "CR_Ramsey" and experiment_params['run']:
        p_vec = np.linspace(experiment_params['p_min'], experiment_params['p_max'], experiment_params['num_p'])
        exp = RamseyCR(p_vec, experiment_params['shots'], experiment_dict['calib_params'],
                               experiment_dict['confusion_matrix'], meta_dict=meta_dict, config=config,
                               outer_folder=experiment_dict['outer_folder'], path=PATH,
                               suffix=experiment,)
    if experiment == "CNOT" and experiment_params['run']:
        exp = CNOT(experiment_params['shots'], experiment_dict['calib_params'],
                               experiment_dict['confusion_matrix'], meta_dict=meta_dict, config=config,
                               outer_folder=experiment_dict['outer_folder'], path=PATH,
                               suffix=experiment,)