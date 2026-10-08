import numpy as np
import matplotlib
import copy
# CHANGE THIS DEPENDING ON WHICH META DICT TO MAKE
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Gates.m_ramsey_phase  import RamseyPhase
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_amplified_phase_error import APE
from LabCode.Experiments.Gates.m_all_XY import AllXY




from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

PULSE_TYPE = "X90"
# PULSE_TYPE = "X180"
QUBIT = "q6"
optimizer_convergence_params ={
    "amplitude_accuracy": 0.00005,
    'detuning_accuracy': 0.001 * u.MHz,
    'phase_accuracy': 0.005,
    "max_iter": 10,
}
experiment_dict = {
            'SS_Cal': {
                'shots': 300,
                'number_pi_pulses': 1,
                'ground_threshold': 0.7,
                'object': SingleShot1Q,
            },
            'Amplified_Amplitude_Error':{
                'shots': 300,
                'N_min': 1,
                'N_max': 11,
                'N_step': 1,
                'object': AAE,
            },
            'Frequency_Calibration_XY': {
                'run': False,
                'shots': 100,
                'pulse_sequence': [("X90", "I"),
                                   ("Y90", "I"),
                                   ("X90", "Y90"),
                                   ("Y90", "X90"),],
                                #    ("X90", "Y90", "Y90"),
                                #    ("Y90", "X90", "X90"),
                                #    ("X90", "X90", "Y90"),
                                #    ("Y90", "Y90", "X90"),
                                #    ("X90", "X90", "X90"),
                                #    ("Y90", "Y90", "Y90"), ],
                'freq_span': 5 * u.MHz,
                'freq_points': 301,
                'num_repeats': 1,
                'object': Frequency_Calibration_XY
            },
            'Amplified_Phase_Error': {
                'shots': 1000,
                'N_min': 0,
                'N_max': 5,
                'N_step': 1,
                'freq_span': 2 * u.MHz,
                'freq_points': 5,
                'object': APE,
            },
            'Ramsey_Y_Cal': {
                'shots': 300,
                'tau': 1 * u.us,
                'phi_min': -0.15,
                'phi_max': 0.15,
                'num_phi': 51,
                'object': RamseyPhase
    },
}


#Set LO for Expertiment
meta_dict = make_meta_dict(QUBIT)[QUBIT]
config = meta_dict_to_config_1Q(meta_dict)
LOs = [meta_dict['q_LO'], meta_dict['r_LO']]
# set_LO_parameters(LOs)
CALIB_PARAMS=None

# Calibrate Readout
SS = SingleShot1Q(QUBIT, experiment_dict["SS_Cal"]['shots'], meta_dict=meta_dict,
                  outer_folder=meta_dict['global_params']['outer_folder'], path=QUBIT,
                  suffix="SS_Cal", repeats=experiment_dict["SS_Cal"]['number_pi_pulses'],
                  config=config, confidence_threshold=experiment_dict["SS_Cal"]['ground_threshold'])
CALIB_PARAMS = SS.calib_params
fidelity = SS.max_F
meta_dict['read_theta'] = SS.calib_params["read_theta"]
config = meta_dict_to_config_1Q(meta_dict)
i = 0

prev = dict(
    amplitude=0.0,
    detuning=0.0,
    phase=0.0,
)

calibrated = dict(
    amplitude=False,
    detuning=False,
    phase=False,
)

while i < optimizer_convergence_params["max_iter"]:

    # ---------- Amplitude ----------
    N_vec = np.arange(
        experiment_dict["Amplified_Amplitude_Error"]['N_min'],
        experiment_dict["Amplified_Amplitude_Error"]['N_max'],
        experiment_dict["Amplified_Amplitude_Error"]['N_step']
    )

    aae = AAE(
        QUBIT, N_vec,
        experiment_dict["Amplified_Amplitude_Error"]['shots'],
        calib_params=CALIB_PARAMS,
        pulse_type=PULSE_TYPE,
        meta_dict=meta_dict,
        outer_folder=meta_dict['global_params']['outer_folder'],
        path=QUBIT,
        suffix="Amplified_Amplitude_Error",
        config=config
    )

    amp_err = aae.data['amplitude_error']
    meta_dict['X90']['amp'] += amp_err
    meta_dict['Y90']['amp'] += amp_err

    # ---------- Detuning ----------
    # N_vec = np.arange(
    #     experiment_dict["Amplified_Phase_Error"]['N_min'],
    #     experiment_dict["Amplified_Phase_Error"]['N_max'],
    #     experiment_dict["Amplified_Phase_Error"]['N_step']
    # )

    freq_vec = np.linspace(
        -experiment_dict["Frequency_Calibration_XY"]['freq_span'] / 2,
        experiment_dict["Frequency_Calibration_XY"]['freq_span'] / 2,
        experiment_dict["Frequency_Calibration_XY"]['freq_points'],
        dtype=int
    )

    freq_ALL_XY = Frequency_Calibration_XY(QUBIT, freq_vec, experiment_dict["Frequency_Calibration_XY"]['pulse_sequence'],
                                   experiment_dict["Frequency_Calibration_XY"]['shots'], CALIB_PARAMS,
                                   num_repeats=experiment_dict["Frequency_Calibration_XY"]['num_repeats'],
                                   meta_dict=meta_dict, outer_folder=meta_dict['global_params']['outer_folder'],
                                   path=QUBIT, suffix="Frequency_Calibration_XY", config=config)
    # ape = APE(
    #     QUBIT, freq_vec, N_vec,
    #     experiment_dict["Amplified_Phase_Error"]['shots'],
    #     calib_params=CALIB_PARAMS,
    #     pulse_type=PULSE_TYPE,
    #     meta_dict=meta_dict,
    #     outer_folder=meta_dict['global_params']['outer_folder'],
    #     path=QUBIT,
    #     suffix='Amplified_Phase_Error',
    #     config=config
    # )

    det_err = freq_ALL_XY.data['optimal_detuning_MHz']
    meta_dict['detuning'] = det_err

    # # ---------- Phase ----------
    # phi_vec = np.linspace(
    #     experiment_dict["Ramsey_Y_Cal"]['phi_min'],
    #     experiment_dict["Ramsey_Y_Cal"]['phi_max'],
    #     experiment_dict["Ramsey_Y_Cal"]['num_phi']
    # )

    # ramsey_phase = RamseyPhase(
    #     QUBIT, phi_vec,
    #     experiment_dict["Ramsey_Y_Cal"]['tau'],
    #     experiment_dict["Ramsey_Y_Cal"]['shots'],
    #     calib_params=CALIB_PARAMS,
    #     meta_dict=meta_dict,
    #     outer_folder=meta_dict['global_params']['outer_folder'],
    #     path=QUBIT,
    #     suffix="Ramsey_Y_Cal",
    #     config=config
    # )

    phase_err = 0#*ramsey_phase.data['phase']
    meta_dict['Y90']['phase'] += phase_err

    # ---------- Convergence Check ----------
    calibrated['amplitude'] = abs(amp_err) < optimizer_convergence_params['amplitude_accuracy']
    calibrated['detuning'] = abs(det_err - prev['detuning']) < optimizer_convergence_params['detuning_accuracy']
    calibrated['phase'] = abs(phase_err-prev['phase']) < optimizer_convergence_params['phase_accuracy']

    prev['amplitude'] = amp_err
    prev['detuning'] = det_err
    prev['phase'] = phase_err

    # ---------- Logging ----------
    not_calibrated = [k for k, v in calibrated.items() if not v]

    print(
        f"Iter {i:02d} | "
        f"amp_err = {amp_err:+.3e}  "
        f"det_err = {det_err:+.3e}  "
        f"phase_err = {phase_err:+.3e}  | "
        f"calibrated = {calibrated}"
    )

    if not_calibrated:
        print(f"    Not yet calibrated: {', '.join(not_calibrated)}")
        config = meta_dict_to_config_1Q(meta_dict)

    if all(calibrated.values()):
        print(f"\n Calibration converged after {i+1} iterations")
        print("\nFinal calibration parameters:")
        print("-" * 40)
        print(f"X90 amplitude : {meta_dict['X90']['amp']}")
        print(f"Y90 amplitude : {meta_dict['Y90']['amp']}")
        print(f"Detuning (MHz): {meta_dict['detuning']}")
        print(f"Y90 phase     : {meta_dict['Y90']['phase']}")
        print("-" * 40)

        break
    i += 1

exp = AllXY(QUBIT, 1000,
            CALIB_PARAMS,
            num_pulses=1, meta_dict=meta_dict,
            outer_folder=meta_dict['global_params']['outer_folder'],
            path=QUBIT, suffix="ALLXY", config=config)