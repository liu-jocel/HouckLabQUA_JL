import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

plt.ion()
import copy

# CHANGE THIS DEPENDING ON WHICH META DICT TO MAKE
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Gates.m_ramsey_frequency import RamseyFrequency
from LabCode.Experiments.Gates.m_ramsey_phase import RamseyPhase
from LabCode.Experiments.Gates.m_ramsey_anharmonicity import RamseyAnharmonicity
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_amplified_phase_error import APE
from LabCode.Experiments.Gates.m_APE_1D import APE1D
from LabCode.Experiments.Gates.m_APE_1D_freq import APE1DFreq
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

PULSE_TYPE = "X90"
# PULSE_TYPE = "X180"
experiment_dict = {
    "qubits": {
        "q6": {
            "experiments": {
                "SS_Cal": {
                    "run": True,  # Must run to preform SS experiments
                    "shots": 600,
                    "number_pi_pulses": 1,
                    "ground_threshold": 0.7,
                    "object": SingleShot1Q,
                },
                "Ramsey_Frequency": {
                    "run": False,
                    "shots": 1000,
                    "detuning": 0.5 * u.MHz,
                    "t_min": 16 * u.ns,
                    "t_max": 5 * u.us,
                    "t_step": 32 * u.ns,
                    "object": RamseyFrequency,
                },
                "Ramsey_Anharmonicity": {
                    "run": False,
                    "shots": 400,
                    "detuning": 1 * u.MHz,
                    "t_min": 16 * u.ns,
                    "t_max": 2 * u.us,
                    "t_step": 32 * u.ns,
                    "object": RamseyAnharmonicity,
                },
                "Amplified_Amplitude_Error": {
                    "run": True,
                    "shots": 1000,
                    "N_min": 1,
                    "N_max": 51,
                    "N_step": 1,
                    "object": AAE,
                },
                "Amplified_Phase_Error": {
                    "run": False,
                    "shots": 1000,
                    "N_min": 0,
                    "N_max": 31,
                    "N_step": 1,
                    "freq_span": 0.5 * u.MHz,
                    "freq_points": 5,
                    "object": APE,
                },
                "Ramsey_Y_Cal": {
                    "run": False,
                    "shots": 1000,
                    "tau": 1 * u.us,
                    "phi_min": -0.2,
                    "phi_max": 0.2,
                    "num_phi": 21,
                    "object": RamseyPhase,
                },
                "1D_APE": {
                    "run": False,
                    "shots": 1000,
                    "phi_min": -0.5,
                    "phi_max": 2,
                    "n_vec": [1, 41, 101, 201],
                    "num_phi": 21,
                    "object": APE1D,
                },
                "1D_APE_Frequency": {
                    "run": False,
                    "shots": 1000,
                    "phi_min": -0.5,
                    "phi_max": 2,
                    "num_phi": 51,
                    "freq_span": 1 * u.MHz,
                    "freq_points": 11,
                    "n_vec": [1, 3, 5, 11],
                    "object": APE1DFreq,
                },
            },
        },
    },
}


print("Running the Following Experiments:")
print("####################################################")
for qubit, qubit_dict in experiment_dict["qubits"].items():
    for experiment, experiment_params in qubit_dict["experiments"].items():
        if experiment_params["run"]:
            print(f"{qubit} {experiment}")
print("####################################################")

for qubit, qubit_dict in experiment_dict["qubits"].items():
    meta_dict = make_meta_dict(qubit)[qubit]

    experiment_dict["outer_folder"] = meta_dict["global_params"]["outer_folder"]
    config = meta_dict_to_config_1Q(meta_dict)
    LOs = [meta_dict["q_LO"], meta_dict["r_LO"]]
    print(LOs)
    # set_LO_parameters(LOs)

    for experiment, experiment_params in qubit_dict["experiments"].items():
        if experiment == "SS_Cal" and experiment_params["run"]:
            fidelity = 0
            while fidelity < 0.1:
                SS = SingleShot1Q(
                    qubit,
                    experiment_params["shots"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    repeats=experiment_params["number_pi_pulses"],
                    config=config,
                    confidence_threshold=experiment_params["ground_threshold"],
                )
                qubit_dict["calib_params"] = SS.calib_params
                fidelity = SS.max_F
                meta_dict["read_theta"] = SS.calib_params["read_theta"]
                config = meta_dict_to_config_1Q(meta_dict)
                print(SS.calib_params)
                print(f"SS Cal for {qubit} complete")
        if experiment == "Ramsey_Frequency" and experiment_params["run"]:
            t_vec = np.arange(
                experiment_params["t_min"],
                experiment_params["t_max"],
                experiment_params["t_step"],
            )
            exp = RamseyFrequency(
                qubit,
                t_vec,
                experiment_params["shots"],
                experiment_params["detuning"],
                calib_params=qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"T2R {qubit} complete")

        if experiment == "Ramsey_Anharmonicity" and experiment_params["run"]:
            t_vec = np.arange(
                experiment_params["t_min"],
                experiment_params["t_max"],
                experiment_params["t_step"],
            )
            exp = RamseyAnharmonicity(
                qubit,
                t_vec,
                experiment_params["shots"],
                experiment_params["detuning"],
                calib_params=qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Ramsey Anharmonicity {qubit} complete")

        if experiment == "Amplified_Amplitude_Error" and experiment_params["run"]:
            N_vec = np.arange(
                experiment_params["N_min"],
                experiment_params["N_max"],
                experiment_params["N_step"],
            )
            exp = AAE(
                qubit,
                N_vec,
                experiment_params["shots"],
                calib_params=qubit_dict["calib_params"],
                pulse_type=PULSE_TYPE,
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            # exp = AAE(qubit, N_vec, experiment_params['shots'], calib_params=qubit_dict['calib_params'],
            #           pulse_type="Y"+PULSE_TYPE[1:], meta_dict=meta_dict, outer_folder=experiment_dict['outer_folder'],
            #           path=qubit, suffix=experiment, config=config)
            print(f"Ramsey Anharmonicity {qubit} complete")

        if experiment == "Amplified_Phase_Error" and experiment_params["run"]:
            N_vec = np.arange(
                experiment_params["N_min"],
                experiment_params["N_max"],
                experiment_params["N_step"],
            )
            freq_vec = np.linspace(
                -experiment_params["freq_span"] / 2,
                experiment_params["freq_span"] / 2,
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = APE(
                qubit,
                freq_vec,
                N_vec,
                experiment_params["shots"],
                calib_params=qubit_dict["calib_params"],
                pulse_type=PULSE_TYPE,
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Ramsey Anharmonicity {qubit} complete")

        if experiment == "Ramsey_Y_Cal" and experiment_params["run"]:
            phi_vec = np.linspace(
                experiment_params["phi_min"],
                experiment_params["phi_max"],
                experiment_params["num_phi"],
            )

            exp = RamseyPhase(
                qubit,
                phi_vec,
                experiment_params["tau"],
                experiment_params["shots"],
                calib_params=qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Ramsey_Y_Cal {qubit} complete")

        if experiment == "1D_APE" and experiment_params["run"]:
            phi_vec = np.linspace(
                experiment_params["phi_min"],
                experiment_params["phi_max"],
                experiment_params["num_phi"],
            )
            exp = APE1D(
                qubit,
                phi_vec,
                experiment_params["n_vec"],
                experiment_params["shots"],
                calib_params=qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"1D_APE_Frequency {qubit} complete")

        if experiment == "1D_APE_Frequency" and experiment_params["run"]:
            phi_vec = np.linspace(
                experiment_params["phi_min"],
                experiment_params["phi_max"],
                experiment_params["num_phi"],
            )
            freq_vec = np.linspace(
                -experiment_params["freq_span"] / 2,
                experiment_params["freq_span"] / 2,
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = APE1DFreq(
                qubit,
                phi_vec,
                freq_vec,
                experiment_params["n_vec"],
                experiment_params["shots"],
                calib_params=qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"1D_APE {qubit} complete")
