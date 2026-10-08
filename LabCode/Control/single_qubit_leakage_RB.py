import numpy as np
from copy import deepcopy, copy
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Gates.m_amplified_leakage_error import ALE
from LabCode.Experiments.Gates.m_amplified_leakage_error_thermal import ALE_Thermal
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage
from LabCode.Experiments.RB.m_LRB_DRAG_sweep import RB1Q_Leakage_DRAGSweep
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Gates.m_clifford_DRAG_cal import CliffordDRAGCal
from LabCode.Experiments.DRAG.m_DRAG_cal import DragCal1Q
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses up to DRAG.
"""

experiment_dict = {
    "Infinite_Loop": False,
    "qubits": {
        "q2": {
            "experiments": {
                "SS_Cal_3L": {
                    "run": True,  # Must run to preform SS experiments
                    "shots": 3000,
                    "object": SingleShot1Q3Level,
                },
                "Amplified_Leakage_Error": {
                    "run": True,
                    "shots": 400,
                    "theta_min": 0,
                    "theta_max": 1.05,
                    "num_theta": 101,
                    "N_min": 2,
                    "N_max": 2001,
                    "N_step": 10,
                    "object": ALE,
                },
                "Amplified_Leakage_Error_Thermal": {
                    "run": False,
                    "shots": 100,
                    "theta_min": 0,
                    "theta_max": 1.05,
                    "num_theta": 106,
                    "N_min": 2,
                    "N_max": 402,
                    "N_step": 10,
                    "object": ALE_Thermal,
                },
                "RB1Q_Leakage": {
                    "run": True,
                    "shots": 10,
                    "num_circuits": 1000,
                    "circuit_lengths": [
                        2,
                        4,
                        8,
                        16,
                        64,
                        128,
                        250,
                        500,
                        600,
                        1000,
                        1500,
                        2000,
                        2250,
                        2500,
                        2750,
                        3000,
                    ],
                    "object": RB1Q_Leakage,
                },
                "Clifford_DRAG_Cal": {
                    "run": False,
                    "shots": 2,
                    "num_circuits": 500,
                    "circuit_length": 2000,
                    "alpha_min": -2,  # scaling factor
                    "alpha_max": 2,  # scaling factor
                    "alpha_step": 0.01,
                    "object": CliffordDRAGCal,
                },
                "DRAG_Calibation": {
                    "run": False,
                    "shots": 5000,
                    "number_pi_pulses": [2000],  # experiment does X80 followed by mX180
                    "alpha_min": -2,  # Volts
                    "alpha_max": 2,  # Volts. Note: must be less than  2 * pulse amplitude
                    "alpha_step": 0.1,
                    "pulse_type": "X90",
                    "object": DragCal1Q,
                },
                "RB1Q_Leakage_DRAGSweep": {
                    "run": False,
                    "shots": 10,  # shots per (alpha, depth, circuit)
                    "num_circuits": 1000,  # random Clifford sequences per alpha
                    "circuit_lengths": [
                        2,
                        4,
                        8,
                        16,
                        64,
                        250,
                        500,
                        1000,
                        2000,
                        2500,
                        3000,
                    ],
                    "alpha_min": -2,
                    "alpha_max": 2,
                    "alpha_step": 0.1,
                    "object": RB1Q_Leakage_DRAGSweep,
                },
            },
        },
    },
}

print("Running the Following Experiments:")
print("####################################################")
print(f"Infinite Loop: {experiment_dict['Infinite_Loop']}")
print("####################################################")
for qubit, qubit_dict in experiment_dict["qubits"].items():
    for experiment, experiment_params in qubit_dict["experiments"].items():
        if experiment_params["run"]:
            print(f"{qubit} {experiment}")
print("####################################################")
while True:
    for qubit, qubit_dict in experiment_dict["qubits"].items():
        meta_dict = make_meta_dict(qubit)[qubit]

        experiment_dict["outer_folder"] = meta_dict["global_params"]["outer_folder"]
        config = meta_dict_to_config_1Q(meta_dict)
        LOs = [meta_dict["q_LO"], meta_dict["r_LO"]]
        print(LOs)
        # set_LO_parameters(LOs)
        # qubit_dict['calib_params'] = [[0.00018418950401246547, -0.00016395466215908528], [0.00020784268714487553, -2.949855849146843e-05],
        #  [4.481714963912964e-05, -7.216770201921463e-06]]

        for experiment, experiment_params in qubit_dict["experiments"].items():
            if experiment == "SS_Cal_3L" and experiment_params["run"]:
                SS = SingleShot1Q3Level(
                    qubit,
                    experiment_params["shots"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config,
                )

                qubit_dict["calib_params"] = SS.calib_params
                config = meta_dict_to_config_1Q(meta_dict)
                print(f"SS Cal for {qubit} complete")
            if experiment == "Amplified_Leakage_Error" and experiment_params["run"]:
                N_vec = np.arange(
                    experiment_params["N_min"],
                    experiment_params["N_max"],
                    experiment_params["N_step"],
                )
                theta_vec = np.linspace(
                    experiment_params["theta_min"],
                    experiment_params["theta_max"],
                    experiment_params["num_theta"],
                )
                exp = ALE(
                    qubit,
                    theta_vec,
                    N_vec,
                    experiment_params["shots"],
                    qubit_dict["calib_params"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment + "_all90",
                    config=config,
                )

            if experiment == "Amplified_Leakage_Error_Thermal" and experiment_params["run"]:
                N_vec = np.arange(
                    experiment_params["N_min"],
                    experiment_params["N_max"],
                    experiment_params["N_step"],
                )
                theta_vec = np.linspace(
                    experiment_params["theta_min"],
                    experiment_params["theta_max"],
                    experiment_params["num_theta"],
                )
                exp = ALE_Thermal(
                    qubit,
                    theta_vec,
                    N_vec,
                    experiment_params["shots"],
                    qubit_dict["calib_params"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config,
                )

            if experiment == "RB1Q_Leakage" and experiment_params["run"]:
                exp = RB1Q_Leakage(
                    qubit,
                    experiment_params["circuit_lengths"],
                    experiment_params["num_circuits"],
                    experiment_params["shots"],
                    qubit_dict["calib_params"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment + "_all90",
                    config=config,
                )
            if experiment == "Clifford_DRAG_Cal" and experiment_params["run"]:
                alpha_vec = np.arange(
                    experiment_params["alpha_min"],
                    experiment_params["alpha_max"],
                    experiment_params["alpha_step"],
                )
                MD = deepcopy(meta_dict)
                MD["DRAG_beta"] = 1
                config_copy = meta_dict_to_config_1Q(MD)
                print(meta_dict["DRAG_beta"])
                drag_res = CliffordDRAGCal(
                    qubit,
                    alpha_vec,
                    experiment_params["circuit_length"],
                    experiment_params["num_circuits"],
                    experiment_params["shots"],
                    qubit_dict["calib_params"],
                    meta_dict=MD,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config_copy,
                )
            if experiment == "DRAG_Calibation" and experiment_params["run"]:
                MD = deepcopy(meta_dict)
                MD["DRAG_beta"] = 1
                alpha_vec = np.arange(
                    experiment_params["alpha_min"],
                    experiment_params["alpha_max"],
                    experiment_params["alpha_step"],
                )
                config_copy = meta_dict_to_config_1Q(MD)
                exp = DragCal1Q(
                    qubit,
                    alpha_vec,
                    experiment_params["number_pi_pulses"],
                    experiment_params["shots"],
                    qubit_dict["calib_params"],
                    pulse_type=experiment_params["pulse_type"],
                    meta_dict=meta_dict,
                    config=config,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                )
                print(f"DRAG_Calibation for {qubit} complete")
            if experiment == "RB1Q_Leakage_DRAGSweep" and experiment_params["run"]:
                MD = deepcopy(meta_dict)
                MD["DRAG_beta"] = 1
                config_copy = meta_dict_to_config_1Q(MD)
                alpha_vec = np.arange(
                    experiment_params["alpha_min"],
                    experiment_params["alpha_max"],
                    experiment_params["alpha_step"],
                )
                exp = RB1Q_Leakage_DRAGSweep(
                    qubit,
                    alpha_vec,
                    experiment_params["circuit_lengths"],
                    experiment_params["num_circuits"],
                    experiment_params["shots"],
                    qubit_dict["calib_params"],
                    meta_dict=MD,
                    config=config_copy,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                )
                print(f"RB1Q_Leakage_DRAGSweep for {qubit} complete")
    if not experiment_dict["Infinite_Loop"]:
        break
