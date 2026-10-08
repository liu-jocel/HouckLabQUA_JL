import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import atexit
import matplotlib.pyplot as plt
import tkinter as tk


def cleanup():
    try:
        plt.close("all")
        if tk._default_root:
            tk._default_root.quit()
            tk._default_root.destroy()
    except Exception:
        pass


atexit.register(cleanup)
plt.close("all")
import copy

# CHANGE THIS DEPENDING ON WHICH META DICT TO MAKE
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q

# from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_ss_1Q3L_stark import SingleShot1Q3LevelStark
from LabCode.Experiments.Gates.m_stark_ramsey import ACStarkRamsey
from LabCode.Experiments.Coherence.m_T1_stark import ACStarkT1
from LabCode.Experiments.Coherence.m_TLS_count_Stark import ACStarkTLS
from LabCode.Experiments.Coherence.m_TLS_count_Stark_T2 import ACStarkTLST2

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses.
"""

experiment_dict = {
    "infinite_loop": False,
    "qubits": {
        "q2": {
            "experiments": {
                "SS_Cal": {
                    "run": True,  # Must run to preform SS experiments
                    "shots": 1000,
                    "number_pi_pulses": 1,
                    "ground_threshold": 0.7,
                    "object": SingleShot1Q,
                },
                "AC_Stark_Ramsey": {
                    "run": False,
                    "shots": 500,
                    "stark_detuning": 5 * u.MHz,
                    "stark_amp": 0.35,
                    "ramsey_detuning": 0.3 * u.MHz,
                    "t_min": 16 * u.ns,
                    "t_max": 1 * u.us,
                    "t_step": 16 * u.ns,
                    "object": ACStarkRamsey,
                },
                "AC_Stark_T1": {
                    "run": False,
                    "shots": 500,
                    "stark_detuning": -50 * u.MHz,
                    "stark_amp": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                    "t_min": 1 * u.us,
                    "t_max": 3000 * u.us,
                    "t_points": 11,
                    "object": ACStarkT1,
                },
                "AC_Stark_Population": {
                    "run": True,
                    "shots": 500,
                    "stark_detuning": 5 * u.MHz,
                    "amp_min": 0,
                    "amp_max": 0.3,
                    "num_amps": 500,
                    "tau": 300 * u.us,
                    "object": ACStarkTLS,
                },
                "AC_Stark_SS_Cal_3L": {  # Used for calibration of TLS-induced MISTS
                    "run": False,
                    "shots": 10000,
                    "stark_detuning": -50 * u.MHz,
                    "stark_amp": [0, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4],
                    "object": SingleShot1Q3LevelStark,
                },
                "AC_Stark_Population_T2": {
                    "run": False,
                    "shots": 500,
                    "stark_detuning": 50 * u.MHz,
                    "amp_min": 0,
                    "amp_max": 0.15,
                    "num_amps": 501,
                    "tau": 1 * u.us,  # total Fre evolution time of echo experiment
                    "object": ACStarkTLST2,
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


def calc_phase(amp_vec, tau_ns):
    detun = 354.873 * np.array(amp_vec) ** 1.921 * 1e-3  # GHz
    phase = detun * tau_ns
    return np.mod(phase, 1)


while True:
    for qubit, qubit_dict in experiment_dict["qubits"].items():
        meta_dict = make_meta_dict(qubit)[qubit]

        experiment_dict["outer_folder"] = meta_dict["global_params"]["outer_folder"]
        config = meta_dict_to_config_1Q(meta_dict)
        LOs = [meta_dict["q_LO"], meta_dict["r_LO"]]
        print(LOs)
        # set_LO_parameters(LOs)

        for experiment, experiment_params in qubit_dict["experiments"].items():
            if experiment == "SS_Cal" and experiment_params["run"]:
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
            if experiment == "AC_Stark_Ramsey" and experiment_params["run"]:
                MD = copy.deepcopy(meta_dict)
                MD["cw_amp"] = experiment_params["stark_amp"]
                MD["cw_len"] = experiment_params["t_max"]
                config_copy = meta_dict_to_config_1Q(MD)

                t_vec = np.arange(
                    experiment_params["t_min"],
                    experiment_params["t_max"],
                    experiment_params["t_step"],
                )
                exp = ACStarkRamsey(
                    qubit,
                    t_vec,
                    experiment_params["shots"],
                    experiment_params["ramsey_detuning"],
                    experiment_params["stark_detuning"],
                    experiment_params["stark_amp"],
                    calib_params=qubit_dict["calib_params"],
                    meta_dict=MD,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config_copy,
                )
                print(f"AC Stark Ramsey for {qubit} complete")
            if experiment == "AC_Stark_T1" and experiment_params["run"]:
                t_vec = np.logspace(
                    np.log10(experiment_params["t_min"]),
                    np.log10(experiment_params["t_max"]),
                    experiment_params["t_points"],
                )

                for stark_amp in experiment_params["stark_amp"]:
                    MD = copy.deepcopy(meta_dict)
                    MD["cw_amp"] = stark_amp
                    MD["cw_len"] = experiment_params["t_max"]
                    config_copy = meta_dict_to_config_1Q(MD)

                    print(type(config["waveforms"]["ps_wf_q2"]["sample"]))
                    print(config["waveforms"]["ps_wf_q2"]["sample"])
                    exp = ACStarkT1(
                        qubit,
                        t_vec,
                        experiment_params["shots"],
                        experiment_params["stark_detuning"],
                        stark_amp,
                        calib_params=qubit_dict["calib_params"],
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment,
                        meta_dict=MD,
                        config=config_copy,
                        num_pulses=1,
                    )
                    print(f"T1 {qubit} complete for Stark amp {stark_amp}")
            if experiment == "AC_Stark_SS_Cal_3L" and experiment_params["run"]:
                for stark_amp in experiment_params["stark_amp"]:
                    MD = copy.deepcopy(meta_dict)
                    MD["cw_amp"] = stark_amp
                    MD["cw_len"] = meta_dict["read_len"]
                    config_copy = meta_dict_to_config_1Q(MD)
                    exp = SingleShot1Q3LevelStark(
                        qubit,
                        experiment_params["shots"],
                        experiment_params["stark_detuning"],
                        stark_amp,
                        meta_dict=MD,
                        outer_folder=experiment_dict["outer_folder"],
                        path=f"{qubit}",
                        suffix=experiment,
                        config=config_copy,
                    )
                    print(
                        f"AC Stark SS Cal 3L for {qubit} complete for stark_amp={stark_amp}"
                    )
            if experiment == "AC_Stark_Population" and experiment_params["run"]:
                amp_vec = np.linspace(
                    experiment_params["amp_min"],
                    experiment_params["amp_max"],
                    experiment_params["num_amps"],
                )

                MD = copy.deepcopy(meta_dict)
                MD["cw_amp"] = experiment_params["amp_max"]
                MD["cw_len"] = experiment_params["tau"]
                config_copy = meta_dict_to_config_1Q(MD)
                exp = ACStarkTLS(
                    qubit,
                    experiment_params["shots"],
                    experiment_params["tau"],
                    amp_vec,
                    experiment_params["stark_detuning"],
                    calib_params=qubit_dict["calib_params"],
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    meta_dict=MD,
                    config=config_copy,
                )
            if experiment == "AC_Stark_Population_T2" and experiment_params["run"]:
                amp_vec = np.linspace(
                    experiment_params["amp_min"],
                    experiment_params["amp_max"],
                    experiment_params["num_amps"],
                )

                MD = copy.deepcopy(meta_dict)
                MD["cw_amp"] = experiment_params["amp_max"]
                MD["cw_len"] = experiment_params["tau"] / 2
                config_copy = meta_dict_to_config_1Q(MD)
                phase_vec = calc_phase(amp_vec, experiment_params["tau"] / 2)
                print(phase_vec)
                exp = ACStarkTLST2(
                    qubit,
                    experiment_params["shots"],
                    experiment_params["tau"],
                    amp_vec,
                    phase_vec,
                    experiment_params["stark_detuning"],
                    calib_params=qubit_dict["calib_params"],
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    meta_dict=MD,
                    config=config_copy,
                )
    if not experiment_dict["infinite_loop"]:
        break
