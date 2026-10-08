import numpy as np
import matplotlib.pyplot as plt
import atexit
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
from LabCode.Config.make_meta_dict import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.m_transmission import Transmission
from LabCode.Experiments.m_chi import ChiMeasurement
from LabCode.Experiments.m_transmission_sweep import TransmissionSweep
from LabCode.Experiments.m_qubit_spec import QubitSpec
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Rabi.m_Rabi_Chevron_IQ import RabiChevronIQ
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Rabi.m_Rabi_Chevron_Timed_SS import RabiChevronTimedSS
from LabCode.Experiments.Rabi.m_Rabi_linecut_timed_SS import RabiLinecutTimedSS
from LabCode.Experiments.Rabi.m_Rabi_Linecut_SS import RabiLinecutSS
from LabCode.Experiments.Rabi.m_freq_cal import FreqCal1Q
from LabCode.Experiments.DRAG.m_DRAG_cal import DragCal1Q
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses.
"""


PULSE_TYPE = "X90"
experiment_dict = {
    "qubits": {
        "q5": {
            "experiments": {
                "Transmission_Sweep": {
                    "run": False,
                    "shots": 101,
                    "a_min": 0.001,
                    "a_max": 0.45,
                    "num_a": 31,
                    "freq_min": -201 * u.MHz,
                    "freq_max": -198 * u.MHz,
                    "freq_step": 0.005 * u.MHz,
                    "object": TransmissionSweep,
                },
                "Transmission": {
                    "run":False,
                    "shots": 200,
                    "freq_min": -202 * u.MHz,
                    "freq_max": -198 * u.MHz,
                    "freq_step": 0.01 * u.MHz,
                    "object": Transmission,
                },
                "Chi_Measurement": {
                    "run": False,
                    "shots": 100,
                    "freq_min": -201 * u.MHz,
                    "freq_max": -199 * u.MHz,
                    "freq_step": 0.05 * u.MHz,
                    "object": ChiMeasurement,
                },
                "Qubit_Spec": {
                    "run": False,
                    "shots": 800,  # 4000
                    "spec_amp": 0.005,
                    "spec_len": 100 * u.us,
                    "freq_min": -105 * u.MHz,
                    "freq_max": -90 * u.MHz,
                    "freq_step": 0.005 * u.MHz,  # 0.05
                    "object": QubitSpec,
                },
                "SS_Cal": {
                    "run": True,  # Must run to preform SS experiments
                    "shots": 500,
                    "number_pi_pulses": 1,
                    "ground_threshold": 0.7,
                    "print_temp": True,
                    "object": SingleShot1Q,
                },
                "Rabi_Chevron_IQ": {
                    "run": True,  # Must run SS_Cal first to preform Rabi experiments
                    "shots": 400,
                    "number_pi_pulses": 1,
                    "a_min": 0.0,  # Volts
                    "a_max": 0.1,  # Volts note: 2 * pulse amplitude
                    "a_points": 31,
                    "freq_span": 3 * u.MHz,
                    "freq_points": 21,
                    "pulse_type": PULSE_TYPE,
                    "object": RabiChevronIQ,
                },
                "Rabi_Chevron_SS": {
                    "run": True,  # Must run SS_Cal first to preform Rabi experiments
                    "shots": 500,
                    "number_pi_pulses": 1,
                    "a_min": 0.0,
                    "a_max": 0.45,
                    "a_points": 21,
                    "freq_span": 3 * u.MHz,
                    "freq_points": 21,
                    "pulse_type": PULSE_TYPE,
                    "object": RabiChevronSS,
                },
                "Rabi_Chevron_Timed_SS": {
                    "run": False,
                    "shots": 1000,
                    "t_min": 16 * u.ns,
                    "t_max": 0 * u.ns,
                    "t_points": 50,
                    "freq_span": 50 * u.MHz,
                    "freq_points": 41,
                    "object": RabiChevronTimedSS,
                },
                "Rabi_Linecut_Timed_SS": {
                    "run": False,
                    "shots": 100,
                    "t_min": 16 * u.ns,
                    "t_max": 1000 * u.ns,
                    "t_step": 4 * u.ns,
                    "ampx": 0.2,
                    "object": RabiLinecutTimedSS,
                },
                "Rabi_Linecut_SS": {
                    "run": False,
                    "shots": 100,
                    "number_pi_pulses": 11,
                    "a_min": 0.012,  # Volts
                    "a_max": 0.0165,  # Volts note: 2 * pulse amplitude
                    "a_points": 51,
                    "pulse_type": PULSE_TYPE,
                    "object": RabiLinecutSS,
                },
                "Frequency_Calibration": {  # not working
                    "run": False,
                    "shots": 500,
                    "number_pi_pulses": 1,  # experiment does X80 followed by mX180
                    "freq_span": 11 * u.MHz,
                    "freq_points": 51,
                    "pulse_type": PULSE_TYPE,
                    "object": FreqCal1Q,
                },
                "DRAG_Calibation": {
                    "run": False,
                    "shots": 10000,
                    "number_pi_pulses": [1000],  # experiment does X80 followed by mX180
                    "alpha_min": 0,  # Volts
                    "alpha_max": 1.5,  # Volts. Note: must be less than  2 * pulse amplitude
                    "alpha_step": 0.01,
                    "pulse_type": PULSE_TYPE,
                    "object": DragCal1Q,
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
        if experiment == "Transmission_Sweep" and experiment_params["run"]:
            copy_MD = copy.deepcopy(meta_dict)
            copy_MD["read_amp"] = 0.45
            copy_config = meta_dict_to_config_1Q(copy_MD)
            f_vec = np.arange(
                experiment_params["freq_min"],
                experiment_params["freq_max"],
                experiment_params["freq_step"],
            )
            a_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["num_a"],
            )
            exp = TransmissionSweep(
                qubit,
                a_vec,
                f_vec,
                experiment_params["shots"],
                meta_dict=copy_MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=copy_config,
            )
            print(f"Transmission for {qubit} complete")

        if experiment == "Transmission" and experiment_params["run"]:
            f_vec = np.arange(
                experiment_params["freq_min"],
                experiment_params["freq_max"],
                experiment_params["freq_step"],
            )
            exp = Transmission(
                qubit,
                f_vec,
                experiment_params["shots"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Transmission Sweep for {qubit} complete")

        if experiment == "Chi_Measurement" and experiment_params["run"]:
            exp = ChiMeasurement(
                qubit,
                experiment_params["freq_min"],
                experiment_params["freq_max"],
                experiment_params["freq_step"],
                experiment_params["shots"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Transmission for {qubit} complete")

        if experiment == "Qubit_Spec" and experiment_params["run"]:
            # set_LO_parameters([meta_dict['q_LO']])
            MD = copy.deepcopy(meta_dict)
            MD["cw_amp"] = experiment_params["spec_amp"]
            MD["cw_len"] = experiment_params["spec_len"]
            config_copy = meta_dict_to_config_1Q(MD)
            exp = QubitSpec(
                qubit,
                experiment_params["freq_min"],
                experiment_params["freq_max"],
                experiment_params["freq_step"],
                experiment_params["shots"],
                meta_dict=MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config_copy,
            )
            print(f"Spec for {qubit} complete")

        if experiment == "SS_Cal" and experiment_params["run"]:
            SS = SingleShot1Q(
                qubit,
                experiment_params["shots"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )

            if SS.max_F < 0:
                print(f"SS Cal for {qubit} failed, passing to next")
                break
            else:
                qubit_dict["calib_params"] = SS.calib_params
                meta_dict["read_theta"] = SS.calib_params["read_theta"]
                config = meta_dict_to_config_1Q(meta_dict)
                print(f"SS Cal for {qubit} complete")

        if experiment == "Rabi_Chevron_IQ" and experiment_params["run"]:
            MD = copy.deepcopy(meta_dict)
            MD[PULSE_TYPE]["amp"] = experiment_params["a_max"]
            config_copy = meta_dict_to_config_1Q(MD)
            amp_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["a_points"],
            )
            freq_vec = np.linspace(
                -experiment_params["freq_span"] / 2,
                experiment_params["freq_span"] / 2,
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = RabiChevronIQ(
                qubit,
                amp_vec,
                freq_vec,
                experiment_params["shots"],
                experiment_params["number_pi_pulses"],
                pulse_type=experiment_params["pulse_type"],
                meta_dict=MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config_copy,
            )
            print(f"Rabi Chevron IQ for {qubit} complete")

        if experiment == "Rabi_Chevron_Timed_SS" and experiment_params["run"]:
            t_vec = np.arange(
                experiment_params["t_min"],
                experiment_params["t_max"],
                (experiment_params["t_max"] - experiment_params["t_min"])
                // experiment_params["t_points"],
            )
            freq_vec = np.linspace(
                -experiment_params["freq_span"] / 2,
                experiment_params["freq_span"] / 2,
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = RabiChevronTimedSS(
                qubit,
                t_vec,
                freq_vec,
                experiment_params["shots"],
                qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Rabi_Chevron_Timed_SS for {qubit} complete")

        if experiment == "Rabi_Linecut_Timed_SS" and experiment_params["run"]:
            # run time rabi
            time_rabi_t_vec = np.arange(
                experiment_params["t_min"],
                experiment_params["t_max"],
                experiment_params["t_step"],
            ).astype(int)
            exp1 = RabiLinecutTimedSS(
                qubit,
                time_rabi_t_vec,
                experiment_params["shots"],
                ampx=experiment_params["ampx"],
                calib_params=qubit_dict["calib_params"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Time Rabi for {qubit} complete")

        if experiment == "Rabi_Chevron_SS" and experiment_params["run"]:
            MD = copy.deepcopy(meta_dict)
            MD[PULSE_TYPE]["amp"] = experiment_params["a_max"]
            config_copy = meta_dict_to_config_1Q(MD)
            amp_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["a_points"],
            )
            freq_vec = np.linspace(
                -experiment_params["freq_span"] / 2,
                experiment_params["freq_span"] / 2,
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = RabiChevronSS(
                qubit,
                amp_vec,
                freq_vec,
                experiment_params["shots"],
                qubit_dict["calib_params"],
                experiment_params["number_pi_pulses"],
                pulse_type=experiment_params["pulse_type"],
                meta_dict=MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config_copy,
            )
            print(f"Rabi_Chevron_SS for {qubit} complete")

        if experiment == "Rabi_Linecut_SS" and experiment_params["run"]:
            MD = copy.deepcopy(meta_dict)
            MD[PULSE_TYPE]["amp"] = experiment_params["a_max"]
            config_copy = meta_dict_to_config_1Q(MD)
            amp_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["a_points"],
            )
            exp = RabiLinecutSS(
                qubit,
                amp_vec,
                experiment_params["shots"],
                qubit_dict["calib_params"],
                num_pi_pulses=experiment_params["number_pi_pulses"],
                pulse_type=experiment_params["pulse_type"],
                meta_dict=MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config_copy,
            )
            print(f"Rabi_Linecut_SS for {qubit} complete")

        if experiment == "Frequency_Calibration" and experiment_params["run"]:
            freq_vec = np.linspace(
                -experiment_params["freq_span"] / 2,
                experiment_params["freq_span"] / 2,
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = FreqCal1Q(
                qubit,
                freq_vec,
                experiment_params["shots"],
                qubit_dict["calib_params"],
                experiment_params["number_pi_pulses"],
                pulse_type=experiment_params["pulse_type"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Frequency_Calibration for {qubit} complete")

        if experiment == "DRAG_Calibation" and experiment_params["run"]:
            MD = copy.deepcopy(meta_dict)
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
                meta_dict=MD,
                config=config_copy,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
            )
            print(f"DRAG_Calibation for {qubit} complete")
