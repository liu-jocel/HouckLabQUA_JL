import pickle
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import atexit
import matplotlib.pyplot as plt
import tkinter as tk

plt.ion()


def cleanup():
    try:
        plt.close("all")
        if tk._default_root:
            tk._default_root.quit()
            tk._default_root.destroy()
    except Exception:
        pass


atexit.register(cleanup)
# CHANGE THIS DEPENDING ON WHICH META DICT TO MAKE
from LabCode.Config.make_meta_dict import make_meta_dict

from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2R import T2R
from LabCode.Experiments.Coherence.m_T2R_purity import T2R_Purity
from LabCode.Experiments.Coherence.m_T2E import T2E
from LabCode.Experiments.Coherence.m_T2CPMG import T2CPMG
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Rabi.m_Rabi_linecut_timed_SS import RabiLinecutTimedSS
from LabCode.Experiments.Coherence.m_T1p_spin_locking import T1pSpinLocking
from LabCode.Experiments.Coherence.m_T2R_quad_repeated import T2RQuadRepeated
from LabCode.Experiments.Coherence.m_T2R_repeated import T2RRepeated
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

"""
Single qubit coherence measurements.
"""

experiment_dict = {
    "infinite_loop": True,
    "qubits": {
        "q5": {
            "experiments": {
                "SS_Cal": {
                    "run": True,  # Must run to preform coherence experiments
                    "shots": 400,
                    "object": SingleShot1Q,
                },
                "T1": {
                    "run": True,
                    "shots": 400,
                    "num_pulses": [1],
                    "t_min": 100 * u.ns,  # 1 * u.us,
                    "t_max": 10000 * u.us,
                    "t_points": 71,
                    "object": T1,
                },
                "T2R": {
                    "run": False,
                    "shots": 150,
                    "detuning": -0.05 * u.MHz,
                    "t_min": 16 * u.ns,
                    "t_max": 800 * u.us,
                    "t_step": 2 * u.us,
                    "object": T2R,
                },
                "T2E": {
                    "run": True,
                    "shots": 400,
                    "t_min": 1 * u.us,
                    "t_max": 10000 * u.us,
                    "t_points": 51,
                    "object": T2E,
                },
                "T2CPMG": {
                    "run": False,
                    "shots": 1000,
                    "t_max": [10000 * u.us],  # [5000 * u.us] * 38,
                    "t_points": 201,
                    "pi_pulses": [1],
                    """
                        [1, 2, 3, 4, 5,
                                  11, 21, 31, 41,61,
                                  81, 101, 201, 301, 401,
                                  501, 701, 1001, 1201, 1501,
                                  1701, 2001, 2501, 3001, 4001],
                        """
                    "object": T2CPMG,
                },
                "T1p_SL": {  # ensure qubit freq is optimized before running this experiment
                    "run": False,
                    "shots": 400,
                    # set qubit cw_amp = 0.25, then sl_amp, the amplitude scaling factor for the spin-locking pulse, can range from -2 to +2
                    "sl_amps": [
                        0.01095
                    ],  # np.random.permutation(np.arange(0.0052, 0.0152, 0.00025)),
                    "t_min": 100 * u.ns,
                    "t_max": 10000 * u.us,
                    "t_points": 51,
                    "time_rabi_params": [  # start, stop, step, shots
                        16 * u.ns,
                        4000 * u.ns,
                        80 * u.ns,
                        200,
                    ],  # t_min, t_max, t_points, shots for time rabi experiment prior to T1p
                    "t1_params": [
                        100 * u.ns,
                        10000 * u.us,
                        51,
                        400,
                    ],  # t_min, t_max, t_points, shots for T1 experiment prior to T1p
                    "object": T1pSpinLocking,
                },
                "T2R_Repeated": {
                    "run": False,
                    "shots": 200000,
                    "tau_0": 50 * u.us,
                    "object": T2RRepeated,
                },
                "T2R_Quad": {
                    "run": False,
                    "shots": 5000000,
                    "tau_0": 40 * u.us,
                    "object": T2RQuadRepeated,
                },
            },
        },
        #     "q2": {
        #         "experiments": {
        #             "SS_Cal": {
        #                 "run": False,  # Must run to preform coherence experiments
        #                 "shots": 1000,
        #                 "object": SingleShot1Q,
        #             },
        #             "T1": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "num_pulses": [1],
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T1,
        #             },
        #             "T2R": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "detuning": 0.3 * u.MHz,
        #                 "t_min": 16 * u.ns,
        #                 "t_max": 400 * u.us,
        #                 "t_step": 1000 * u.ns,
        #                 "object": T2R,
        #             },
        #             "T2E": {
        #                 "run": False,
        #                 "shots": 1000,
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T2E,
        #             },
        #             "T2CPMG": {
        #                 "run": False,
        #                 "shots": 2000,
        #                 "t_max": [1000 * u.us],
        #                 "t_points": 101,
        #                 "pi_pulses": [1701],  #
        #                 "object": T2CPMG,
        #             },
        #         },
        #     },
        #     "q3": {
        #         "experiments": {
        #             "SS_Cal": {
        #                 "run": False,  # Must run to preform coherence experiments
        #                 "shots": 1000,
        #                 "object": SingleShot1Q,
        #             },
        #             "T1": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "num_pulses": [1],
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T1,
        #             },
        #             "T2R": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "detuning": 0.2 * u.MHz,
        #                 "t_min": 16 * u.ns,
        #                 "t_max": 100 * u.us,
        #                 "t_step": 500 * u.ns,
        #                 "object": T2R,
        #             },
        #             "T2E": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T2E,
        #             },
        #             "T2CPMG": {
        #                 "run": False,
        #                 "shots": 500,
        #                 "t_points": 101,
        #                 "t_max": [2000 * u.us] * 5 + [2000 * u.us] * 3 + [2000 * u.us],
        #                 "pi_pulses": [3, 7, 21, 61, 241, 501, 1001, 2501, 4001],
        #                 "object": T2CPMG,
        #             },
        #         },
        #     },
        #     "q4": {
        #         "experiments": {
        #             "SS_Cal": {
        #                 "run": False,  # Must run to preform coherence experiments
        #                 "shots": 1000,
        #                 "object": SingleShot1Q,
        #             },
        #             "T1": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "num_pulses": [1],
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T1,
        #             },
        #             "T2R": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "detuning": 0.2 * u.MHz,
        #                 "t_min": 16 * u.ns,
        #                 "t_max": 100 * u.us,
        #                 "t_step": 500 * u.ns,
        #                 "object": T2R,
        #             },
        #             "T2E": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T2E,
        #             },
        #             "T2CPMG": {
        #                 "run": False,
        #                 "shots": 5000,
        #                 "t_max": [
        #                     1500 * u.us,
        #                 ]
        #                 * 14,
        #                 "t_points": 101,
        #                 "pi_pulses": [
        #                     3711,
        #                     3811,
        #                     3901,
        #                     4001,
        #                     4101,
        #                     4201,
        #                     4301,
        #                     4401,
        #                     4501,
        #                     4601,
        #                     4701,
        #                     4801,
        #                     4901,
        #                     5001,
        #                 ],  #
        #                 "object": T2CPMG,
        #             },
        #         },
        #     },
        #     "q5": {
        #         "experiments": {
        #             "SS_Cal": {
        #                 "run": False,  # Must run to preform coherence experiments
        #                 "shots": 1000,
        #                 "object": SingleShot1Q,
        #             },
        #             "T1": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "num_pulses": [1],
        #                 "t_min": 1 * u.us,
        #                 "t_max": 3000 * u.us,
        #                 "t_points": 71,
        #                 "object": T1,
        #             },
        #             "T2R": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "detuning": 0 * u.MHz,
        #                 "t_min": 16 * u.ns,
        #                 "t_max": 300 * u.us,
        #                 "t_step": 1000 * u.ns,
        #                 "object": T2R,
        #             },
        #             "T2E": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "t_min": 0.5 * u.us,
        #                 "t_max": 5000 * u.us,
        #                 "t_points": 71,
        #                 "object": T2E,
        #             },
        #             "T2CPMG": {
        #                 "run": False,
        #                 "shots": 1000,
        #                 "t_max": [500 * u.us] * 21,
        #                 "t_points": 101,
        #                 "pi_pulses": [
        #                     5,
        #                     7,
        #                     15,
        #                     31,
        #                     41,
        #                     61,
        #                     71,
        #                     81,
        #                     91,
        #                     125,
        #                     151,
        #                     175,
        #                     225,
        #                     251,
        #                     275,
        #                     325,
        #                     351,
        #                     371,
        #                     425,
        #                     451,
        #                     471,
        #                 ],  #
        #                 "object": T2CPMG,
        #             },
        #         },
        #     },
        #     "q6": {
        #         "experiments": {
        #             "SS_Cal": {
        #                 "run": False,  # Must run to preform coherence experiments
        #                 "shots": 1000,
        #                 "object": SingleShot1Q,
        #             },
        #             "T1": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "num_pulses": [1],
        #                 "t_min": 1 * u.us,
        #                 "t_max": 2000 * u.us,
        #                 "t_points": 101,
        #                 "object": T1,
        #             },
        #             "T2R": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "detuning": 0.3 * u.MHz,
        #                 "t_min": 16 * u.ns,
        #                 "t_max": 50 * u.us,
        #                 "t_step": 100 * u.ns,
        #                 "object": T2R,
        #             },
        #             "T2E": {
        #                 "run": False,
        #                 "shots": 400,
        #                 "t_min": 0.5 * u.us,
        #                 "t_max": 5000 * u.us,
        #                 "t_points": 71,
        #                 "object": T2E,
        #             },
        #             "T2CPMG": {
        #                 "run": False,
        #                 "shots": 1000,
        #                 "t_max": [500 * u.us] * 21,
        #                 "t_points": 101,
        #                 "pi_pulses": [
        #                     5,
        #                     7,
        #                     15,
        #                     31,
        #                     41,
        #                     61,
        #                     71,
        #                     81,
        #                     91,
        #                     125,
        #                     151,
        #                     175,
        #                     225,
        #                     251,
        #                     275,
        #                     325,
        #                     351,
        #                     371,
        #                     425,
        #                     451,
        #                     471,
        #                 ],  #
        #                 "object": T2CPMG,
        #             },
        #         },
        #     },
    },
}
# [1, 3, 5, 11, 31, 51, 71, 101, 201, 301,501, 701, 1001, 1501, 2001, 3001,]
print("Running the Following Experiments:")
if experiment_dict["infinite_loop"]:
    print("INFINITE LOOP")
    print("####################################################")

for qubit, qubit_dict in experiment_dict["qubits"].items():
    for experiment, experiment_params in qubit_dict["experiments"].items():
        if experiment_params["run"]:
            print(f"{qubit} {experiment}")
print("####################################################")

while True:
    # Run Experiments
    for qubit, qubit_dict in experiment_dict["qubits"].items():
        meta_dict = make_meta_dict(qubit)[qubit]
        experiment_dict["outer_folder"] = meta_dict["global_params"]["outer_folder"]
        config = meta_dict_to_config_1Q(meta_dict)
        LOs = [meta_dict["q_LO"], meta_dict["r_LO"]]
        print(f"{qubit} LO: {LOs}")
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
                    config=config,
                )

                if SS.max_F < 0.7:
                    print(f"SS Cal for {qubit} failed, passing to next")
                    break
                else:
                    qubit_dict["calib_params"] = SS.calib_params
                    meta_dict["read_theta"] = SS.calib_params["read_theta"]
                    config = meta_dict_to_config_1Q(meta_dict)
                    print(f"SS Cal for {qubit} complete")

            if experiment == "T1" and experiment_params["run"]:
                t_vec = np.logspace(
                    np.log10(experiment_params["t_min"]),
                    np.log10(experiment_params["t_max"]),
                    experiment_params["t_points"],
                )
                if experiment_params["num_pulses"]:
                    for num_pusle in experiment_params["num_pulses"]:
                        exp = T1(
                            qubit,
                            t_vec,
                            experiment_params["shots"],
                            calib_params=qubit_dict["calib_params"],
                            meta_dict=meta_dict,
                            outer_folder=experiment_dict["outer_folder"],
                            path=qubit,
                            suffix=experiment,
                            config=config,
                            num_pulses=num_pusle,
                        )
                else:
                    exp = T1(
                        qubit,
                        t_vec,
                        experiment_params["shots"],
                        calib_params=qubit_dict["calib_params"],
                        meta_dict=meta_dict,
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment,
                        config=config,
                    )
                print(f"T1 {qubit} complete")

            if experiment == "T2R" and experiment_params["run"]:
                t_vec = np.arange(
                    experiment_params["t_min"],
                    experiment_params["t_max"],
                    experiment_params["t_step"],
                )
                exp = T2R(
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

            if experiment == "T2E" and experiment_params["run"]:
                t_vec = np.logspace(
                    np.log10(experiment_params["t_min"]),
                    np.log10(experiment_params["t_max"]),
                    experiment_params["t_points"],
                )
                exp = T2E(
                    qubit,
                    t_vec,
                    experiment_params["shots"],
                    calib_params=qubit_dict["calib_params"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config,
                )
                print(f"T2E {qubit} complete")

            if experiment == "T2CPMG" and experiment_params["run"]:
                assert len(experiment_params["pi_pulses"]) == len(
                    experiment_params["t_max"]
                )
                data = []
                error = []
                pulses = []
                for i, pi_pulses in enumerate(experiment_params["pi_pulses"]):
                    t_min = (16 + pi_pulses * 20) * u.ns
                    print(pi_pulses)
                    t_max = experiment_params["t_max"][i]
                    t_vec = np.linspace(t_min, t_max, experiment_params["t_points"])
                    exp = T2CPMG(
                        qubit,
                        t_vec,
                        pi_pulses,
                        experiment_params["shots"],
                        calib_params=qubit_dict["calib_params"],
                        meta_dict=meta_dict,
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment,
                        config=config,
                    )
                    try:
                        T2, err = exp.data["T2CPMG"]
                    except:
                        T2 = 0
                        err = 0
                    data.append(T2)
                    error.append(err)
                    pulses.append(pi_pulses)
                    output = {"T2": data, "error": error, "pulses": pulses}
                    pickle_name = exp.pname[:-12] + "_CPMG_TOTAL.pkl"
                    with open(pickle_name, "wb") as f:
                        pickle.dump(output, f)

                    plt.errorbar(pulses, data, yerr=error)
                    plt.title(f"{qubit} T2CPMG Convergence")
                    plt.xlabel("Pi Pulses")
                    plt.ylabel("T2 (us)")
                    image_name = exp.iname[:-12] + "_CPMG_TOTAL.png"
                    plt.savefig(image_name, bbox_inches="tight")
                    plt.close()
                print(f"T2CPMG {qubit} complete")

            if experiment == "T1p_SL" and experiment_params["run"]:
                t1p_t_vec = np.logspace(
                    np.log10(experiment_params["t_min"]),
                    np.log10(experiment_params["t_max"]),
                    experiment_params["t_points"],
                )

                for sl_amp in experiment_params["sl_amps"]:
                    # run time rabi
                    time_rabi_t_vec = np.arange(
                        experiment_params["time_rabi_params"][0],
                        experiment_params["time_rabi_params"][1],
                        experiment_params["time_rabi_params"][2],
                    ).astype(int)
                    exp1 = RabiLinecutTimedSS(
                        qubit,
                        time_rabi_t_vec,
                        experiment_params["time_rabi_params"][3],
                        ampx=sl_amp,
                        pulse_name="constant_tone",
                        calib_params=qubit_dict["calib_params"],
                        meta_dict=meta_dict,
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment + f"_time_rabi_sl_amp_{sl_amp:.5f}",
                        config=config,
                    )
                    print(f"Time Rabi for {qubit} sl_amp {sl_amp:.5f} complete")

                    # run T1
                    t1_t_vec = np.logspace(
                        np.log10(experiment_params["t1_params"][0]),
                        np.log10(experiment_params["t1_params"][1]),
                        experiment_params["t1_params"][2],
                    )
                    exp2 = T1(
                        qubit,
                        t1_t_vec,
                        experiment_params["t1_params"][3],
                        calib_params=qubit_dict["calib_params"],
                        meta_dict=meta_dict,
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment + f"_T1_sl_amp_{sl_amp:.5f}",
                        config=config,
                    )
                    print(f"T1 for {qubit} sl_amp {sl_amp:.5f} complete")

                    # run T1p spin locking
                    exp = T1pSpinLocking(
                        qubit,
                        t1p_t_vec,
                        sl_amp,
                        experiment_params["shots"],
                        calib_params=qubit_dict["calib_params"],
                        meta_dict=meta_dict,
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment + f"_sl_amp_{sl_amp:.5f}",
                        config=config,
                    )
                    print(f"T1p Spin Locking {qubit} sl_amp {sl_amp:.2f} complete")
            if experiment == "T2R_Repeated" and experiment_params["run"]:
                exp = T2RRepeated(
                    qubit,
                    experiment_params["tau_0"],
                    experiment_params["shots"],
                    calib_params=qubit_dict["calib_params"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config,
                )
                print(f"T2R Repeated {qubit} complete")

            if experiment == "T2R_Quad" and experiment_params["run"]:
                exp = T2RQuadRepeated(
                    qubit,
                    experiment_params["tau_0"],
                    experiment_params["shots"],
                    calib_params=qubit_dict["calib_params"],
                    meta_dict=meta_dict,
                    outer_folder=experiment_dict["outer_folder"],
                    path=qubit,
                    suffix=experiment,
                    config=config,
                )
                print(f"T2R Quadrature {qubit} complete")

    if not experiment_dict["infinite_loop"]:
        break
