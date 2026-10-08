import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import copy

matplotlib.use("Agg")
from LabCode.Config.make_meta_dict import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q

# from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level_sweep import (
    SingleShot1Q3LevelSweep,
)
from LabCode.Experiments.Temperature.m_equilibrium_population import (
    EquilibriumPopulation,
)
from LabCode.Experiments.Temperature.m_temperature_RPM import TemperatureRPM
from LabCode.Experiments.Rabi.m_Rabi_Chevron_ef_IQ import RabiChevronFStateIQ
from LabCode.Experiments.Gates.m_ramsey_anharmonicity import RamseyAnharmonicity
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

"""
This Provides all the experiments needed to calibrate single qubit pi and pi/2 pulses.
"""
experiment_dict = {
    "qubits": {
        "q1": {
            "experiments": {
                # ToDo: Fix for redefinition of ef pulse
                "Rabi_f_State_Chevron_IQ": {
                    "run": False,
                    "shots": 41,
                    "number_pi_pulses": 1,
                    "a_min": 0.0,  # Volts
                    "a_max": 0.4,  # Volts note: 2 * pulse amplitude
                    "a_points": 21,
                    "anharm_min": -110 * u.MHz,
                    "anharm_max": -106 * u.MHz,
                    "freq_points": 11,
                    "object": RabiChevronFStateIQ,
                },
                "Readout_Cal_1Q3L": {
                    "run": False,
                    "shots": 100,
                    "f_span": 0.4 * u.MHz,
                    "num_f": 6,
                    "a_min": 0.005,  # V
                    "a_max": 0.009,  # V. Note: must be < 2 * read_amp
                    "num_a": 6,
                    "object": SingleShot1Q3Level,
                },
                "Readout_Cal_1Q3L_Sweep": {
                    "run": False,
                    "shots": 100,
                    "f_span": 0.30 * u.MHz,
                    "num_f": 7,
                    "a_min": 0.008,  # V
                    "a_max": 0.05,  # V. Note: must be < 2 * read_amp
                    "num_a": 6,
                    "object": SingleShot1Q3LevelSweep,
                },
                "SS_Cal_3L": {
                    "run": True,  # Must run to preform SS experiments
                    "shots": 1000,
                    "object": SingleShot1Q3Level,
                },
                "SS_Cal": {
                    "run": False,  # Must run to preform SS experiments
                    "shots": 1000,
                    "number_pi_pulses": 1,
                    "ground_threshold": 0.7,
                    "object": SingleShot1Q,
                },
                "Ramsey_Anharmonicity": {
                    "run": False,
                    "shots": 100,
                    "detuning": 2 * u.MHz,
                    "t_min": 16 * u.ns,
                    "t_max": 5 * u.us,
                    "t_step": 32 * u.ns,
                    "object": RamseyAnharmonicity,
                },
                "Equilibrium_Population": {
                    "run": False,  # Must run to preform SS experiments
                    "shots": 100000,
                    "number_pi_pulses": 1,
                    "ground_threshold": 0.7,
                    "object": EquilibriumPopulation,
                },
                "RPM": {
                    "run": False,  # RPM Designed with just 2 state discrimination.
                    "shots": 3001,
                    "max_amp_ef": 0.45,
                    "a_points": 101,
                    "object": TemperatureRPM,
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
        if experiment == "Rabi_f_State_Chevron_IQ" and experiment_params["run"]:
            MD = copy.deepcopy(meta_dict)
            MD["ef_amp"] = experiment_params["a_max"]
            config_copy = meta_dict_to_config_1Q(MD)
            amp_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["a_points"],
            )
            freq_vec = np.linspace(
                experiment_params["anharm_min"],
                experiment_params["anharm_max"],
                experiment_params["freq_points"],
                dtype=int,
            )
            exp = RabiChevronFStateIQ(
                qubit,
                amp_vec,
                freq_vec,
                experiment_params["shots"],
                experiment_params["number_pi_pulses"],
                meta_dict=MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config_copy,
            )
            print(f"Rabi Chevron IQ for {qubit} complete")
        if experiment == "SS_Cal_3L" and experiment_params["run"]:
            while True:
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
                print(SS.calib_params)
                fidelity = SS.max_F
                config = meta_dict_to_config_1Q(meta_dict)
                print(f"SS Cal for {qubit} complete")
        if experiment == "Readout_Cal_1Q3L" and experiment_params["run"]:
            r_IF = meta_dict["r_IF"]
            amp_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["num_a"],
            )
            f_vec = np.linspace(
                -experiment_params["f_span"] / 2,
                experiment_params["f_span"] / 2,
                experiment_params["num_f"],
            )
            print(
                f"Readout_Cal_1Q est. completion time: {len(amp_vec) * len(f_vec) * experiment_params['shots'] * meta_dict['global_params']['reset_time'] / u.s / 60} Minutes"
            )

            fidels = np.zeros([len(f_vec), len(amp_vec)])
            for j, amp in enumerate(amp_vec):
                meta_dict["read_amp"] = amp
                for i, df in enumerate(f_vec):
                    meta_dict["r_IF"] = r_IF + df
                    config = meta_dict_to_config_1Q(meta_dict)
                    SS = SingleShot1Q3Level(
                        qubit,
                        experiment_params["shots"],
                        meta_dict=meta_dict,
                        config=config,
                        outer_folder=experiment_dict["outer_folder"],
                        path=qubit,
                        suffix=experiment,
                        plot=False,
                        save_data=False,
                    )
                    if i == 0:
                        if j == 0:
                            name = SS.iname[:-6] + "2Q"

                    fidels[i][j] = SS.max_F
                    if SS.max_F > 0.85:
                        print(SS.max_F)
                        print(amp)
                        print((df + r_IF) / u.MHz)

                    vmin = np.max([np.min([fidels]), 0.5])
                    vmax = np.max([fidels])

                    f_idx_max, a_idx_max = np.unravel_index(
                        np.argmax(fidels), fidels.shape
                    )

                    fig, ax = plt.subplots(1, 1, constrained_layout=True)
                    pcm1 = ax.pcolormesh(
                        amp_vec, (f_vec + r_IF) / u.MHz, fidels, vmin=vmin, vmax=vmax
                    )
                    ax.set_xlabel("Readout amplitude (V)")
                    ax.set_ylabel("Resonator IF (MHz)")
                    ax.set_title(
                        f"Readout Cal {qubit}, optimal freq: {(f_vec[f_idx_max] + r_IF) / u.MHz} MHz. amp optimal: {amp_vec[a_idx_max]} "
                    )
                    fig.colorbar(pcm1, ax=ax)
                    plt.savefig(name + "2QC", bbox_inches="tight")
                    plt.close()

            print(f"Optimal Freq: {(f_vec[f_idx_max] + r_IF) / u.MHz} MHz")
            print(f"Optimal amp: {amp_vec[a_idx_max]}")

        if experiment == "Readout_Cal_1Q3L_Sweep" and experiment_params["run"]:
            a_vec = np.linspace(
                experiment_params["a_min"],
                experiment_params["a_max"],
                experiment_params["num_a"],
            )
            f_vec = np.linspace(
                -experiment_params["f_span"] / 2,
                experiment_params["f_span"] / 2,
                experiment_params["num_f"],
                dtype=int,
            )
            exp = SingleShot1Q3LevelSweep(
                qubit,
                a_vec,
                f_vec,
                experiment_params["shots"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            print(f"Readout_Cal_1Q3L_Sweep for {qubit} complete. max_F={exp.max_F:.4f}")

        if experiment == "SS_Cal" and experiment_params["run"]:
            # fidelity = 0
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
            print(f"SS Cal for {qubit} complete")

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

        if experiment == "Equilibrium_Population" and experiment_params["run"]:
            # fidelity = 0
            SS = EquilibriumPopulation(
                qubit,
                experiment_params["shots"],
                meta_dict=meta_dict,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=config,
            )
            config = meta_dict_to_config_1Q(meta_dict)
            print(f"Equilibrium_Population for {qubit} complete")
        if experiment == "RPM" and experiment_params["run"]:
            amp_vec = np.linspace(0, 1, experiment_params["a_points"])
            MD = copy.deepcopy(meta_dict)
            MD["ef_amp"] = experiment_params["max_amp_ef"]
            new_config = meta_dict_to_config_1Q(MD)
            RPM = TemperatureRPM(
                qubit,
                amp_vec,
                experiment_params["shots"],
                meta_dict=MD,
                outer_folder=experiment_dict["outer_folder"],
                path=qubit,
                suffix=experiment,
                config=new_config,
            )
