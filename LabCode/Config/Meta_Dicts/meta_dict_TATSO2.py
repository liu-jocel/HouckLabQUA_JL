import sys
import numpy as np

sys.path.append("C:/Users/my/Desktop/Houck-Lab-Qua")
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

outer_folder = r"Z:\t1Team\Data\2026-04-27_BFG_cooldown\TATCL02\OPX"

reset_time = 3000 * u.us  ### Test if you need explicit reset tics
resonator_dump_time = 50 * u.us
global_params = {
    "read_len": 17000,  # ns
    "reset_time": reset_time,  # s
    "outer_folder": outer_folder,
}

q1 = {
    # Global parameters #Q3
    "global_params": global_params,
    "element_name": "q1",
    "q_name": "q1",
    "r_name": "r1",
    "q_I_channel": 1,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-200 + 1.13) * u.MHz,  # -2.635
    "detuning": 0.0 * u.MHz,
    "anharmonicity": -98 * u.MHz,
    "ef_amp": 0.06,
    "DRAG_beta": 0,
    "X180": {"amp": 0.10103606911936054, "phase": 0, "length": 4 * 20},
    "Y180": {"amp": 0.10103606911936054, "phase": 0.0, "length": 4 * 20},
    "X90": {
        "amp": 0.10103606911936054,
        "phase": 0,
        "length": 4 * 10,
    },
    "Y90": {
        "amp": 0.10103606911936054,
        "phase": 0.0,
        "length": 4 * 10,
    },
    "read_amp": 0.045,  # V
    "read_len": 20 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.45,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q1",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 3.838 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq1",
        "mixer_off_I": -0.008825645878732753,
        "mixer_off_Q": -0.024779668927476454,
        "mixer_gain": 0,  # -0.038995361328125,
        "mixer_phase": 0,  # -0.03492431640625,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r1",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.816 * u.GHz,
    },
    "r_IF": -200.8 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr1",
        "mixer_off_I": -0.005782739073038102,
        "mixer_off_Q": -0.003554366901516914,
        "mixer_gain": -0.013535118103027342,
        "mixer_phase": -0.07991676330566408,
    },
}

q2 = {
    "global_params": global_params,
    "element_name": "q2",
    "q_name": "q2",
    "r_name": "r2",
    "q_I_channel": 1,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-200 + 1.5) * u.MHz,  # -2.635
    "detuning": 0.42 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.2101226847789492,
    "DRAG_beta": 0,
    "X180": {"amp": 0.17, "phase": 0.0, "length": 4 * 120},
    "Y180": {"amp": 0.17, "phase": 0.0, "length": 4 * 120},
    "X90": {"amp": 0.17, "phase": 0.0, "length": 4 * 80},
    "Y90": {
        "amp": 0.17,  # 2101226847789492
        "phase": 0.0,
        "length": 4 * 80,
    },
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q2",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 4.357 * u.GHz,
    },
    "cw_amp": 0.4,
    "cw_len": 500 * u.ns,
    "read_amp": 0.1,  #
    "read_len": 8 * u.us,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "read_theta": 0,
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq2",
        "mixer_off_I": -0.008463113847392381,
        "mixer_off_Q": -0.02458746743941447,
        "mixer_gain": -0.038995361328125,
        "mixer_phase": -0.03492431640625,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r2",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.905 * u.GHz,
    },
    "r_IF": -200.3 * u.MHz,  # 200.25 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr2",
        "mixer_off_I": -0.005782739073038102,
        "mixer_off_Q": -0.003554366901516914,
        "mixer_gain": -0.013535118103027342,
        "mixer_phase": -0.07991676330566408,
    },
}

q3 = {
    # Global parameters #Q3
    "global_params": global_params,
    "element_name": "q3",
    "q_name": "q3",
    "r_name": "r3",
    "q_I_channel": 1,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-200 + 0.7) * u.MHz,  # -2.635
    "detuning": 0.0 * u.MHz,
    "anharmonicity": -98 * u.MHz,
    "ef_amp": 0.06,
    "DRAG_beta": 0,
    "X180": {"amp": 0.21, "phase": 0, "length": 4 * 160},
    "Y180": {"amp": 0.21, "phase": 0.0, "length": 4 * 160},
    "X90": {
        "amp": 0.21,
        "phase": 0,
        "length": 4 * 80,
    },
    "Y90": {
        "amp": 0.21,
        "phase": 0.0,
        "length": 4 * 80,
    },
    "read_amp": 0.1,  # V
    "read_len": 7 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.45,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q3",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 4.564 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq3",
        "mixer_off_I": -0.008825645878732753,
        "mixer_off_Q": -0.024779668927476454,
        "mixer_gain": -0.038995361328125,
        "mixer_phase": -0.03492431640625,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r3",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.014 * u.GHz,
    },
    "r_IF": -199.8 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr3",
        "mixer_off_I": -0.005782739073038102,
        "mixer_off_Q": -0.003554366901516914,
        "mixer_gain": -0.013535118103027342,
        "mixer_phase": -0.07991676330566408,
    },
}

q4 = {
    # Global parameters #q5
    "global_params": global_params,
    "element_name": "q4",
    "q_name": "q4",
    "r_name": "r4",
    "q_I_channel": 2,
    "q_Q_channel": 1,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-200) * u.MHz,  # -2.635
    "detuning": 0.12 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "DRAG_beta": 0,
    "X180": {"amp": 0.25, "detuning": 0 * u.MHz, "phase": 0.0, "length": 4 * 160},
    "Y180": {"amp": 0.25, "phase": 0.0, "length": 4 * 160},
    "X90": {"amp": 0.25, "detuning": 0.0 * u.MHz, "phase": 0.0, "length": 4 * 80},
    "Y90": {
        "amp": 0.25,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 80,
    },
    "read_amp": 0.1,  # 08, # V
    "read_len": 10 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    "q_LO": {
        "LO_name": "q4",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 4.801 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq4",
        "mixer_off_I": 0,  # -0.008602704109580368,
        "mixer_off_Q": 0,  # -0.025946928709890925,
        "mixer_gain": 0,  # -0.038995361328125,
        "mixer_phase": 0,  # -0.03492431640625,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r4",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.101 * u.GHz,
    },
    "r_IF": -200.08 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr4",
        "mixer_off_I": -0.005782739073038102,
        "mixer_off_Q": -0.003554366901516914,
        "mixer_gain": -0.013535118103027342,
        "mixer_phase": -0.07991676330566408,
    },
}

q5 = {
    # Global parameters #q5
    "global_params": global_params,
    "element_name": "q5",
    "q_name": "q5",
    "r_name": "r5",
    "q_I_channel": 1,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-200 + 2) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "DRAG_beta": 0,
    "X180": {"amp": 0.03, "detuning": 0 * u.MHz, "phase": 0, "length": 4 * 80},
    "Y180": {"amp": 0.03, "detuning": 0.0 * u.MHz, "phase": 0.0, "length": 4 * 80},
    "X90": {"amp": 0.03, "detuning": 0.0 * u.MHz, "phase": 0.0, "length": 4 * 40},
    "Y90": {
        "amp": 0.03,
        "detuning": 0.0 * u.MHz,
        "phase": 0,
        "length": 4 * 40,
    },
    "read_amp": 0.1,  # 08, # V
    "read_len": 8 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q5",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 5.638 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq5",
        "mixer_off_I": -0.008602704109580368,
        "mixer_off_Q": -0.025946928709890925,
        "mixer_gain": -0.038995361328125,
        "mixer_phase": -0.03492431640625,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r5",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.218 * u.GHz,
    },
    "r_IF": -199.5 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr5",
        "mixer_off_I": -0.005782739073038102,
        "mixer_off_Q": -0.003554366901516914,
        "mixer_gain": -0.013535118103027342,
        "mixer_phase": -0.07991676330566408,
    },
}

q6 = {
    # Global parameters #q6
    "global_params": global_params,
    "element_name": "q6",
    "q_name": "q6",
    "r_name": "r6",
    "q_I_channel": 1,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": -200 * u.MHz,
    "detuning": 0 * u.MHz,
    "anharmonicity": -144.81186474028618 * u.MHz,
    "ef_amp": 0.1350980772798654,
    "DRAG_beta": 0.00,
    "X180": {"amp": 0.023690251167764004, "phase": 0, "length": 4 * 80},
    "Y180": {"amp": 0.023690251167764004, "phase": 0, "length": 4 * 80},
    "X90": {"amp": 0.023690251167764004, "phase": 0, "length": 4 * 40},
    "Y90": {
        "amp": 0.023690251167764004,
        "phase": 0,
        "length": 4 * 40,
    },
    "read_amp": 0.1,  # V
    "read_len": 12 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q6",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 5.938 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq6",
        "mixer_off_I": -0.01,
        "mixer_off_Q": -0.022,
        "mixer_gain": 0.00,
        "mixer_phase": -0.04,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r6",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.296 * u.GHz,
    },
    "r_IF": -200 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr6",
        "mixer_off_I": -0.008646490414685098,
        "mixer_off_Q": -0.02805557856627274,
        "mixer_gain": 0.02705135345458984,
        "mixer_phase": -0.09634590148925784,
    },
}


def make_meta_dict(qubit):
    if qubit == "q1":
        return {"global_params": global_params, "q1": q1, "outer_folder": outer_folder}
    if qubit == "q2":
        return {"global_params": global_params, "q2": q2, "outer_folder": outer_folder}
    if qubit == "q3":
        return {"global_params": global_params, "q3": q3, "outer_folder": outer_folder}
    if qubit == "q4":
        return {"global_params": global_params, "q4": q4, "outer_folder": outer_folder}
    if qubit == "q5":
        return {"global_params": global_params, "q5": q5, "outer_folder": outer_folder}
    if qubit == "q6":
        return {"global_params": global_params, "q6": q6, "outer_folder": outer_folder}


def make_meta_dict_6Q():
    return {
        qubit: make_meta_dict(qubit) for qubit in ["q1", "q2", "q3", "q4", "q5", "q6"]
    }
