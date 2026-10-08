import sys
import numpy as np

sys.path.append("C:/Users/my/Desktop/Houck-Lab-Qua")
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

# outer_folder = r"Z:\t1Team\Data\2026-04-27_BFG_cooldown\TATCL02\OPX\rb_fidelity_sweep"
outer_folder = r"Z:\t1Team\Data\2026-04-27_BFG_cooldown\TATCL02\OPX"

reset_time = 3000 * u.us  ### Test if you need explicit reset tics
resonator_dump_time = 50 * u.us
global_params = {
    "read_len": 17000,  # ns
    "reset_time": reset_time,  # s
    "outer_folder": outer_folder,
}

q1 = {
    # Global parameters #Q1
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
    "IF_freq": (-152.55557039849612 + 0.33333) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -130 * u.MHz,
    "ef_amp": 0.2,
    "DRAG_beta": 0,
    "X180": {"amp": 0.10711763479665834, "phase": 0, "length": 4 * 8},
    "Y180": {"amp": 0.10711763479665834, "phase": 0.0, "length": 4 * 8},
    "X90": {
        "amp": 0.10711763479665834,
        "phase": 0,
        "length": 4 * 4,
    },
    "Y90": {
        "amp": 0.10711763479665834,
        "phase": 0.0,
        "length": 4 * 4,
    },
    "read_amp": 0.025,  # V
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
        "LO_freq": 2.315 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq1",
        "mixer_off_I": -0.00853381028864533,
        "mixer_off_Q": -0.024793138401582832,
        "mixer_gain": 0.024737078907128313,
        "mixer_phase": -0.04573686451203686,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r1",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.905 * u.GHz,
    },
    "r_IF": -203.24 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr1",
        "mixer_off_I": -0.00442640408873558,
        "mixer_off_Q": 0.0008535817265510563,
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
    "IF_freq": (-199) * u.MHz,  # -2.635
    "detuning": 0.42 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.2101226847789492,
    "DRAG_beta": 0,
    "X180": {"amp": 0.07718419726119313, "phase": 0.0, "length": 4 * 50},
    "Y180": {"amp": 0.07718419726119313, "phase": 0.0, "length": 4 * 50},
    "X90": {"amp": 0.07718419726119313, "phase": 0.0, "length": 4 * 25},
    "Y90": {
        "amp": 0.07718419726119313,  # 2101226847789492
        "phase": 0.0,
        "length": 4 * 25,
    },
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q2",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 2.32 * u.GHz,
    },
    "cw_amp": 0.00,
    "cw_len": 500 * u.ns,
    "read_amp": 0.09,  # 19,  #
    "read_len": 20 * u.us,
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
        "LO_freq": 6.989 * u.GHz,
    },
    "r_IF": -199.925 * u.MHz,  # 200.25 * u.MHz,  # Hz
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
    # Global parameters #q5
    "global_params": global_params,
    "element_name": "q3",
    "q_name": "q3",
    "r_name": "r3",
    "q_I_channel": 5,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-197.06749773526013) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.15426715109427228,
        "detuning": 0 * u.MHz,
        "phase": 0,
        "length": 4 * 200,
    },
    "Y180": {
        "amp": 0.15426715109427228,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 200,
    },
    "X90": {
        "amp": 0.15426715109427228,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 100,
    },
    "Y90": {
        "amp": 0.15426715109427228,
        "detuning": 0.0 * u.MHz,
        "phase": -0.01139 + 0.00728,
        "length": 4 * 100,
    },
    "read_amp": 0.08,  # 08, # V
    "read_len": 10 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q3",
        "LO_IP": "192.168.1.102",
        "LO_pow": 18,  # dBm
        "LO_freq": 2.645 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq3",
        "mixer_off_I": -0.008602704109580368,
        "mixer_off_Q": -0.025946928709890925,
        "mixer_gain": -0.038995361328125,
        "mixer_phase": -0.03492431640625,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r3",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.091 * u.GHz,
    },
    "r_IF": -200.26 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr3",
        "mixer_off_I": -0.0020825495827011763,
        "mixer_off_Q": -0.0017681648721918468,
        "mixer_gain": 0.0,
        "mixer_phase": 0.0,
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
    "IF_freq": (-198.53359762937433) * u.MHz,  # -2.635
    "detuning": 0.12 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.029016721875383608,
        "detuning": 0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 200,
    },
    "Y180": {
        "amp": 0.029016721875383608,
        "phase": 0.026735403755329838,
        "length": 4 * 200,
    },
    "X90": {
        "amp": 0.029016721875383608,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 100,
    },
    "Y90": {
        "amp": 0.029016721875383608,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 100,
    },
    "read_amp": 0.02,  # 08, # V
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
        "LO_freq": 2.29 * u.GHz,
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
        "LO_freq": 7.218 * u.GHz,
    },
    "r_IF": -200.0 * u.MHz,  # Hz
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
    "IF_freq": -204.95266532065972 * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -153.02860657350504 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.2630131948963623,
        "detuning": 0 * u.MHz,
        "phase": 0,
        "length": 4 * 20,
    },
    "Y180": {
        "amp": 0.2630131948963623,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 20,
    },
    "X90": {
        "amp": 0.2630131948963623,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 10,
    },
    "Y90": {
        "amp": 0.2630131948963623,
        "detuning": 0.0 * u.MHz,
        "phase": -0.01139 + 0.00728,
        "length": 4 * 10,
    },
    "read_amp": 0.038,  # 08, # V
    "read_len": 10 * u.us,
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
        "LO_freq": 2.85 * u.GHz,
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
        "LO_freq": 7.332 * u.GHz,
    },
    "r_IF": -199.6 * u.MHz,  # Hz
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
    "IF_freq": -175.07131917257726 * u.MHz,
    "detuning": 0 * u.MHz,
    "anharmonicity": -144.81186474028618 * u.MHz,
    "ef_amp": 0.1350980772798654,
    "DRAG_beta": 0.00,
    "X180": {"amp": 0.023690251167764004, "phase": 0, "length": 4 * 200},
    "Y180": {"amp": 0.023690251167764004, "phase": 0, "length": 4 * 200},
    "X90": {"amp": 0.023690251167764004, "phase": 0, "length": 4 * 100},
    "Y90": {
        "amp": 0.023690251167764004,
        "phase": 0,
        "length": 4 * 100,
    },
    "read_amp": 0.02,  # V
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
        "LO_freq": 3.075 * u.GHz,
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
        "LO_freq": 7.4615 * u.GHz,
    },
    "r_IF": -200.3 * u.MHz,  # Hz
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
