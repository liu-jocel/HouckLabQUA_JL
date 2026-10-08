import sys
import numpy as np

sys.path.append("C:/Users/my/Desktop/Houck-Lab-Qua")
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

outer_folder = r"Z:\t1Team\Data\2026-08-17 BFE Cooldown\TATQ04_CL_01\OPX"
# outer_folder = r"Z:/t1Team/Data/2026-04-27_BFG_cooldown/TATCL02/OPX/rb_fidelity_sweep_DRAG_1"

reset_time = 10000 * u.us  ### Test if you need explicit reset tics
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
    "IF_freq": (-198.9) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": (-106.000) * u.MHz,
    "ef_amp": 0.0900,
    "ef_len": 4 * 10,
    "DRAG_beta": 0.0,  # 1.250000,
    "X180": {"amp": 0.1333, "phase": 0, "length": 4 * 8},
    "Y180": {"amp": 0.1333, "phase": 0.0, "length": 4 * 8},
    "X90": {
        "amp": 0.1333,  # 0.09
        "phase": 0,
        "length": 4 * 4,  # 8
    },
    "Y90": {
        "amp": 0.1333,
        "phase": 0.0,
        "length": 4 * 4,  # 8
    },
    "read_amp": 0.0066,  # 08 +.2, 0.078
    "read_len": 20 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.3,
    "cw_len": 16 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q1",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 1.762 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq1",
        "mixer_off_I": -0.00281509073031998,
        "mixer_off_Q": -0.002349534922507246,
        "mixer_gain": -0.011259302377584393,
        "mixer_phase": -0.025582190504064784,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r1",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.90468 * u.GHz,
    },
    "r_IF": -200.2325 * u.MHz,  # -200.14 mHz for 3 state
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr1",
        "mixer_off_I": 0.0018344152433201091,  # -0.004999034877255325,
        "mixer_off_Q": -0.006174865512366524,  # 0.00017403283418389084,
        "mixer_gain": 0.009304311871528628,  # -0.013582520843192471,
        "mixer_phase": 0.0023788467049598696,  # 0.022366259401314904,
    },
}
# q1 = {
#     # Global parameters #Q1
#     "global_params": global_params,
#     "element_name": "q1",
#     "q_name": "q1",
#     "r_name": "r1",
#     "q_I_channel": 1,
#     "q_Q_channel": 2,
#     "r_I_channel": 9,
#     "r_Q_channel": 10,
#     "out_channel": 1,
#     "T_1": 1000 * u.us,
#     "T_2R": 100 * u.us,
#     "T_2E": 100 * u.us,
#     "envelope_type": "COSINE",
#     "IF_freq": (-198.5 + 1) * u.MHz,  # -2.635
#     "detuning": 0 * u.MHz,
#     "anharmonicity": (-100) * u.MHz,
#     "ef_amp": 0.21,
#     "ef_len": 4 * 10,
#     "DRAG_beta": 0,  # 1.250000,
#     "X180": {"amp": 0.288, "phase": 0, "length": 4 * 8},
#     "Y180": {"amp": 0.288, "phase": 0.0, "length": 4 * 8},
#     "X90": {
#         "amp": 0.288,
#         "phase": 0,
#         "length": 4 * 4,  # 8
#     },
#     "Y90": {
#         "amp": 0.288,
#         "phase": 0.0,
#         "length": 4 * 4,  # 8
#     },
#     "read_amp": 0.0075,  # 08 +.2,
#     "read_len": 15 * u.us,
#     "read_theta": 1.21522617081074,
#     "reset_time": int(reset_time),
#     "resonator_dump_time": resonator_dump_time,
#     "cw_amp": 0.4,
#     "cw_len": 5000 * u.ns,
#     # Qubit frequency parameters
#     "q_LO": {
#         "LO_name": "q1",
#         "LO_IP": "192.168.1.102",
#         "LO_pow": 14,  # dBm
#         "LO_freq": 1550 * u.MHz,
#     },
#     # Qubit mixer info
#     "q_mixer": {
#         "mixer_name": "mq1",
#         "mixer_off_I": 0.002494120078154082,
#         "mixer_off_Q": -0.0018745108744042235,
#         "mixer_gain": 0.01564006952394266,
#         "mixer_phase": -0.0115656292939093,
#     },
#     # Resonator frequency parameters
#     "r_LO": {
#         "LO_name": "r1",
#         "LO_IP": "192.168.1.178",
#         "LO_mod": 31,
#         "LO_pow": 15,  # dBm
#         "LO_freq": 6904.68 * u.MHz,
#     },
#     "r_IF": -200.1 * u.MHz,  # Hz
#     # Resonator mixer info
#     "r_mixer": {
#         "mixer_name": "mr1",
#         "mixer_off_I": -0.008499760754057206,
#         "mixer_off_Q": -0.005672241916181519,
#         "mixer_gain": 0.002491080109029983,
#         "mixer_phase": -0.05174053851515057,
#     },
# }

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
    "IF_freq": (-98.75 - 0.150) * u.MHz,  # -2.635
    "anharmonicity": -212.5 * u.MHz,
    "ef_amp": 0.1800,
    "ef_len": 4 * 10,
    "DRAG_beta": 0.0,
    # "detuning": 0 * u.MHz,
    # "X180": {"amp": 0.0850, "phase": 0.0, "length": 4 * 200},
    # "Y180": {"amp": 0.0850, "phase": 0.0, "length": 4 * 200},
    # "X90": {"amp": 0.0850, "phase": 0.0, "length": 4 * 100},
    # "Y90": {"amp": 0.0850, "phase": 0.0, "length": 4 * 100,},
    "detuning": 0.0 * u.MHz,
    "X180": {"amp": 0.0675, "phase": 0.0, "length": 4 * 120},
    "Y180": {"amp": 0.0675, "phase": 0.0, "length": 4 * 120},
    "X90": {"amp": 0.0675, "phase": 0.0, "length": 4 * 60},
    "Y90": {
        "amp": 0.0675,
        "phase": 0.0,
        "length": 4 * 60,
    },
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q2",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 1910 * u.MHz,
    },
    "cw_amp": 0.27 / 2,  # 0.036117,
    "cw_len": 1000 * u.ns,
    "read_amp": 0.0072,
    "read_len": 15 * u.us,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "read_theta": 0,
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq2",
        "mixer_off_I": 0.0028017272627565144,  # --#0.0016280851833855086,
        "mixer_off_Q": -0.014898554228520312,  # - 0.013066217145851058,
        "mixer_gain": 0.02351983562111854,  # 0.01689636335431715,
        "mixer_phase": -0.06813005656003951,  # 0.04219113860199286,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r2",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6928.75 * u.MHz,
    },
    "r_IF": -200.125 * u.MHz,  # Hz
    # "r_IF": -199.89 * u.MHz,  # 200.25 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr2",
        "mixer_off_I": -0.0016348354964975443,
        "mixer_off_Q": -0.01305353267580358,
        "mixer_gain": 0.01699355096934596,
        "mixer_phase": 0.04212052769798902,
    },
}

q3 = {
    # Global parameters #q5
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
    "IF_freq": (-99.39 + 0.200 - 0.350) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -100 * u.MHz,
    "ef_amp": 0.1500,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.0900,
        "detuning": 0 * u.MHz,
        "phase": 0,
        "length": 4 * 1200,
    },
    "Y180": {
        "amp": 0.0900,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 1200,
    },
    "X90": {
        "amp": 0.0900,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 600,
    },
    "Y90": {
        "amp": 0.0900,
        "detuning": 0.0 * u.MHz,
        "phase": -0.01139 + 0.00728,
        "length": 4 * 600,
    },
    "read_amp": 0.007,  # 0.024,  # 08, # V
    "read_len": 25 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q3",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 2121 * u.MHz,
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
        "LO_freq": 7111 * u.MHz,
    },
    "r_IF": -200.0575 * u.MHz,  # Hz
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
    "IF_freq": (-97.87 - 0.010) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -100 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.28,
        "detuning": 0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 600,
    },
    "Y180": {
        "amp": 0.28,
        "phase": 0.026735403755329838,
        "length": 4 * 600,
    },
    "X90": {
        "amp": 0.28,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 300,
    },
    "Y90": {
        "amp": 0.28,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 300,
    },
    "read_amp": 0.01,  # 08, # V
    "read_len": 15 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    "q_LO": {
        "LO_name": "q4",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 2249 * u.MHz,
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
        "LO_IP": "192.168.1.102",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7211 * u.MHz,
    },
    "r_IF": -200.4 * u.MHz,  # Hz
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
    "IF_freq": -103 * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -201 * u.MHz,
    "ef_amp": 0.0325,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.0260,
        "detuning": 0 * u.MHz,
        "phase": 0,
        "length": 4 * 200,
    },
    "Y180": {
        "amp": 0.0260,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 200,
    },
    "X90": {
        "amp": 0.0260,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 2000,
    },
    "Y90": {
        "amp": 0.0260,
        "detuning": 0.0 * u.MHz,
        "phase": -0.01139 + 0.00728,
        "length": 4 * 100,
    },
    "read_amp": 0.004,  # 08, # V
    "read_len": 15 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.06994490989045858,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q5",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 3.135 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq5",
        "mixer_off_I": -0.002682060003280641,
        "mixer_off_Q": -0.009790980815887454,
        "mixer_gain": 0.005842852592468259,
        "mixer_phase": 0.12374358177185057,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r5",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.19594 * u.GHz,
    },
    "r_IF": -200.45 * u.MHz,  # Hz
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
    "IF_freq": (-148.328) * u.MHz,
    "anharmonicity": -100 * u.MHz,
    "DRAG_beta": 0,
    "ef_amp": 0.135,
    "ef_len": 4 * 25,
    "detuning": 0 * u.MHz,
    "X180": {"amp": 0.27049656021146584, "phase": 0, "length": 4 * 8},
    "Y180": {"amp": 0.27049656021146584, "phase": 0, "length": 4 * 8},
    "X90": {"amp": 0.4, "phase": 0, "length": 4 * 4},
    "Y90": {
        "amp": 0.27049656021146584,
        "phase": 0,
        "length": 4 * 4,
    },
    "read_amp": 0.04,  # V
    "read_len": 20 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.4,
    "cw_len": 100 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q6",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 3.5896 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq6",
        "mixer_off_I": 0.0014828080714323743,
        "mixer_off_Q": -0.009450908423787041,
        "mixer_gain": 0.008414803647974624,
        "mixer_phase": 0.030603900043206525,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r6",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.2639 * u.GHz,
    },
    "r_IF": -198.4 * u.MHz,  # Hz
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
