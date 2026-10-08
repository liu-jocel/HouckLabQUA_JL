import sys
import numpy as np

sys.path.append("C:/Users/my/Desktop/Houck-Lab-Qua")
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

outer_folder = r"Z:\t1Team\Data\2026-07-06 BFE Cooldown\TATQ01_CL_KOH_recooldown\OPX"
# outer_folder = r"Z:/t1Team/Data/2026-04-27_BFG_cooldown/TATCL02/OPX/rb_fidelity_sweep_DRAG_1"

reset_time = 5000 * u.us  ### Test if you need explicit reset tics
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
    "IF_freq": (-98.50) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": (-200) * u.MHz,
    "ef_amp": 0.1,
    "ef_len": 4 * 10,
    "DRAG_beta": 0,  # 1.250000,
    "X180": {"amp": 0.1, "phase": 0, "length": 4 * 150},
    "Y180": {"amp": 0.1, "phase": 0.0, "length": 4 * 150},
    "X90": {
        "amp": 0.1,
        "phase": 0,
        "length": 4 * 4,  # 8
    },
    "Y90": {
        "amp": 0.1,
        "phase": 0.0,
        "length": 4 * 4,  # 8
    },
    "read_amp": 0.02,  # 08 +.2,
    "read_len": 15 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.45,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q1",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 2.650 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq1",
        "mixer_off_I": -0.00459541414893465,
        "mixer_off_Q": -0.01212649365552352,
        "mixer_gain": 0.05105850228865166,
        "mixer_phase": 0.028802921000169603,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r1",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.783 * u.GHz,
    },
    "r_IF": -199.87 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr1",
        "mixer_off_I": -0.008499760754057206,
        "mixer_off_Q": -0.005672241916181519,
        "mixer_gain": 0.002491080109029983,
        "mixer_phase": -0.05174053851515057,
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
    "IF_freq": (-99.11) * u.MHz,  # -2.635
    "anharmonicity": -211.25 * u.MHz,
    "ef_amp": 0.18,
    "ef_len": 4 * 20,
    "DRAG_beta": 0.0,
    # "detuning": 0 * u.MHz,
    # "X180": {"amp": 0.0850, "phase": 0.0, "length": 4 * 200},
    # "Y180": {"amp": 0.0850, "phase": 0.0, "length": 4 * 200},
    # "X90": {"amp": 0.0850, "phase": 0.0, "length": 4 * 100},
    # "Y90": {"amp": 0.0850, "phase": 0.0, "length": 4 * 100,},
    "detuning": 0.0 * u.MHz,
    "X180": {"amp": 0.2700, "phase": 0.0, "length": 4 * 16},
    "Y180": {"amp": 0.2700, "phase": 0.0, "length": 4 * 16},
    "X90": {"amp": 0.2700, "phase": 0.0, "length": 4 * 8},
    "Y90": {
        "amp": 0.2700,
        "phase": 0.0,
        "length": 4 * 8,
    },
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q2",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 2.9464 * u.GHz,
    },
    "cw_amp": 0.27 / 2,  # 0.036117,
    "cw_len": 1000 * u.ns,
    "read_amp": 0.01,
    "read_len": 20 * u.us,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "read_theta": 0,
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq2",
        "mixer_off_I": 0.0028017272627565144,  # --#0.0016280851833855086,
        "mixer_off_Q": -0.014898554228520312,  # - 0.013066217145851058,
        "mixer_gain": 0.020021786118377353,  # 0.01689636335431715,
        "mixer_phase": 0.037188794012581634,  # 0.04219113860199286,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r2",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.8733 * u.GHz,
    },
    "r_IF": -200.07 * u.MHz,  # Hz
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
    "q_I_channel": 5,
    "q_Q_channel": 2,
    "r_I_channel": 9,
    "r_Q_channel": 10,
    "out_channel": 1,
    "T_1": 1000 * u.us,
    "T_2R": 100 * u.us,
    "T_2E": 100 * u.us,
    "envelope_type": "COSINE",
    "IF_freq": (-98.95) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -200 * u.MHz,
    "ef_amp": 0.1500,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.1500,
        "detuning": 0 * u.MHz,
        "phase": 0,
        "length": 4 * 200,
    },
    "Y180": {
        "amp": 0.1500,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 200,
    },
    "X90": {
        "amp": 0.1500,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 100,
    },
    "Y90": {
        "amp": 0.1500,
        "detuning": 0.0 * u.MHz,
        "phase": -0.01139 + 0.00728,
        "length": 4 * 100,
    },
    "read_amp": 0.015, #0.024,  # 08, # V
    "read_len": 14 * u.us,
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
        "LO_freq": 3.2205 * u.GHz,
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
        "LO_freq": 6.992 * u.GHz,
    },
    "r_IF": -199.87 * u.MHz,  # Hz
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
    "IF_freq": (-98.93) * u.MHz,  # -2.635
    "detuning": 0.12 * u.MHz,
    "anharmonicity": -200 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.1347,
        "detuning": 0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 200,
    },
    "Y180": {
        "amp": 0.1347,
        "phase": 0.026735403755329838,
        "length": 4 * 200,
    },
    "X90": {
        "amp": 0.1347,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 100,
    },
    "Y90": {
        "amp": 0.1347,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 100,
    },
    "read_amp": 0.008,  # 08, # V
    "read_len": 20 * u.us,
    "read_theta": 0,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.02,
    "cw_len": 500 * u.ns,
    "q_LO": {
        "LO_name": "q4",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 3.437 * u.GHz,
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
        "LO_freq": 7.08235 * u.GHz,
    },
    "r_IF": -199.98 * u.MHz,  # Hz
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
    "IF_freq": (-248.328) * u.MHz,
    "anharmonicity": -197.67 * u.MHz,
    "DRAG_beta": 0,
    "ef_amp": 0.07,
    "ef_len": 4 * 25,
    "detuning": 0 * u.MHz,
    "X180": {"amp": 0.27049656021146584, "phase": 0, "length": 4 * 8},
    "Y180": {"amp": 0.27049656021146584, "phase": 0, "length": 4 * 8},
    "X90": {"amp": 0.27049656021146584, "phase": 0, "length": 4 * 4},
    "Y90": {
        "amp": 0.27049656021146584,
        "phase": 0,
        "length": 4 * 4,
    },
    "read_amp": 0.005,  # V
    "read_len": 20 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.11,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q6",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 3.7896 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq6",
        "mixer_off_I": -0.002738994702258423,
        "mixer_off_Q": -0.0020501098187196472,
        "mixer_gain": 0.014628346543759113,
        "mixer_phase": 0.1708663679193706,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r6",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7.2639 * u.GHz,
    },
    "r_IF": -200.05 * u.MHz,  # Hz
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
