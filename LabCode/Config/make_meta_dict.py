import sys
import numpy as np

sys.path.append("C:/Users/my/Desktop/Houck-Lab-Qua")
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

outer_folder = r"Z:\t1Team\Data\2026-10-02 BFE Cooldown\TATQ01_CL_KOH_SiO2JJ_CanA\OPX"

# outer_folder = r"Z:\t1Team\Data\2026-08-17 BFE Cooldown\TATQ03_CL_04\OPX\noise_spec\chi"

reset_time = 5000 * u.us  ### Test if you need explicit reset tics
#resonator_dump_time = 50 * u.us
resonator_dump_time = 10 * u.us
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
    "IF_freq": (-198.66983) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": (-108.0) * u.MHz,
    "ef_amp": 0.33,
    "ef_len": 4 * 16,
    "DRAG_beta": 0,  # 1.250000,
    "X180": {
        "amp": 0.33,
        "phase": 0,
        "length": 4 * 10,
    },
    "Y180": {
        "amp": 0.33,
        "phase": 0.0,
        "length": 4 * 10,
    },
    "X90": {
        "amp": 0.33,  # 0.09
        "phase": 0,
        "length": 4 * 5,  # 8
    },
    "Y90": {
        "amp": 0.33,
        "phase": 0.0,
        "length": 4 * 5,  # 8
    },
    "read_amp": 0.0070,  # 08 +.2, 0.078
    "read_len": 15 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.25,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q1",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 1.65 * u.GHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq1",
        "mixer_off_I": 0,  # 0.0006928552324097833,
        "mixer_off_Q": 0,  # -0.028704561886115698,
        "mixer_gain": 0,  # 0.016570239269640297,
        "mixer_phase": 0,  # 0.13635128682362851,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r1",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 6.914741 * u.GHz,
    },
    "r_IF": -200.14 * u.MHz,  # -200.14 mHz for 3 state
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
    "IF_freq": (-198.5 - 1 + 0.4) * u.MHz,  # -2.635
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
    "X180": {"amp": 0.26, "phase": 0.0, "length": 4 * 128},
    "Y180": {"amp": 0.26, "phase": 0.0, "length": 4 * 128},
    "X90": {"amp": 0.26, "phase": 0.0, "length": 4 * 64},
    "Y90": {
        "amp": 0.26,
        "phase": 0.0,
        "length": 4 * 64,
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
    "read_amp": 0.0065,
    "read_len": 15 * u.us,
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
    "r_IF": -200.055 * u.MHz,  # Hz
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
    "read_amp": 0.015,  # 0.024,  # 08, # V
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
    "IF_freq": (-100) * u.MHz,  # -2.635
    "detuning": 0.12 * u.MHz,
    "anharmonicity": -200 * u.MHz,
    "ef_amp": 0.16404141981715978,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.1347,
        "detuning": 0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 1000,
    },
    "Y180": {
        "amp": 0.1347,
        "phase": 0.026735403755329838,
        "length": 4 * 1000,
    },
    "X90": {
        "amp": 0.1347,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 500,
    },
    "Y90": {
        "amp": 0.1347,
        "detuning": 0.0 * u.MHz,
        "phase": 0.02457870117879388,
        "length": 4 * 500,
    },
    "read_amp": 0.009,  # 08, # V
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
    "r_IF": -199.92 * u.MHz,  # Hz
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
    "IF_freq": (-100.89 + 0.75) * u.MHz,  # -2.635
    "detuning": 0 * u.MHz,
    "anharmonicity": -202.09 * u.MHz,
    "ef_amp": 0.0315,
    "ef_len": 4 * 200,
    "DRAG_beta": 0,
    "X180": {
        "amp": 0.0633,
        "detuning": 0 * u.MHz,
        "phase": 0,
        "length": 4 * 200,
    },
    "Y180": {
        "amp": 0.0633,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 200,
    },
    "X90": {
        "amp": 0.0633,
        "detuning": 0.0 * u.MHz,
        "phase": 0.0,
        "length": 4 * 100,
    },
    "Y90": {
        "amp": 0.0633,
        "detuning": 0.0 * u.MHz,
        "phase": -0.01139 + 0.00728,
        "length": 4 * 100,
    },
    "read_amp": 0.018,  # 08, # V
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
        "LO_freq": 2.456 * u.GHz,
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
        "LO_freq": 7.272 * u.GHz,
    },
    "r_IF": -200.2 * u.MHz,  # Hz
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
    "IF_freq": (-99.77 - 0.14 - 0.04 - 0.159 - 0.050) * u.MHz,
    "anharmonicity": -175 * u.MHz,
    "DRAG_beta": 0,
    # "detuning": 0 * u.MHz,
    # "X180": {"amp": 0.37619709236395016, "phase": 0, "length": 4 * 8},
    # "Y180": {"amp": 0.37619709236395016, "phase": 0, "length": 4 * 8},
    # "X90": {"amp": 0.37619709236395016, "phase": 0, "length": 4 * 4},
    # "Y90": {
    #     "amp": 0.37619709236395016,
    #     "phase": 0,
    #     "length": 4 * 4,
    # },
    # "ef_amp": 0.2362,
    # "detuning": 0 * u.MHz,
    # "X180": {"amp": 0.27834282241534436, "phase": 0, "length": 4 * 10},
    # "Y180": {"amp": 0.27834282241534436, "phase": 0, "length": 4 * 10},
    # "X90": {"amp": 0.27834282241534436, "phase": 0, "length": 4 * 5},
    # "Y90": {
    #     "amp": 0.27834282241534436,
    #     "phase": 0,
    #     "length": 4 * 5,
    # },
    "ef_amp": 0.1,
    "ef_len": 4 * 16,
    "detuning": 0 * u.MHz,
    "X180": {"amp": 0.2400, "phase": 0, "length": 4 * 200},
    "Y180": {"amp": 0.2400, "phase": 0, "length": 4 * 200},
    "X90": {"amp": 0.2400, "phase": 0, "length": 4 * 100},
    "Y90": {
        "amp": 0.2400,
        "phase": 0,
        "length": 4 * 100,
    },
    "read_amp": 0.018,  # V
    "read_len": 15 * u.us,
    "read_theta": 1.21522617081074,
    "reset_time": int(reset_time),
    "resonator_dump_time": resonator_dump_time,
    "cw_amp": 0.051,
    "cw_len": 500 * u.ns,
    # Qubit frequency parameters
    "q_LO": {
        "LO_name": "q6",
        "LO_IP": "192.168.1.102",
        "LO_pow": 15,  # dBm
        "LO_freq": 2583 * u.MHz,
    },
    # Qubit mixer info
    "q_mixer": {
        "mixer_name": "mq6",
        "mixer_off_I": -0.004134885864147009,  # 0.008154141862178221,
        "mixer_off_Q": -0.005638547276706162,  # - 0.010805984970647841,
        "mixer_gain": 0.018690947611321465,  # 0.03854081821555155,
        "mixer_phase": -0.029741615687089516,  # - 0.004707105275883807,
    },
    # Resonator frequency parameters
    "r_LO": {
        "LO_name": "r6",
        "LO_IP": "192.168.1.178",
        "LO_mod": 31,
        "LO_pow": 15,  # dBm
        "LO_freq": 7154 * u.MHz,
    },
    "r_IF": -199.58 * u.MHz,  # Hz
    # Resonator mixer info
    "r_mixer": {
        "mixer_name": "mr6",
        "mixer_off_I": 0.00021542196527804045,  # 0.008154141862178221,
        "mixer_off_Q": -0.009428887119722167,  # - 0.010805984970647841,
        "mixer_gain": 0.017707712247010936,  # 0.03854081821555155,
        "mixer_phase": -0.008314381296622745,
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
