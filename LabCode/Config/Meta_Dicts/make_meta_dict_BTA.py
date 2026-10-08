import sys
import numpy as np
sys.path.append('C:/Users/my/Desktop/Houck-Lab-Qua')
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

outer_folder = r'Z:\t1Team\Data\2025-09-19_cooldown\OPX\TATQ01_bTa_Sapphire_02'

reset_time = 3 * u.ms ### Test if you need explicit reset tics
resonator_dump_time = 100 * u.us
global_params = {
    'read_len': 17000,  #ns
    'reset_time': reset_time, #s
    'outer_folder': outer_folder

}

q1 = {
    # Global parameters #Q3
    'global_params': global_params,
    'element_name': 'q1',
    'q_name': 'q1',
    'r_name': 'r1',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.45,#0.08031818181818182, # V
    'pi2_amp': 0.45,#0.08031818181818182, # V
    'ef_amp': 0.2,
    'DRAG_beta': -0,
    'read_amp': 0.4, # V
    'read_len': 25 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 500,  # ns
    'gauss_len_X90': 4 * 250,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -100*u.MHz,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q1',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 5.126 * u.GHz,
        },
    'q_IF': -200 * u.MHz,
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq1',
        'mixer_off_I': -0.01,
        'mixer_off_Q': -0.022,
        'mixer_gain': 0.00,
        'mixer_phase': -0.04,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r1',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, # dBm
        'LO_freq': 7.73597 * u.GHz,
        },
    'r_IF': -200 * u.MHz, # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr1',
        'mixer_off_I': 0.015,
        'mixer_off_Q': -0.0015,
        'mixer_gain': 0.01,
        'mixer_phase': 0.015,
        },
}

q2 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q2',
    'q_name': 'q2',
    'r_name': 'r2',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.125,
    'pi2_amp': 0.125, # V
    'ef_amp': 0.25,
    'DRAG_beta': -0,
    'read_amp': 0.2, # V
    'read_len': 15 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 20,  # ns
    'gauss_len_X90': 4 * 10,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q2',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.8935 * u.GHz,
        },
    'q_IF': -200 * u.MHz,
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq2',
        'mixer_off_I': -0.01,
        'mixer_off_Q': -0.022,
        'mixer_gain': 0.00,
        'mixer_phase': -0.04,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r2',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 7.95049 * u.GHz,
        },
    'r_IF': -201.5 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr2',
        'mixer_off_I': 0.015,
        'mixer_off_Q': -0.0015,
        'mixer_gain': 0.01,
        'mixer_phase': 0.015,
        },
}

q3 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q3',
    'q_name': 'q3',
    'r_name': 'r3',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.125,  # V
    'pi2_amp': 0.125,  # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.05,  # 08, # V
    'read_len': 8 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 50,  # ns
    'gauss_len_X90': 4 * 25,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    # Qubit frequency parameters
    'q_LO': {
        'LO_name': 'q3',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18,  # dBm
        'LO_freq': 5.9785 * u.GHz,
    },
    'q_IF': -199 * u.MHz,  # .509
    'f_state_IF': (-164.55 - 238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq3',
        'mixer_off_I': -0.01,
        'mixer_off_Q': -0.022,
        'mixer_gain': 0.00,
        'mixer_phase': -0.04,
    },
    # Resonator frequency parameters
    'r_LO': {
        'LO_name': 'r3',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15,  # dBm
        'LO_freq': 8.11895 * u.GHz,
    },
    'r_IF': -200.075 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer': {
        'mixer_name': 'mr3',
        'mixer_off_I': 0.015,
        'mixer_off_Q': -0.0015,
        'mixer_gain': 0.01,
        'mixer_phase': 0.015,
    },
}

q4 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q4',
    'q_name': 'q4',
    'r_name': 'r4',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.45, # V
    'pi2_amp': 0.45, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.2, # V
    'read_len': 15 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 50,  # ns
    'gauss_len_X90': 4 * 25,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q4',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.5583 * u.GHz,
        },
    'q_IF': -190 * u.MHz, #.509
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq4',
        'mixer_off_I': -0.01,
        'mixer_off_Q': -0.022,
        'mixer_gain': 0.00,
        'mixer_phase': -0.04,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r4',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 8.0872 * u.GHz,
        },
    'r_IF': -100 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr4',
        'mixer_off_I': 0.015,
        'mixer_off_Q': -0.0015,
        'mixer_gain': 0.01,
        'mixer_phase': 0.015,
        },
}

q5 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q5',
    'q_name': 'q5',
    'r_name': 'r5',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.45, # V
    'pi2_amp': 0.45, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.2, # V 11666666666666667, -199.83333333333334
    'read_len': 15 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 200,  # ns
    'gauss_len_X90': 4 * 100,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -199.83333333333334 * u.MHz,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q5',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.563 * u.GHz,
        },
    'q_IF': -199 * u.MHz,
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq5',
        'mixer_off_I': -0.01,
        'mixer_off_Q': -0.022,
        'mixer_gain': 0.00,
        'mixer_phase': -0.04,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r5',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 8.43322 * u.GHz,
        },
    'r_IF': -200 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr5',
        'mixer_off_I': 0.0, #0.015,
        'mixer_off_Q': 0.0, #-0.0015,
        'mixer_gain': 0.0, #0.01,
        'mixer_phase': 0.0, #0.015,
        },
}

q6 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q6',
    'q_name': 'q6',
    'r_name': 'r6',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.2, # V
    'pi2_amp': 0.2, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp':  0.07, # V
    'read_len': 5 * u.us,
    'read_theta': 1.21522617081074,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q6',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 5.415 * u.GHz,
        },
    'q_IF': -200 * u.MHz, #-199.99141484848485 * u.MHz, #.509
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq6',
        'mixer_off_I': -0.01,
        'mixer_off_Q': -0.022,
        'mixer_gain': 0.00,
        'mixer_phase': -0.04,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r6',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 6.953 * u.GHz,
        },
    'r_IF': -199 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr6',
        'mixer_off_I': 0.015,
        'mixer_off_Q': -0.0015,
        'mixer_gain': 0.01,
        'mixer_phase': 0.015,
        },
}
def make_meta_dict(qubit):
    if qubit == 'q1':
        return {
            'global_params': global_params,
            'q1': q1,
            "outer_folder": outer_folder}
    if qubit == 'q2':
        return {
            'global_params': global_params,
            'q2': q2,
            "outer_folder": outer_folder}
    if qubit == 'q3':
        return {
            'global_params': global_params,
            'q3': q3,
            "outer_folder": outer_folder}
    if qubit == 'q4':
        return {
            'global_params': global_params,
            'q4': q4,
            "outer_folder": outer_folder}
    if qubit == 'q5':
        return {
            'global_params': global_params,
            'q5': q5,
            "outer_folder": outer_folder}
    if qubit == 'q6':
        return {
            'global_params': global_params,
            'q6': q6,
            "outer_folder": outer_folder}


def make_meta_dict_6Q():
    return {qubit: make_meta_dict(qubit) for qubit in ["q1", "q2", "q3", "q4", "q5", "q6"]}