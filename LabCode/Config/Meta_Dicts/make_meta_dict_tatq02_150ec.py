import sys
import numpy as np
sys.path.append('/')
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

outer_folder = r'Z:\t1Team\Data\2025-12-20_BFG_cooldown\TATQ02_150Ec\OPX'

reset_time = 1000 * u.us ### Test if you need explicit reset tics
resonator_dump_time = 50 * u.us
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
    'pi_amp': 0.08202020202020202, # V
    'pi2_amp': 0.08202020202020202, # V
    'ef_amp': 0.022,
    'DRAG_beta': -0,
    'read_amp': 0.019, # V
    'read_len': 10 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 100,  # ns
    'gauss_len_X90': 4 * 50,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -204*u.MHz,
    'cw_amp': 0.15068,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q1',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 2.553 * u.GHz,
        },
    'q_IF': -198 * u.MHz,
    'f_state_IF': (-198.372356974 - 204) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq1',
        'mixer_off_I': -0.010554566470091231,
        'mixer_off_Q': -0.019456801057094714,
        'mixer_gain': -0.003319165465654807,
        'mixer_phase': -0.020661258010659374,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r1',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, # dBm
        'LO_freq': 6.9113 * u.GHz,
        },
    'r_IF': -200.15 * u.MHz, # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr1',
        'mixer_off_I': -0.001145020824333187,
        'mixer_off_Q': -0.0007977482135174799,
        'mixer_gain': -0.00968465805053711,
        'mixer_phase': -0.008866453170776365,
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
    'pi_amp': 0.1868181818181818,
    'pi2_amp': 0.1868181818181818, # V
    'ef_amp': 0.023,
    'DRAG_beta': -0,
    'read_amp': 0.025, # V
    'read_len': 10 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 100,  # ns
    'gauss_len_X90': 4 * 50,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -204.5 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q2',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 2.594 * u.GHz,
        },
    'q_IF': -199.5 * u.MHz,
    'f_state_IF': (-197.738242041 -204.5) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq2',
        'mixer_off_I': -0.011142855831712952,
        'mixer_off_Q': -0.019540043469169172,
        'mixer_gain': -0.010759362671524293,
        'mixer_phase': -0.07537375041283667,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r2',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 7.0092 * u.GHz,
        },
    'r_IF': -200.2 *u.MHz,#200.25 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr2',
        'mixer_off_I': -0.0007536417375376913,
        'mixer_off_Q': -0.001578431572124828,
        'mixer_gain': -0.01343142986297607,
        'mixer_phase': -0.014688253402709961,
        },
}

q3 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q3',
    'q_name': 'q3',
    'r_name': 'r3',
    'q_I_channel': 5,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'out_channel': 1,
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.25, # V
    'pi2_amp': 0.25, # V
    'ef_amp': 0.4,
    'DRAG_beta': -0,
    'read_amp': 0.05,  # 08, # V
    'read_len': 10 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 300,  # ns
    'gauss_len_X90': 4 * 150,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO': {
        'LO_name': 'q3',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18,  # dBm
        'LO_freq': 2.604 * u.GHz,
    },
    'q_IF': -199.3 * u.MHz,  # .509
    'f_state_IF': (-199.5 - 188 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq3',
        'mixer_off_I': -0.01053261756896973,
        'mixer_off_Q': -0.021893405914306645,
        'mixer_gain': 0.003621101379394532,
        'mixer_phase': -0.11101036071777343,
    },
    # Resonator frequency parameters
    'r_LO': {
        'LO_name': 'r3',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15,  # dBm
        'LO_freq': 7.11524 * u.GHz,
    },
    'r_IF': -200.1 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer': {
        'mixer_name': 'mr3',
        'mixer_off_I': -0.0020825495827011763,
        'mixer_off_Q': -0.0017681648721918468,
        'mixer_gain': 0.0,
        'mixer_phase': 0.0,
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
    'pi_amp': 0.044, #0.03673897174873737, # V
    'pi2_amp': 0.044, #0.03673897174873737, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.020, # V
    'read_len': 15 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 20,  # ns
    'gauss_len_X90': 4 * 10,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q4',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 2.698 * u.GHz,
        },
    'q_IF': -200 * u.MHz,
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq4',
        'mixer_off_I': -0.010961978015257045,
        'mixer_off_Q': -0.0196587074897252,
        'mixer_gain': -0.019372490048408506,
        'mixer_phase': -0.08354914337396627,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r4',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 7.2331 * u.GHz,
        },
    'r_IF': -200.3 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr4',
        'mixer_off_I': -0.00343116857111454,
        'mixer_off_Q': -0.001192570477724075,
        'mixer_gain': 0.0,
        'mixer_phase': 0.0,
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
    'pi_amp': 0.04, #0.04, # V
    'pi2_amp': 0.04, #0.04, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.03, # V 11666666666666667, -199.83333333333334
    'read_len': 8 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 200,  # ns
    'gauss_len_X90': 4 * 100,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -199.83333333333334 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q5',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 2.95 * u.GHz,
        },
    'q_IF': -197.6 * u.MHz,
    'f_state_IF': (-164.55-238 * u.MHz) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq5',
        'mixer_off_I': -0.01,
        'mixer_off_Q': 0.02,
        'mixer_gain': 0.0,
        'mixer_phase': 0.0,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r5',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 7.350 * u.GHz,
        },
    'r_IF': -199.6 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr5',
        'mixer_off_I': -0.011107224319130186,
        'mixer_off_Q': -0.020174260623753066,
        'mixer_gain': -0.00859904289245606,
        'mixer_phase': -0.1101691246032715,
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
    'pi_amp': 0.3015151515151515, # V
    'pi2_amp': 0.3015151515151515, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp':  0.03, # V
    'read_len': 10 * u.us,
    'read_theta': 1.21522617081074,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -200 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q6',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 3.126 * u.GHz,
        },
    'q_IF': -199.5 * u.MHz, #-199.99141484848485 * u.MHz, #.509
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
        'LO_freq': 7.4803 * u.GHz,
        },
    'r_IF': -200.2 * u.MHz,  # Hz
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