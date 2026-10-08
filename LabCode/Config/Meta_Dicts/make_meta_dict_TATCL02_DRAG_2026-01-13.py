import sys
import numpy as np
sys.path.append('C:/Users/my/Desktop/Houck-Lab-Qua')
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

outer_folder = r'Z:\t1Team\Data\2025-12-20_BFG_cooldown\TATCL02\OPX'

reset_time = 3000 * u.us ### Test if you need explicit reset tics
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
    'envelope_type': 'COSINE',
    'pi_amp': 0.16404141981715978, # V
    'pi2_amp': 0.16404141981715978, # V
    'X180_len': 4 * 8,  # ns
    'X90_len': 4 * 4,  # ns
    'ef_amp': 0.16404141981715978,
    'DRAG_beta': 0.84,
    'read_amp': 0.006, # V
    'read_len': 17 * u.us,
    'read_theta':  0,
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -153.02860657350504 * u.MHz,
    'cw_amp': 0.390,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q1',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 2.6221 * u.GHz,
        },
    'bare_IF': -300.5185726550711 * u.MHz,
    'q_IF': (-297.68) * u.MHz, # 55
    'f_state_IF': (297.68 -160) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq1',
        'mixer_off_I': -0.011642351469345159,
        'mixer_off_Q': -0.027267622468571062,
        'mixer_gain': 0.02616501632583096,
        'mixer_phase': -0.16358270160926622,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r1',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, # dBm
        'LO_freq': 6.90215 * u.GHz,
        },
    'r_IF': -200.2 * u.MHz, # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr1',
        'mixer_off_I': -0.005782739073038102,
        'mixer_off_Q': -0.003554366901516914,
        'mixer_gain': 0.0071439108345657575,
        'mixer_phase': -0.061055407579988244,
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
    'read_amp': 0.006,#
    'read_len': 17 * u.us,
    'read_theta': 0,
    'envelope_type': 'COSINE',
    'DRAG_beta': -0,
    'pi_amp': 0.13616695938872064, # V
    'pi2_amp': 0.13616695938872064, # V
    'ef_amp': 0.11,
    'X180_len': 4 * 8,  # ns
    'X90_len': 4 * 4,  # ns
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
        'LO_freq': 2.483 * u.GHz,
        },
    'bare_IF': -198.95100000000002 * u.MHz,
    'q_IF': -197.311 * u.MHz,
    'f_state_IF': (-197.738242041 -155) * u.MHz,
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
        'LO_freq': 6.989 * u.GHz,
        },
    'r_IF': -199.8 *u.MHz,#200.25 * u.MHz,  # Hz
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
    'envelope_type': 'COSINE',
    'DRAG_beta': -0,
    'pi_amp': 0.13616695938872064, # V
    'pi2_amp': 0.13616695938872064, # V
    'ef_amp': 0.11,
    'X180_len': 4 * 80,  # ns
    'X90_len': 4 * 40,  # ns
    'read_amp': 0.005,  # 08, # V
    'read_len': 30 * u.us,
    'read_theta': 0,
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
        'LO_freq': 2.645 * u.GHz,
    },
    'q_IF': -198.65 * u.MHz,  # .509
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
        'LO_freq': 7.091 * u.GHz,
    },
    'r_IF': -200 * u.MHz,  # Hz
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
    'envelope_type': 'COSINE',
    'DRAG_beta': -0,
    'pi_amp': 0.13616695938872064, # V
    'pi2_amp': 0.13616695938872064, # V
    'ef_amp': 0.11,
    'X180_len': 4 * 80,  # ns
    'X90_len': 4 * 40,  # ns
    'read_amp': 0.015, # V
    'read_len': 20 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
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
        'LO_freq': 2.29 * u.GHz,
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
        'LO_freq': 7.218 * u.GHz,
        },
    'r_IF': -200.2 * u.MHz,  # Hz
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
    'envelope_type': 'COSINE',
    'DRAG_beta': -0,
    'pi_amp': 0.13616695938872064, # V
    'pi2_amp': 0.13616695938872064, # V
    'ef_amp': 0.11,
    'X180_len': 4 * 80,  # ns
    'X90_len': 4 * 40,  # ns
    'read_amp': 0.01, # V 11666666666666667, -199.83333333333334
    'read_len': 20 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 600,  # ns
    'gauss_len_X90': 4 * 300,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -199.83333333333334 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 1000 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q5',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 2.85 * u.GHz,
        },
    'q_IF': -199 * u.MHz,
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
        'LO_freq': 7.332 * u.GHz,
        },
    'r_IF': -199.75 * u.MHz,  # Hz
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
    'envelope_type': 'COSINE',
    'DRAG_beta': 1,
    'pi_amp': 0.13629472783964305, # V
    'pi2_amp': 0.13629472783964305, # V
    'ef_amp': 0.1416260887179332,
    'X180_len': 4 * 8,  # ns
    'X90_len': 4 * 4,  # ns
    'read_amp':  0.009, # V
    'read_len': 12 * u.us,
    'read_theta': 1.21522617081074,
    'gauss_len': 4 * 100,  # ns
    'gauss_len_X90': 4 * 50,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'anharmonicity': -144.81186474028618 * u.MHz,
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q6',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 3.075 * u.GHz,
        },
    'bare_IF': -298.74958923220066 * u.MHz,
    'q_IF': -299.55 * u.MHz, #-199.99141484848485 * u.MHz, #.509
    'f_state_IF': (-298 -144.81186474028618) * u.MHz,
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
        'LO_freq': 7.4615 * u.GHz,
        },
    'r_IF': -200.2 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr6',
        'mixer_off_I': -0.008646490414685098,
        'mixer_off_Q': -0.02805557856627274,
        'mixer_gain': 0.02705135345458984,
        'mixer_phase': -0.09634590148925784,
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