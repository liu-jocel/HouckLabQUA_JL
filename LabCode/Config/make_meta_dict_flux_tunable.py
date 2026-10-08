import sys
import numpy as np
sys.path.append('C:/Users/my/Desktop/Houck-Lab-Qua')
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

outer_folder = r'Z:\t1Team\Data\2025-12-20_BFG_cooldown\FTT_v02\OPX'

reset_time = 500 * u.us ### Test if you need explicit reset tics
resonator_dump_time = 15 * u.us
flux_settle_time = 100 * u.ns
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
    'flux_name': 'f1',
    'r_name': 'r1',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'flux_rf_I_channel': 3,
    'flux_rf_Q_chanel': 4,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 3,
    'out_channel': 1,
    'flux_dc_offset':  0,
    'YOKO_dc_offset':  0.69,# -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.04, # V
    'pi2_amp': 0.04, # V
    'ef_amp': 0.022,
    'DRAG_beta': -0,
    'read_amp': 0.091, # V
    'read_len': 20 * u.us,
    'read_theta':  0,
    'gauss_len': 4*400,  # ns
    'gauss_len_X90': 4*200,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -204*u.MHz,
    'resonator_fit_parameters': [1501719.5380980964,1.344585417924919,1.3452764370140642,-200422304.99269578],
    'cw_amp': 0.02,
    'cw_len': 100 * u.ns,
    'const_flux_len': 100*u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q1',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 3 * u.GHz,

        },
    'q_IF': -200 * u.MHz,
    'f_state_IF': (-198.372356974 - 204) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq1',
        'mixer_off_I': -0.011010570492362606,
        'mixer_off_Q': -0.019294344994705167,
        'mixer_gain': -0.018072522373404353,
        'mixer_phase': -0.09508663669112141,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r1',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, # dBm
        'LO_freq': 7.372* u.GHz,
        },
    'r_IF': -200 * u.MHz, # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr1',
        'mixer_off_I': -0.0045331554370932284,
        'mixer_off_Q': -0.00020641707233153245,
        'mixer_gain': 0.0337099552154541,
        'mixer_phase': -0.005480670928955078,
        },
}

q2 = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q2',
    'q_name': 'q2',
    'flux_name': 'f2',
    'r_name': 'r2',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 5,
    'out_channel': 1,
    'flux_dc_offset': -0.625,  # (SAME AS q1) static resting point of the qubit
    'YOKO_dc_offset': -0.625,  # -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.029,
    'pi2_amp': 0.029, # V
    'ef_amp': 0.023,
    'DRAG_beta': -0,
    'read_amp': 0.027, #0.065, # V
    'read_len': 5 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 48,  # ns
    'gauss_len_X90': 4 * 24,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -204.5 * u.MHz,
    'resonator_fit_parameters': [1.44750265e+06,  1.38440974e+00,  1.25601359e+00, -2.00360559e+08],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q2',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.454 * u.GHz,
        },
    'q_IF': -249* u.MHz,
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
        'LO_freq': 6.284 * u.GHz,
        },
    'r_IF': -201 * u.MHz,#200.25 * u.MHz,  # Hz
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
    'flux_name': 'f3',
    'r_name': 'r3',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': -0.625,  # 0.60 # defines static resting point of the qubit
    'YOKO_dc_offset': -0.625,  # -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.06, #0.2610775726010102, #0.010265151515151515, #0.12900378787878788,  # V
    'pi2_amp': 0.06,  # 0.2610775726010102, #0.010265151515151515, #0.12900378787878788,  # V
    'ef_amp': 0.4,
    'DRAG_beta': -0,
    'read_amp': 0.03,  # 08, # V
    'read_len': 10 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -200 * u.MHz,
    'resonator_fit_parameters': [1.44750265e+06, 1.38440974e+00, 1.25601359e+00, -2.00360559e+08],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO': {
        'LO_name': 'q3',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18,  # dBm
        'LO_freq': 4.45 * u.GHz,
    },
    'q_IF': -203 * u.MHz,  # .509
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
        'LO_freq': 6.401 * u.GHz,
    },
    'r_IF': -199.9 * u.MHz,  # Hz
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
    'flux_name': 'f4',
    'r_name': 'r4',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': -0.01,  # defines static resting point of the qubit
    'YOKO_dc_offset': -0.01,  # -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.068,  # 0.036739, # V
    'pi2_amp': 0.068,  # 0.036739, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.041249999999999995,  # V
    'read_len': 7 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -200 * u.MHz,
    'resonator_fit_parameters': [1.44750265e+06, 1.38440974e+00, 1.25601359e+00, -2.00360559e+08],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q4',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.46 * u.GHz,
        },
    'q_IF': -197.2 * u.MHz, #.509
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
        'LO_pow': 15,  # dBm
        'LO_freq': 6.585 * u.GHz,
        },
    'r_IF': -201 * u.MHz,  # Hz
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
    'flux_name': 'f5',
    'r_name': 'r5',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': 0,# (SAME AS q4) static resting point of the qubit
    'YOKO_dc_offset': -0.41,  # -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.042, # V
    'pi2_amp': 0.042, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.045, # V 11666666666666667, -199.83333333333334
    'read_len': 5 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 30,  # ns
    'gauss_len_X90': 4 * 15,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -199.83333333333334 * u.MHz,
    'resonator_fit_parameters': [1.44750265e+06, 1.38440974e+00, 1.25601359e+00, -2.00360559e+08],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q5',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.415 * u.GHz,
        },
    'q_IF': -200.8 * u.MHz,
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
        'LO_freq': 6.785 * u.GHz,
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
    'flux_name': 'f6',
    'r_name': 'r6',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': 0,  # defines static resting point of the qubit
    'YOKO_dc_offset': -0.41,  # -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.038,  # V
    'pi2_amp': 0.038,  # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp':  0.04, # V
    'read_len': 5 * u.us,
    'read_theta': 1.21522617081074,
    'gauss_len': 4 * 20,  # ns
    'gauss_len_X90': 4 * 10,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -200 * u.MHz,
    'resonator_fit_parameters': [-686588.9405022117,2.3460576700914486,-1.0599392996186952,-199514107.57629228],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.4,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q6',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq':  4.56 * u.GHz,
        },
    'q_IF': -203 * u.MHz, #-199.99141484848485 * u.MHz, #.509
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
    'r_LO': {
        'LO_name': 'r6',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 6.953 * u.GHz,
        },
    'r_IF': -198.6 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer': {
        'mixer_name': 'mr6',
        'mixer_off_I': 0.015,
        'mixer_off_Q': -0.0015,
        'mixer_gain': 0.01,
        'mixer_phase': 0.015,
        },
}
q1_sweet_spot = {
    # Global parameters #Q3
    'global_params': global_params,
    'element_name': 'q1',
    'q_name': 'q1',
    'flux_name': 'f1',
    'r_name': 'r1',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'flux_rf_I_channel': 3,
    'flux_rf_Q_chanel': 4,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 6,
    'out_channel': 1,
    'flux_dc_offset':  0.17,  # -0.1527, defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.024, # V
    'pi2_amp': 0.024, # V
    'ef_amp': 0.022,
    'DRAG_beta': -0,
    'read_amp': 0.03, # V
    'read_len': 8 * u.us,
    'read_theta':  0,
    'gauss_len': 4*20,  # ns
    'gauss_len_X90': 4*10,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -204*u.MHz,
    'resonator_fit_parameters': [1501719.5380980964,1.344585417924919,1.3452764370140642,-200422304.99269578],
    'cw_amp': 0.02,
    'cw_len': 100 * u.ns,
    'const_flux_len': 100*u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q1',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.6 * u.GHz,

        },
    'q_IF': -222 * u.MHz,
    'f_state_IF': (-198.372356974 - 204) * u.MHz,
    # Qubit mixer info
    'q_mixer': {
        'mixer_name': 'mq1',
        'mixer_off_I': -0.011010570492362606,
        'mixer_off_Q': -0.019294344994705167,
        'mixer_gain': -0.018072522373404353,
        'mixer_phase': -0.09508663669112141,
        },
    # Resonator frequency parameters
    'r_LO':{
        'LO_name': 'r1',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, # dBm
        'LO_freq': 6.141 * u.GHz,
        },
    'r_IF': -200.5 * u.MHz, # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr1',
        'mixer_off_I': -0.0045331554370932284,
        'mixer_off_Q': -0.00020641707233153245,
        'mixer_gain': 0.0337099552154541,
        'mixer_phase': -0.005480670928955078,
        },
}

q3_sweet_spot = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q3',
    'q_name': 'q3',
    'flux_name': 'f3',
    'r_name': 'r3',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': 0.60,  # 0.60 # defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.051, #0.2610775726010102, #0.010265151515151515, #0.12900378787878788,  # V
    'pi2_amp': 0.051,  # 0.2610775726010102, #0.010265151515151515, #0.12900378787878788,  # V
    'ef_amp': 0.4,
    'DRAG_beta': -0,
    'read_amp': 0.03,  # 08, # V
    'read_len': 10 * u.us,
    'read_theta': 0,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -200 * u.MHz,
    'resonator_fit_parameters': [1.44750265e+06, 1.38440974e+00, 1.25601359e+00, -2.00360559e+08],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO': {
        'LO_name': 'q3',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18,  # dBm
        'LO_freq': 4.460 * u.GHz,
    },
    'q_IF': -197 * u.MHz,  # .509
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
        'LO_freq': 6.401 * u.GHz,
    },
    'r_IF': -201.6 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer': {
        'mixer_name': 'mr3',
        'mixer_off_I': -0.0020825495827011763,
        'mixer_off_Q': -0.0017681648721918468,
        'mixer_gain': 0.0,
        'mixer_phase': 0.0,
    },
}
q4_sweet_spot = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q4',
    'q_name': 'q4',
    'flux_name': 'f4',
    'r_name': 'r4',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': -0.19,  # defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.1306270202020202,  # 0.036739, # V
    'pi2_amp': 0.1306270202020202,  # 0.036739, # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp': 0.023,  # V
    'read_len': 7 * u.us,
    'read_theta':  0,
    'gauss_len': 4 * 20,  # ns
    'gauss_len_X90': 4 * 10,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -200 * u.MHz,
    'resonator_fit_parameters': [1.44750265e+06, 1.38440974e+00, 1.25601359e+00, -2.00360559e+08],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q4',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.480 * u.GHz,
        },
    'q_IF': -198.88 * u.MHz, #.509
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
        'LO_pow': 15,  # dBm
        'LO_freq': 6.585 * u.GHz,
        },
    'r_IF': -200.8 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer':{
        'mixer_name': 'mr4',
        'mixer_off_I': -0.00343116857111454,
        'mixer_off_Q': -0.001192570477724075,
        'mixer_gain': 0.0,
        'mixer_phase': 0.0,
        },
}

q6_sweet_spot = {
    # Global parameters #q5
    'global_params': global_params,
    'element_name': 'q6',
    'q_name': 'q6',
    'flux_name': 'f6',
    'r_name': 'r6',
    'q_I_channel': 1,
    'q_Q_channel': 2,
    'r_I_channel': 9,
    'r_Q_channel': 10,
    'flux_channel': 4,
    'out_channel': 1,
    'flux_dc_offset': -0.21,  # defines static resting point of the qubit
    'T_1': 1000 * u.us,
    'T_2R': 100 * u.us,
    'T_2E': 100 * u.us,
    'pi_amp': 0.016923737373737376,  # V
    'pi2_amp': 0.016923737373737376,  # V
    'ef_amp': 0.3,
    'DRAG_beta': -0,
    'read_amp':  0.04, # V
    'read_len': 5 * u.us,
    'read_theta': 1.21522617081074,
    'gauss_len': 4 * 40,  # ns
    'gauss_len_X90': 4 * 20,  # ns
    'reset_time': int(reset_time),
    'resonator_dump_time': resonator_dump_time,
    'flux_settle_time': flux_settle_time,
    'anharmonicity': -200 * u.MHz,
    'resonator_fit_parameters': [-686588.9405022117,2.3460576700914486,-1.0599392996186952,-199514107.57629228],
    'cw_amp': 0.02,
    'cw_len': 500 * u.ns,
    'const_flux_len': 100 * u.ns,
    'const_flux_amp': 0.45,
    # Qubit frequency parameters
    'q_LO':{
        'LO_name': 'q6',
        'LO_IP': '192.168.1.102',
        'LO_pow': 18, #dBm
        'LO_freq': 4.575 * u.GHz,
        },
    'q_IF': -203.18 * u.MHz, #-199.99141484848485 * u.MHz, #.509
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
    'r_LO': {
        'LO_name': 'r6',
        'LO_IP': '192.168.1.178',
        'LO_mod': 31,
        'LO_pow': 15, #dBm
        'LO_freq': 6.953 * u.GHz,
        },
    'r_IF': -198.8 * u.MHz,  # Hz
    # Resonator mixer info
    'r_mixer': {
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