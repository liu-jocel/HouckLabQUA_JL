import numpy as np
from qm.qua import *

def time_vec_to_clk_vec_wait(t_vec_raw):

    #Takes in an arbitrary time array and converts it to clock ticks

    assert t_vec_raw[0]>=16e-9,'You must pick a starting time larger than four clock ticks for the wait() command.'

    clk_vec = np.ceil(t_vec_raw / 4e-9).astype(int)  # Convert to a clock tick array consistent with the wait() command in qua
    return clk_vec


def measure_1q(element, I, Q):

    measure('read', element['r_name'], None, demod.full('integ_W1_' + element['r_name'], I, 'out'), demod.full('integ_W2_' + element['r_name'], Q, 'out'))
    return I, Q

def measure_1q_amp(element, amplitude, I, Q):
    measure('read'*amp(amplitude), element['r_name'], None, demod.full('integ_W1_' + element['r_name'], I, 'out'), demod.full('integ_W2_' + element['r_name'], Q, 'out'))
    return I, Q

def measure_2q(element_1, element_2, I1, Q1, I2, Q2):

    measure('read', element_1['r_name'], None, demod.full('integ_W1_' + element_1['r_name'], I1, 'out'),
            demod.full('integ_W2_' + element_1['r_name'], Q1, 'out'))
    # align()
    measure('read', element_2['r_name'], None, demod.full('integ_W1_' + element_2['r_name'], I2, 'out'),
            demod.full('integ_W2_' + element_2['r_name'], Q2, 'out'))
    return I1, Q1, I2, Q2

def measure_3q(element_1, element_2, element_c, I1, Q1, I2, Q2, Ic, Qc):

    measure('read', element_1['r_name'], None, demod.full('integ_W1_' + element_1['r_name'], I1, 'out'),
            demod.full('integ_W2_' + element_1['r_name'], Q1, 'out'))
    # align()
    measure('read', element_2['r_name'], None, demod.full('integ_W1_' + element_2['r_name'], I2, 'out'),
            demod.full('integ_W2_' + element_2['r_name'], Q2, 'out'))
    # align()
    measure('read', element_c['r_name'], None, demod.full('integ_W1_' + element_c['r_name'], Ic, 'out'),
            demod.full('integ_W2_' + element_c['r_name'], Qc, 'out'))

    return I1, Q1, I2, Q2, Ic, Qc
