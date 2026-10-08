from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
import matplotlib.pyplot as plt
from Drivers.SAN9030A import SAN9030A
from PythonDrivers.control_8257 import set_LO_parameters
from Config.config_helpers import meta_dict_to_config_2Q
import time
import pyvisa as visa
import datetime
import pickle

### Accessing files and equipment
datetimenow = datetime.datetime.now()
datestring = datetimenow.strftime("%Y_%m_%d")
outer_folder = 'Z://t1Team/Data/2024-12-22_cooldown/TATQ01_Topsil_01/OPX/'
pathPickle = outer_folder + 'meta_dict/meta_dict.pkl'
VISAaddress = 'TCPIP0::192.168.1.140::inst0::INSTR' # Spectrum analyzer visa address - add spec analyzer - check for convergence and update metadict
#########################

with open(pathPickle, 'rb') as f:
    meta_dict = pickle.load(f)

### Experiment specific parameters gathering: Update before running ###
qubit = meta_dict['qc']
element = 'qc'
element_prefix = element[0]
mixer = qubit[element_prefix+'_mixer']
mixer_name = mixer['mixer_name']
startIndex = 0
lastIndex =6

with program() as ssb_calib:
    with infinite_loop_():
        play("ps_op", element) # "pt_op" is for resonator, "iswap_const" for coupler
        # play("ps_op", element)
        # play("iSWAP_const", element)

LOs = [qubit[element_prefix + '_LO']]

# set_LO_parameters(LOs)

config = meta_dict_to_config_2Q(meta_dict)
###



#############
#############


def get_amp():
    time.sleep(1)
    data = sa.get_trace()
    # print(data)
    # print(max(data))
    max_signal_power = max(data)
    return max_signal_power


def leakageMap(span_indx, OffIm, OffQm):
    # Wide
    # offI = np.arange(-0.29, 0.3, 0.04)
    # offQ = np.arange(-0.29, 0.3, 0.04)

    # Standard #1
    # offI = np.linspace(-0.01, 0.01, 10)
    # offQ = np.linspace(-0.01, 0.01, 10)

    IQ_span = [0.1,0.01,0.005,0.001,0.0005,0.0001,0.00005]

    offI = np.linspace(OffIm-IQ_span[span_indx], OffIm+IQ_span[span_indx], 10)
    offQ = np.linspace(OffQm-IQ_span[span_indx], OffQm+IQ_span[span_indx], 10)

    print(offI, offQ)
    amps = []

    for i in offI:
        for j in offQ:
            qm.set_output_dc_offset_by_element(element, "I", i)
            qm.set_output_dc_offset_by_element(element, "Q", j)
            amp_ = get_amp()
            amps.append(amp_)
            print("amp = {}, offI = {}, offQ = {}".format(amp_, i, j))

    return offI, offQ, amps

def gamma(prev_x, prev_y, curr_x, curr_y, prev_grad_x, prev_grad_y, curr_grad_x, curr_grad_y):
    df = np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])
    d_grad = np.array([curr_grad_x, curr_grad_y]) - np.array([prev_grad_x, prev_grad_y])
    norm = np.sum(d_grad * d_grad)
    g = np.abs(np.sum(df * d_grad) / norm)

    return g


def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]

def imbalancesMap(span_indx, gain_m, phase_m):
    # Wide
    # phi = np.linspace(-0.11, -0.9, 10)
    # g = np.linspace(-0.01, 0.0, 10)

    # Standard
    # phi = np.linspace(-0.01, 0.01, 10)
    # g = np.linspace(-0.01, 0.01, 10)

    PG_span = [0.2,0.05,0.01,0.005,0.001,0.0005]

    phi = np.linspace(phase_m - PG_span[span_indx], phase_m + PG_span[span_indx], 10)
    g = np.linspace(gain_m - PG_span[span_indx], gain_m + PG_span[span_indx], 10)

    print(phi, g)
    amps = []

    for i in phi:

        for j in g:
            qm.set_mixer_correction(mixer_name, int(IF), int(LO), IQ_imbalance_corr(j, i))
            amp_ = get_amp()
            amps.append(amp_)
            print("amp = {}, phase = {}, gain = {}".format(amp_, i, j))

    return phi, g, amps


def gradImbalances(phase_, gain_):
    eps = 0.0002
    qm.set_mixer_correction(mixer_name, int(IF), int(LO), IQ_imbalance_corr(gain_, phase_ + eps))
    a1 = get_amp()
    qm.set_mixer_correction(mixer_name, int(IF), int(LO), IQ_imbalance_corr(gain_, phase_ - eps))
    a2 = get_amp()
    qm.set_mixer_correction(mixer_name, int(IF), int(LO), IQ_imbalance_corr(gain_ + eps, phase_ + eps))
    a3 = get_amp()
    qm.set_mixer_correction(mixer_name, int(IF), int(LO), IQ_imbalance_corr(gain_ - eps, phase_ + eps))
    a4 = get_amp()

    gradx = -(a2 - a1) / (2 * eps)
    grady = -(a4 - a3) / (2 * eps)

    return gradx, grady


for i in range(startIndex,lastIndex+1):
    ##########
    # Execute:
    ##########

    OffIm = mixer['mixer_off_I']
    OffQm = mixer['mixer_off_Q']
    phase_m = mixer['mixer_phase']
    gain_m = mixer['mixer_gain']
    IF = qubit[element_prefix+'_IF']
    LO = qubit[element_prefix+'_LO']['LO_freq']

    # Execute the mixer_cal program:
    qmm = QuantumMachinesManager(host='192.168.1.127', port='80')
    qm = qmm.open_qm(config)

    job = qm.execute(ssb_calib)

    # Configure the SA :
    rm = visa.ResourceManager()
    sa = SAN9030A(VISAaddress, rm)
    print("Successfully connected to N9030A.")
    datetimenow = datetime.datetime.now()
    datetimestring = datetimenow.strftime("%Y_%m_%d_%H_%M_%S")
    IF_MHz = IF / 1e6
    LO_MHz = LO / 1e6

    delta_F = 20  # MHz
    span = 2 * delta_F
    sa.set_freq_span_MHz(freq=LO_MHz, band=span)
    time.sleep(1)

    # LO leakage 2D map
    offI, offQ, amps = leakageMap(i, OffIm, OffQm)
    plt.figure()
    offI_grid, offQ_grid = np.meshgrid(offI, offQ)
    amps_grid = np.transpose(np.reshape(amps, [len(offI), len(offQ)]))
    plt.pcolormesh(offI_grid, offQ_grid, amps_grid)
    plt.colorbar()
    plt.xlabel("I offset")
    plt.ylabel("Q offset")
    plt.savefig(outer_folder + datetimestring + "_iter_" + str(i) + "_offset.png", bbox_inches='tight')
    plt.close()

    # Finding minimum leakage point
    indices = np.where(amps_grid == np.min(amps_grid))
    offI0 = offI_grid[indices[0][0]][indices[1][0]]
    offQ0 = offQ_grid[indices[0][0]][indices[1][0]]
    print("OffIm = " + str(offI0))
    print("OffQm = " + str(offQ0))

    # Configure the SA:
    LSB = LO_MHz - IF_MHz
    USB = LO_MHz + IF_MHz
    delta_F = 20  # MHz
    span = 2 * delta_F
    sa.set_freq_span_MHz(freq=LSB, band=span)
    time.sleep(1)

    # IQ imbalances 2D map
    phase, gain, amps = imbalancesMap(i, gain_m, phase_m)
    # phase, gain, amps = imbalancesMap_old()

    plt.figure()
    phase_grid, gain_grid = np.meshgrid(phase, gain)
    amps_grid = np.transpose(np.reshape(amps, [len(phase), len(gain)]))
    plt.pcolormesh(phase_grid, gain_grid, amps_grid)
    plt.colorbar()
    plt.xlabel("phase")
    plt.ylabel("gain")
    plt.xticks(rotation=45)
    plt.savefig(outer_folder + datetimestring + "_iter_" + str(i) + "_phase.png", bbox_inches='tight')
    plt.close()

    # Finding minimum imbalance point
    indices = np.where(amps_grid == np.min(amps_grid))
    phase0 = phase_grid[indices[0][0]][indices[1][0]]
    gain0 = gain_grid[indices[0][0]][indices[1][0]]
    print("phase_m = " + str(phase0))
    print("gain_m = " + str(gain0))

    # offI0=0; offQ0=0; phase0=0; gain0=0;
    # with open(pathPickle, 'wb') as f:
    #     cfgstring = element + "cfg"

    meta_dict[qubit['element_name']][element_prefix + '_mixer']['mixer_off_I'] = offI0
    meta_dict[qubit['element_name']][element_prefix + '_mixer']['mixer_off_Q'] = offQ0
    meta_dict[qubit['element_name']][element_prefix + '_mixer']['mixer_phase'] = phase0
    meta_dict[qubit['element_name']][element_prefix + '_mixer']['mixer_gain'] = gain0

    #     pickle.dump(metaDict,f)
    config = meta_dict_to_config_2Q(meta_dict)
