import numpy as np
from qualang_tools.units import unit
from lmfit import Model
from lmfit.models import LinearModel
u = unit(coerce_to_integer=True)


def exponential(x, a, b, c, d):
    return a * np.exp(- (x - b) / c) + d

def exponential_1(x, a, b, c):
    return a * np.exp(-x/b) + c

def exponential_2(x, n, a, b, c,):
    return a*np.exp(-1 * (x / np.abs(b)) ** n) + c

def decaying_cosine(x, a, b, c, d, e):
    return a * np.exp(-x / e) * np.cos(2. * np.pi * b * x + c * np.pi / 180.) + d


def cosine(x, a, b, c, d):
    return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d

def iSWAP_phase(beta, a, phi, off):
    # https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033447
    return a * np.cos(2 * (beta + phi)) + off


def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    res = a1 * np.exp(- (x - mu1)**2 / (2 * sigma1**2))\
          + a2 * np.exp(- (x - mu2)**2 / (2 * sigma2**2))
    return res


def iSWAP_phase_seed(data):

    probs = data[1]

    a_guess = (np.max(probs) - np.min(probs)) / 2
    phi_guess = 0
    off_guess = np.mean(probs)

    return [a_guess,phi_guess, off_guess]


def phase_seed(data):

    probs = data[1]

    a_guess = (np.max(probs) - np.min(probs)) / 2
    b_guess = 1
    c_guess = 0
    d_guess = np.mean(probs)

    return [a_guess, b_guess, c_guess, d_guess]


def rabi_seed(x, y):
    a_guess = (np.max(y) - np.min(y)) / 2
    b_guess = 1 / (0.5*x[-1])
    c_guess = 0
    d_guess = np.mean(y)
    e_guess = 5*x[-1]

    return [a_guess, b_guess, c_guess, d_guess, e_guess]


def T1_seed(t_vec, data):

    # assuming log10 spaced times
    # ratio R of logs of subsequent points should be constant
    # 10^R is estimate of parameter c in exponential()

    times = t_vec
    probs = data
    num_times = len(times)

    a_guess = np.max(probs)-np.min(probs)
    b_guess = 0
    c_guess = times[-1]/10
    d_guess = np.min(probs)

    return [a_guess, b_guess, c_guess, d_guess]


def T2E_seed(t_vec, data):
    # assuming log10 spaced times
    # ratio R of logs of subsequent points should be constant
    # 10^R is estimate of parameter c in exponential()

    times = t_vec
    probs = data
    num_times = len(times)

    a_guess = np.max(probs) - np.min(probs)
    b_guess = 0
    c_guess = times[-1] / 10
    d_guess = np.min(probs)

    return [a_guess, b_guess, c_guess, d_guess]

def T1_seed_1(data):

    # assuming log10 spaced times
    # ratio R of logs of subsequent points should be constant
    # 10^R is estimate of parameter c in exponential()

    times = data[0]
    probs = data[1]
    num_times = len(times)


    a_guess = np.max(probs)
    b_guess = 300e-6
    c_guess = np.min(probs)

    return [a_guess, b_guess, c_guess]


# def T2E_seed(t_vec, data):
#     a_guess = np.round(np.max(data)-0.5, 4)
#     b_guess = 0
#     # c_guess = 10**np.mean(log_rats)
#     c_guess = np.round(t_vec[-1]/10, 4)
#     d_guess = 0.5
#
#     return [a_guess, b_guess, c_guess, d_guess]

def exponential_2(x, n, a, b, c,):
    return a*(1-np.exp(-1*(x/b)**n)) + c
def exponential_2_forced_origin(x, n, a, b,):
    return a*(1-np.exp(-1*(x/b)**n))

def T2CPMG_seed(times, probs):
    n_guess = 2
    a_guess = 0.5
    b_guess = 100
    c_guess = 0
    return [n_guess, a_guess, b_guess, c_guess]
def test_line(x,a,b):
    return a*x + b


def test_seed(data,detuning):
    print(detuning)
    return [1,4]


def T2R_seed(data, detuning):

    times = data['t_vec']
    probs = data['ss_data']

    a_guess = (np.max(probs) - np.min(probs))/2
    b_guess = detuning
    c_guess = 0
    d_guess = np.mean(probs)
    e_guess = times[-1] / 4

    return [a_guess, b_guess, c_guess, d_guess, e_guess]


def freq_cal_seed(data, detuning):

    # assuming linear spaced times
    # return detuning

    times = data[0]
    probs = data[1]
    num_times = len(times)

    a_guess = (np.max(probs) - np.min(probs)) / 2
    b_guess = detuning
    c_guess = 0
    d_guess = np.mean(probs)

    return [a_guess, b_guess, c_guess, d_guess]

def RPM_seed(x, y):
    """ """
    fs = np.fft.rfftfreq(len(x), x[1] - x[0])
    ofs = np.mean(y)
    fft = np.fft.rfft(y - ofs)
    idx = np.argmax(abs(fft))
    return [np.std(y-ofs), fs[idx], np.angle(fft[idx]), ofs]


def sine(x, amp, f0, phi, ofs):
    """ """
    return ofs + amp * np.sin(2 * np.pi * f0 * x + phi)