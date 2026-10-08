import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


def Fx(t, px, py, pz, toff, b):
    Omega = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    tp = t + toff
    return Omega ** (-2) * (-pz * px + px * pz * np.cos(Omega * tp) + Omega * py * np.sin(Omega * tp)) + b

def Fy(t, px, py, pz, toff, b):
    Omega = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    tp = t + toff
    return Omega ** (-2) * (pz * py - pz * py * np.cos(Omega * tp) - Omega * px * np.sin(Omega * tp)) + b

def Fz(t, px, py, pz, toff, b):
    Omega = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    tp = t + toff
    return Omega ** (-2) * (pz ** 2 + (px ** 2 + py ** 2) * np.cos(Omega * tp)) + b

def combined_func(t, px, py, pz, toff, b):
    Fx_vals = Fx(t, px, py, pz,toff, b)
    Fy_vals = Fy(t, px, py, pz,toff, b)
    Fz_vals = Fz(t, px, py, pz,toff, b)
    return np.concatenate([Fx_vals, Fy_vals, Fz_vals])


def find_first_minima_idx(y_data, prominence=0):
    minima_indices, _ = find_peaks(-y_data, prominence=prominence)
    return minima_indices[0]

def calculate_Omega_MHz(fit, pcov):
    Omega = np.sqrt(fit[0]**2 + fit[1]**2 + fit[2]**2) / (2 * np.pi) * 1e3
    Omega_error = np.sqrt(pcov[0][0] + pcov[1][1] + pcov[2][2]) / (2 * np.pi) * 1e3
    return Omega, Omega_error

def calculate_interaction_rates(ground_fit, ground_fit_error, excited_fit, excited_fit_error):
    """
    Calculate the following paramaters for the effective CR hamiltonian:
    IX, IY, IZ, ZX, ZY, ZZ in units MHZ/(2 * pi)
    """
    Omega_X_0 = 4*(ground_fit[0]) / (2 * np.pi) * 1e3
    Omega_Y_0 = 4*(ground_fit[1]) / (2 * np.pi) * 1e3
    Delta_0 = 4*(ground_fit[2]) / (2 * np.pi) * 1e3

    Omega_X_1 = 4*(excited_fit[0]) / (2 * np.pi) * 1e3
    Omega_Y_1 = 4*(excited_fit[1]) / (2 * np.pi) * 1e3
    Delta_1 = 4*(excited_fit[2]) / (2 * np.pi) * 1e3

    Omega_X_0_err = ground_fit_error[0][0]
    Omega_Y_0_err = ground_fit_error[1][1]
    Delta_0_err = ground_fit_error[2][2]

    Omega_X_1_err = excited_fit_error[0][0]
    Omega_Y_1_err = excited_fit_error[1][1]
    Delta_1_err = excited_fit_error[2][2]

    IX = 0.5 * (Omega_X_0 + Omega_X_1)
    IY = 0.5 * (Omega_Y_0 + Omega_Y_1)
    IZ = 0.5 * (Delta_0 + Delta_1)
    ZX = 0.5 * (Omega_X_0 - Omega_X_1)
    ZY = 0.5 * (Omega_Y_0 - Omega_Y_1)
    ZZ = 0.5 * (Delta_0 - Delta_1)

    IX_err = 0.5 * np.sqrt(Omega_X_0_err + Omega_X_1_err) * 4 / (2 * np.pi) * 1e3
    IY_err = 0.5 * np.sqrt(Omega_Y_0_err + Omega_Y_1_err) * 4 / (2 * np.pi) * 1e3
    IZ_err = 0.5 * np.sqrt(Delta_0_err + Delta_1_err) * 4 / (2 * np.pi) * 1e3
    ZX_err = 0.5 * np.sqrt(Omega_X_0_err + Omega_X_1_err) * 4 / (2 * np.pi) * 1e3
    ZY_err = 0.5 * np.sqrt(Omega_Y_0_err + Omega_Y_1_err) * 4 / (2 * np.pi) * 1e3
    ZZ_err = 0.5 * np.sqrt(Delta_0_err + Delta_1_err) * 4 / (2 * np.pi) * 1e3
    rates = {
        "IX": (IX, IX_err),
        "IY": (IY, IY_err),
        "IZ": (IZ, IZ_err),
        "ZX": (ZX, ZX_err),
        "ZY": (ZY, ZY_err),
        "ZZ": (ZZ, ZZ_err),
    }
    return rates

def fft_expectation_values(t_vec, data):
    N = len(data)
    dt = t_vec[1]-t_vec[0]
    freqs = fftfreq(N, d=dt)
    signal_zero_mean = data - np.mean(data)
    window = np.hanning(N)

    fft_result = fft(signal_zero_mean*window)
    fft_mag = np.abs(fft_result[:N//2])
    max_fft_mag = np.argmax(fft_mag)
    print(np.max(fft_mag))
    if np.max(fft_mag) > 1:
        return freqs[max_fft_mag]
    else:
        return 0