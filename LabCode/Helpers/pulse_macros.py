from qm.qua import *
import numpy as np
def play_pulse(element, pulse, phase):
    play(pulse, element)
def play_pulse_duration(element, pulse, duration, phase):
    play(pulse, element, duration=duration)
def play_X90(element, phase):
    play("X90", element)

def play_mX90(element, phase):
    play("mX90", element)


def play_Y90(element, phase):
    play("Y90", element)

def play_mY90(element, phase):
    play("mY90", element)

def drag_cosine_pulse_waveforms(amplitude, length, alpha, anharmonicity, detuning=0.0, sampling_rate=1e9, **kwargs):
    """
    Creates Cosine based DRAG waveforms that compensate for the leakage and for the AC stark shift.
    modified from qua waveform tools
    These DRAG waveforms has been implemented following the next Refs.:
    Chen et al. PRL, 116, 020501 (2016)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501
    and Chen's thesis
    https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf

    :param float amplitude: The amplitude in volts.
    :param int length: The pulse length in ns.
    :param float alpha: The DRAG coefficient.
    :param float anharmonicity: f_21 - f_10 - The differences in energy between the 2-1 and the 1-0 energy levels, in Hz.
    :param float detuning: The frequency shift to correct for AC stark shift, in Hz.
    :param float sampling_rate: The sampling rate used to describe the waveform, in samples/s. Default is 1G samples/s.
    :return: Returns a tuple of two lists. The first list is the 'I' waveform (real part) and the second is the
        'Q' waveform (imaginary part)

    Use: define bare IF of qubit and permutate detuning off of it.
    """
    if alpha != 0 and anharmonicity == 0:
        raise Exception("Cannot create a DRAG pulse with `anharmonicity=0`")
    t = np.arange(length, step=1e9 / sampling_rate)  # An array of size pulse length in ns
    end_point = length - 1e9 / sampling_rate
    cos_wave = 0.5 * amplitude * (1 - np.cos(t * 2 * np.pi / end_point))  # The cosine function
    sin_wave = (
        0.5 * amplitude * (2 * np.pi / end_point * 1e9) * np.sin(t * 2 * np.pi / end_point)
    )  # The derivative of cosine function
    z = cos_wave + 1j * 0
    if alpha != 0:
        # The complex DRAG envelope:
        z += 1j * sin_wave * (alpha / (2 * np.pi * anharmonicity - 2 * np.pi * detuning))
        # The complex detuned DRAG envelope:
    # Frequency Modulaiton of the envelope (not used)
    z *= np.exp(-1j * 2 * np.pi * detuning * (t - length/2) * 1e-9)

    I_wf = z.real.tolist()  # The `I` component is the real part of the waveform
    Q_wf = z.imag.tolist()  # The `Q` component is the imaginary part of the waveform
    return I_wf, Q_wf