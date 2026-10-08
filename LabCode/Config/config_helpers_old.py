import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
from mergedeep import merge
from qualang_tools.config.waveform_tools import (
    drag_cosine_pulse_waveforms,
    drag_gaussian_pulse_waveforms,
)

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def make_qubit_res_config(q):
    """
    Config File. X90 is chosen as the base value for the config IF.
    """
    # make elements
    envelope_type = q.get("envelope_type", "COSINE")
    controllers = {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                q["q_I_channel"]: {"offset": q["q_mixer"]["mixer_off_I"]},
                q["q_Q_channel"]: {"offset": q["q_mixer"]["mixer_off_Q"]},
                q["r_I_channel"]: {"offset": q["r_mixer"]["mixer_off_I"]},
                q["r_Q_channel"]: {"offset": q["r_mixer"]["mixer_off_Q"]},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {},
                8: {},
                9: {},
                10: {},
            },
            "analog_inputs": {
                1: {"offset": 0.119977},
                # 2: {"offset": 0.05, }
            },
        }
    }

    elements = {
        q["q_name"]: {
            "mixInputs": {
                "I": ("con1", q["q_I_channel"]),
                "Q": ("con1", q["q_Q_channel"]),
                "lo_frequency": q["q_LO"]["LO_freq"],
                "mixer": q["q_mixer"]["mixer_name"],
            },
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", q["q_I_channel"]),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": q["IF_freq"] + q["detuning"],
            "operations": {
                "constant_tone": "ps_pulse_" + q["q_name"],
                "X180_ef": "X180_ef_pulse_" + q["q_name"],
                "X180": "X180_pulse_" + q["q_name"],
                "mX180": "Xm180_pulse_" + q["q_name"],
                "Y180": "Y180_pulse_" + q["q_name"],
                "mY180": "Ym180_pulse_" + q["q_name"],
                "X90": "X90_pulse_" + q["q_name"],
                "mX90": "Xm90_pulse_" + q["q_name"],
                "Y90": "Y90_pulse_" + q["q_name"],
                "mY90": "Ym90_pulse_" + q["q_name"],
                "Id": "Id_pulse",
                "Zero": "zero_pulse",
            },
        },
        q["r_name"]: {
            "mixInputs": {
                "I": ("con1", q["r_I_channel"]),
                "Q": ("con1", q["r_Q_channel"]),
                "lo_frequency": q["r_LO"]["LO_freq"],
                "mixer": q["r_mixer"]["mixer_name"],
            },
            "intermediate_frequency": q["r_IF"],
            "operations": {
                "constant_tone": "pt_pulse_" + q["r_name"],
                "read": "read_pulse_" + q["r_name"],
            },
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", q["r_I_channel"]),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out": ("con1", q["out_channel"]),
            },
            "time_of_flight": 368,  # ns, must be multiple of 4
            "smearing": 0,
        },
    }
    pulses = {
        "pt_pulse_" + q["r_name"]: {
            "operation": "control",
            "length": 40000,
            "waveforms": {
                "I": "pt_wf_" + q["r_name"],
                "Q": "zero_wf",
            },
        },
        "ps_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["cw_len"],
            "waveforms": {
                "I": "ps_wf_" + q["q_name"],
                "Q": "zero_wf",
            },
        },
        "zero_pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {"I": "zero_wf", "Q": "zero_wf"},
        },
        "X90_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["X90"]["length"],
            "waveforms": {
                "I": "X90_wf_" + q["q_name"],
                "Q": "X90_wf_der_" + q["q_name"],  #'zero_wf'
            },
            "digital_marker": "trig",
        },
        "Xm90_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["X90"]["length"],
            "waveforms": {
                "I": "Xm90_wf_" + q["q_name"],
                "Q": "Xm90_wf_der_"
                + q["q_name"],  # 'zero_wf' # the drive phase changes with I and Q
            },
            "digital_marker": "trig",
        },
        "X180_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["X180"]["length"],
            "waveforms": {
                "I": "X180_wf_" + q["q_name"],
                "Q": "X180_wf_der_" + q["q_name"],
            },
            "digital_marker": "trig",
        },
        "Xm180_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["X180"]["length"],
            "waveforms": {
                "I": "Xm180_wf_" + q["q_name"],
                "Q": "Xm180_wf_der_" + q["q_name"],
            },
            "digital_marker": "trig",
        },
        "X180_ef_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["ef_len"],
            "waveforms": {
                "I": "ef180_wf_" + q["q_name"],
                "Q": "ef180_wf_der_" + q["q_name"],  # 'zero_wf'
            },
            "digital_marker": "trig",
        },
        "Y90_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["Y90"]["length"],
            "waveforms": {
                "I": "Y90_wf_der_" + q["q_name"],
                "Q": "Y90_wf_" + q["q_name"],
            },
            "digital_marker": "trig",
        },
        "Y180_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["Y180"]["length"],
            "waveforms": {
                "I": "Y180_wf_der_" + q["q_name"],
                "Q": "Y180_wf_" + q["q_name"],
            },
            "digital_marker": "trig",
        },
        "Ym180_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["Y180"]["length"],
            "waveforms": {
                "I": "Ym180_wf_der_" + q["q_name"],
                "Q": "Ym180_wf_" + q["q_name"],
            },
            "digital_marker": "trig",
        },
        "Ym90_pulse_" + q["q_name"]: {
            "operation": "control",
            "length": q["Y90"]["length"],
            "waveforms": {
                "I": "Ym90_wf_der_" + q["q_name"],
                "Q": "Ym90_wf_" + q["q_name"],
            },
            "digital_marker": "trig",
        },
        "Id_pulse": {
            "operation": "control",
            "length": q["X90"]["length"],
            "waveforms": {"I": "zero_wf", "Q": "zero_wf"},
        },
        "read_pulse_" + q["r_name"]: {
            "operation": "measurement",
            "length": q["read_len"],
            "waveforms": {
                "I": "read_wf_" + q["r_name"],
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integ_W1_" + q["r_name"]: "integ_W1_" + q["r_name"],
                "integ_W2_" + q["r_name"]: "integ_W2_" + q["r_name"],
            },
            "digital_marker": "ON",
        },
    }
    if envelope_type.upper() == "COSINE":
        waveforms = {
            "pt_wf_" + q["r_name"]: {"type": "constant", "sample": 0.45},
            "ps_wf_" + q["q_name"]: {"type": "constant", "sample": q["cw_amp"]},
            "read_wf_" + q["r_name"]: {"type": "constant", "sample": q["read_amp"]},
            "zero_wf": {"type": "constant", "sample": 0.0},
            "X90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": drag_cosine_pulse_waveforms(
                    q["X90"]["amp"],
                    q["X90"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    detuning=q["detuning"],
                )[0],
            },
            "X90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        q["X90"]["amp"],
                        q["X90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[1],
            },
            "X180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": drag_cosine_pulse_waveforms(
                    q["X180"]["amp"],
                    q["X180"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    detuning=q["detuning"],
                )[0],
            },
            "X180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        q["X180"]["amp"],
                        q["X180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[1],
            },
            "Xm90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        -q["X90"]["amp"],
                        q["X90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[0],
            },
            "Xm90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        -q["X90"]["amp"],
                        q["X90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )[1]
                ),
            },
            "Xm180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    drag_cosine_pulse_waveforms(
                        q["X180"]["amp"],
                        q["X180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )[0]
                ),
            },
            "Xm180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    drag_cosine_pulse_waveforms(
                        q["X180"]["amp"],
                        q["X180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[1],
            },
            "Y90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": drag_cosine_pulse_waveforms(
                    q["Y90"]["amp"],
                    q["Y90"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    detuning=q["detuning"],
                )[0],
            },
            "Y90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    drag_cosine_pulse_waveforms(
                        q["Y90"]["amp"],
                        q["Y90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[1],
            },
            "Y180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": drag_cosine_pulse_waveforms(
                    q["Y180"]["amp"],
                    q["Y180"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    detuning=q["detuning"],
                )[0],
            },
            "Y180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    drag_cosine_pulse_waveforms(
                        q["Y180"]["amp"],
                        q["Y180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[1],
            },
            "Ym90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        -q["Y90"]["amp"],
                        q["Y90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[0],
            },
            "Ym90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    drag_cosine_pulse_waveforms(
                        -q["Y90"]["amp"],
                        q["Y90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )[1]
                ),
            },
            "Ym180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        -q["Y180"]["amp"],
                        q["Y180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )[0]
                ),
            },
            "Ym180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        q["Y180"]["amp"],
                        q["Y180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        detuning=q["detuning"],
                    )
                )[1],
            },
            "ef180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": drag_cosine_pulse_waveforms(
                    q["ef_amp"], q["ef_len"], 0, q["anharmonicity"], detuning=0
                )[0],
            },
            "ef180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    drag_cosine_pulse_waveforms(
                        -q["ef_amp"], q["ef_len"], 0, q["anharmonicity"], detuning=0
                    )[1]
                ),
            },
        }
    if envelope_type.upper() == "GAUSSIAN":
        sigma = q.get("sigma", None)

        def _gauss(amp, length, beta, anharm, det):
            s = sigma if sigma is not None else length / 5
            return drag_gaussian_pulse_waveforms(
                amp, length, s, beta, anharm, detuning=det
            )

        waveforms = {
            "pt_wf_" + q["r_name"]: {"type": "constant", "sample": 0.45},
            "ps_wf_" + q["q_name"]: {"type": "constant", "sample": q["cw_amp"]},
            "read_wf_" + q["r_name"]: {"type": "constant", "sample": q["read_amp"]},
            "zero_wf": {"type": "constant", "sample": 0.0},
            "X90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": _gauss(
                    q["X90"]["amp"],
                    q["X90"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    q["detuning"],
                )[0],
            },
            "X180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": _gauss(
                    q["X180"]["amp"],
                    q["X180"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    q["detuning"],
                )[0],
            },
            "Xm180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["X180"]["amp"],
                        q["X180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[0]
                ),
            },
            "Xm90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["X90"]["amp"],
                        q["X90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[0]
                ),
            },
            "X90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["X90"]["amp"],
                        q["X90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "X180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["X180"]["amp"],
                        q["X180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "Xm180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    _gauss(
                        q["X180"]["amp"],
                        q["X180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "Xm90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    _gauss(
                        q["X90"]["amp"],
                        q["X90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "Y90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": _gauss(
                    q["Y90"]["amp"],
                    q["Y90"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    q["detuning"],
                )[0],
            },
            "Y180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": _gauss(
                    q["Y180"]["amp"],
                    q["Y180"]["length"],
                    q["DRAG_beta"],
                    q["anharmonicity"],
                    q["detuning"],
                )[0],
            },
            "Ym180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["Y180"]["amp"],
                        q["Y180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[0]
                ),
            },
            "Ym90_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["Y90"]["amp"],
                        q["Y90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[0]
                ),
            },
            "Y90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["Y90"]["amp"],
                        q["Y90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "Y180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": -1
                * np.array(
                    _gauss(
                        q["Y180"]["amp"],
                        q["Y180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "Ym180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    _gauss(
                        q["Y180"]["amp"],
                        q["Y180"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "Ym90_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    _gauss(
                        q["Y90"]["amp"],
                        q["Y90"]["length"],
                        q["DRAG_beta"],
                        q["anharmonicity"],
                        q["detuning"],
                    )[1]
                ),
            },
            "ef180_wf_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": _gauss(q["ef_amp"], q["ef_len"], 0, q["anharmonicity"], 0)[
                    0
                ],
            },
            "ef180_wf_der_" + q["q_name"]: {
                "type": "arbitrary",
                "samples": np.array(
                    _gauss(-q["ef_amp"], q["ef_len"], 0, q["anharmonicity"], 0)[1]
                ),
            },
        }
    digital_waveforms = {"ON": {"samples": [(1, 0)]}, "trig": {"samples": [(1, 100)]}}

    integration_weights = {
        "integ_W1_" + q["r_name"]: {
            "cosine": [np.cos(q["read_theta"])] * int(q["read_len"] / 4),
            "sine": [np.sin(q["read_theta"])] * int(q["read_len"] / 4),
        },
        "integ_W2_" + q["r_name"]: {
            "cosine": [np.cos(q["read_theta"] + np.pi / 2)] * int(q["read_len"] / 4),
            "sine": [np.sin(q["read_theta"] + np.pi / 2)] * int(q["read_len"] / 4),
        },
    }

    mixers = {
        q["q_mixer"]["mixer_name"]: [
            {
                "intermediate_frequency": q["IF_freq"] + q["detuning"],
                "lo_frequency": q["q_LO"]["LO_freq"],
                "correction": IQ_imbalance_corr(
                    q["q_mixer"]["mixer_gain"], q["q_mixer"]["mixer_phase"]
                ),
            }
        ],
        q["r_mixer"]["mixer_name"]: [
            {
                "intermediate_frequency": q["r_IF"],
                "lo_frequency": q["r_LO"]["LO_freq"],
                "correction": IQ_imbalance_corr(
                    q["r_mixer"]["mixer_gain"], q["r_mixer"]["mixer_phase"]
                ),
            }
        ],
    }

    return {
        "controllers": controllers,
        "elements": elements,
        "pulses": pulses,
        "waveforms": waveforms,
        "digital_waveforms": digital_waveforms,
        "integration_weights": integration_weights,
        "mixers": mixers,
    }


def meta_dict_to_config_6Q(meta_dict):
    q1_config = make_qubit_res_config(
        meta_dict["q1"]["q1"],
    )
    q2_config = make_qubit_res_config(
        meta_dict["q2"]["q2"],
    )
    q3_config = make_qubit_res_config(
        meta_dict["q3"]["q3"],
    )
    q4_config = make_qubit_res_config(
        meta_dict["q4"]["q4"],
    )
    q5_config = make_qubit_res_config(
        meta_dict["q5"]["q5"],
    )
    q6_config = make_qubit_res_config(
        meta_dict["q6"]["q6"],
    )
    config = merge({}, q1_config, q2_config, q3_config, q4_config, q5_config, q6_config)
    config["version"] = 1  # IMPORTANT TO INCLUDE
    return config


def meta_dict_to_config_1Q(single_qubit_meta_dict):
    """
    Single Qubit
    """
    config = make_qubit_res_config(single_qubit_meta_dict)
    config["version"] = 1  # IMPORTANT TO INCLUDE
    return config


def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]
