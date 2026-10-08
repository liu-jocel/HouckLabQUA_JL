import os
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.optimize import curve_fit
from qm.qua import *
from qm import QuantumMachinesManager

# Add LabCode to path
import sys
from pathlib import Path

# Get repo root (4 levels up: Gates → Experiments → LabCode → repo_root)
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))


from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.RB.m_RB_1Q_CRAB import RB1Q_CRAB
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def make_crab_waveform(tg, a_k, omega_k, phi_k, n_samples=None, normalize=False):
    """
    Returns a CRAB correction waveform as a sample array.
    omega_k must be in rad/ns (= 2*pi*f_MHz / 1000).
    n_samples: OPX requires multiples of 4, minimum 16.
    normalize: if True, rescales peak to 0.5 V (use for standalone pulse).
               if False (default), a_k directly sets the amplitude in V (use as additive correction).
    """
    if n_samples is None:
        n_samples = int(round(tg / 4)) * 4
        n_samples = max(n_samples, 16)

    t = np.arange(n_samples)
    pulse = np.zeros(n_samples)
    for k in range(len(a_k)):
        pulse += a_k[k] * np.cos(omega_k[k] * t + phi_k[k])

    if normalize:
        peak = np.max(np.abs(pulse))
        if peak > 0:
            pulse = pulse / peak * 0.5

    return pulse.tolist()


def add_x90_oc_to_config(
    config, q_name, crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y
):
    """
    Adds X90_OC, mX90_OC, Y90_OC, mY90_OC operations to q_name in the config.

    X90 and Y90 receive INDEPENDENT corrections, allowing for asymmetric distortions.
    This is more generic than shared corrections as it accounts for:
    - Different pulse envelopes (X vs Y have different shapes)
    - Asymmetric leakage/nonlinearities
    - Gate-specific distortion signatures

    X90_OC : X90_I + crab_I_X,  X90_Q + crab_Q_X
    Y90_OC : Y90_I + crab_I_Y,  Y90_Q + crab_Q_Y  (independent correction)
    mX90/mY90 are the negatives of the above.

    CRAB corrections are rescaled independently using headroom for each gate.
    """
    # X90: I←X90_wf_, Q←X90_wf_der_
    # Y90: I←Y90_wf_der_, Q←Y90_wf_   (note reversed _/der_ convention vs X90)
    x90_I = np.array(
        config["waveforms"]["X90_wf_" + q_name]["samples"]
    )  # phys-I of X90
    x90_Q = np.array(
        config["waveforms"]["X90_wf_der_" + q_name]["samples"]
    )  # phys-Q of X90
    y90_phyI = np.array(
        config["waveforms"]["Y90_wf_der_" + q_name]["samples"]
    )  # phys-I of Y90
    y90_phyQ = np.array(
        config["waveforms"]["Y90_wf_" + q_name]["samples"]
    )  # phys-Q of Y90

    def _fit(corr, headroom, channel):
        corr = np.array(corr)
        peak = np.max(np.abs(corr))
        if peak > headroom:
            scale = headroom / peak
            print(
                f"  WARNING: CRAB-{channel} peak={peak:.4f} V > headroom={headroom:.4f} V — rescaling by {scale:.4f}."
            )
            corr = corr * scale
        return corr

    # X90 headroom (independent)
    headroom_I_X = 0.5 - np.max(np.abs(x90_I))
    headroom_Q_X = 0.5 - np.max(np.abs(x90_Q))
    crab_I_X = _fit(crab_wf_I_X, headroom_I_X, "I_X")
    crab_Q_X = _fit(crab_wf_Q_X, headroom_Q_X, "Q_X")

    # Y90 headroom (independent)
    headroom_I_Y = 0.5 - np.max(np.abs(y90_phyI))
    headroom_Q_Y = 0.5 - np.max(np.abs(y90_phyQ))
    crab_I_Y = _fit(crab_wf_I_Y, headroom_I_Y, "I_Y")
    crab_Q_Y = _fit(crab_wf_Q_Y, headroom_Q_Y, "Q_Y")

    # X90 family: correction added to physical channels
    x90_oc_phyI = x90_I + crab_I_X
    x90_oc_phyQ = x90_Q + crab_Q_X

    # Y90 family: independent correction on physical channels
    y90_oc_phyI = y90_phyI + crab_I_Y
    y90_oc_phyQ = y90_phyQ + crab_Q_Y

    # Store waveforms — preserving the naming convention of the base gates:
    #   X90: X90_OC_wf_     → phys-I,  X90_OC_wf_der_ → phys-Q
    #   Y90: Y90_OC_wf_der_ → phys-I,  Y90_OC_wf_     → phys-Q
    config["waveforms"]["X90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": x90_oc_phyI.tolist(),
    }
    config["waveforms"]["X90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": x90_oc_phyQ.tolist(),
    }
    config["waveforms"]["mX90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": (-x90_oc_phyI).tolist(),
    }
    config["waveforms"]["mX90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": (-x90_oc_phyQ).tolist(),
    }
    config["waveforms"]["Y90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": y90_oc_phyI.tolist(),
    }
    config["waveforms"]["Y90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": y90_oc_phyQ.tolist(),
    }
    config["waveforms"]["mY90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": (-y90_oc_phyI).tolist(),
    }
    config["waveforms"]["mY90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": (-y90_oc_phyQ).tolist(),
    }

    def _add_pulse(name, wf_I_key, wf_Q_key):
        config["pulses"][name] = {
            "operation": "control",
            "length": len(config["waveforms"][wf_I_key]["samples"]),
            "waveforms": {"I": wf_I_key, "Q": wf_Q_key},
            "digital_marker": "trig",
        }

    _add_pulse(
        "X90_OC_pulse_" + q_name, "X90_OC_wf_" + q_name, "X90_OC_wf_der_" + q_name
    )
    _add_pulse(
        "mX90_OC_pulse_" + q_name, "mX90_OC_wf_" + q_name, "mX90_OC_wf_der_" + q_name
    )
    _add_pulse(
        "Y90_OC_pulse_" + q_name, "Y90_OC_wf_der_" + q_name, "Y90_OC_wf_" + q_name
    )
    _add_pulse(
        "mY90_OC_pulse_" + q_name, "mY90_OC_wf_der_" + q_name, "mY90_OC_wf_" + q_name
    )

    config["elements"][q_name]["operations"]["X90_OC"] = "X90_OC_pulse_" + q_name
    config["elements"][q_name]["operations"]["mX90_OC"] = "mX90_OC_pulse_" + q_name
    config["elements"][q_name]["operations"]["Y90_OC"] = "Y90_OC_pulse_" + q_name
    config["elements"][q_name]["operations"]["mY90_OC"] = "mY90_OC_pulse_" + q_name

    return config


def run_rb_fidelity(
    a_k_I_X,
    a_k_Q_X,
    a_k_I_Y,
    a_k_Q_Y,
    omega_k,
    phi_k_I,
    phi_k_Q,
    qubit,
    tg,
    meta_dict,
    experiment_dict,
    calib_params,
    outer_folder,
    iteration_label,
    phase_knobs=None,
    crab_dir=None,
):
    """
    Builds the CRAB-corrected config with independent X and Y corrections, runs RB1Q,
    returns (gate_fidelity, save_dir). If crab_dir is given, the RB figure is moved there.
    """
    # Fresh config each call so waveform edits don't accumulate
    config = meta_dict_to_config_1Q(meta_dict)

    crab_wf_I_X = make_crab_waveform(tg, a_k_I_X, omega_k, phi_k_I)
    crab_wf_Q_X = make_crab_waveform(tg, a_k_Q_X, omega_k, phi_k_Q)
    crab_wf_I_Y = make_crab_waveform(tg, a_k_I_Y, omega_k, phi_k_I)
    crab_wf_Q_Y = make_crab_waveform(tg, a_k_Q_Y, omega_k, phi_k_Q)
    crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y = (
        add_phase_knob_corrections_to_crab(
            config,
            qubit,
            tg,
            crab_wf_I_X,
            crab_wf_Q_X,
            crab_wf_I_Y,
            crab_wf_Q_Y,
            phase_knobs,
        )
    )
    config = add_x90_oc_to_config(
        config, qubit, crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y
    )

    rb = RB1Q_CRAB(
        qubit,
        experiment_dict["RB"]["circuit_lengths"],
        experiment_dict["RB"]["num_circuits"],
        experiment_dict["RB"]["shots"],
        calib_params,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=qubit,
        suffix=f"RB_OC_{iteration_label}",
        config=config,
    )

    if crab_dir is not None and os.path.isfile(rb.iname):
        shutil.move(rb.iname, os.path.join(crab_dir, os.path.basename(rb.iname)))

    return rb.gate_fidelity(), crab_dir if crab_dir is not None else os.path.dirname(
        rb.iname
    )


# ── Fast feedback: return probability with optional decay fitting ────────────


def run_rb_fast_feedback(
    a_k_I_X,
    a_k_Q_X,
    a_k_I_Y,
    a_k_Q_Y,
    omega_k,
    phi_k_I,
    phi_k_Q,
    qubit,
    tg,
    meta_dict,
    fast_circuit_depths,
    shots,
    num_circuits,
    calib_params,
    outer_folder,
    iteration_label,
    phase_knobs=None,
):
    """
    Run RB at one or more circuit depths for fast feedback with independent X and Y corrections.
    - If len(fast_circuit_depths) == 1: return return_probability at that depth
    - If len(fast_circuit_depths) > 1: fit exponential decay, return fitted decay parameter p
    """
    config = meta_dict_to_config_1Q(meta_dict)
    crab_wf_I_X = make_crab_waveform(tg, a_k_I_X, omega_k, phi_k_I)
    crab_wf_Q_X = make_crab_waveform(tg, a_k_Q_X, omega_k, phi_k_Q)
    crab_wf_I_Y = make_crab_waveform(tg, a_k_I_Y, omega_k, phi_k_I)
    crab_wf_Q_Y = make_crab_waveform(tg, a_k_Q_Y, omega_k, phi_k_Q)
    crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y = (
        add_phase_knob_corrections_to_crab(
            config,
            qubit,
            tg,
            crab_wf_I_X,
            crab_wf_Q_X,
            crab_wf_I_Y,
            crab_wf_Q_Y,
            phase_knobs,
        )
    )
    config = add_x90_oc_to_config(
        config, qubit, crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y
    )

    rb = RB1Q_CRAB(
        qubit,
        fast_circuit_depths,  # Can be single or multiple depths
        num_circuits,
        shots,
        calib_params,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=qubit,
        suffix=f"RB_fast_{iteration_label}",
        config=config,
    )

    # Extract return probability data
    if hasattr(rb, "data") and "return_prob" in rb.data:
        return_prob_array = rb.data["return_prob"]
        if not isinstance(return_prob_array, (list, tuple, np.ndarray)):
            return_prob_array = [return_prob_array]
        return_prob_array = np.array(return_prob_array)
    else:
        return_prob_array = np.full(len(fast_circuit_depths), 0.5)

    # Single depth: return the return probability directly
    if len(fast_circuit_depths) == 1:
        feedback_metric = np.mean(return_prob_array)
    # Multiple depths: fit exponential decay and return decay parameter p
    else:

        def fitRB(x, a, p, b):
            return a * p**x + b

        try:
            # Bounded fit avoids unphysical p>1 / p<0 fits that can fool CMA.
            x_fit = np.asarray(fast_circuit_depths, dtype=float)
            y_fit = np.asarray(return_prob_array, dtype=float)
            popt, _ = curve_fit(
                fitRB,
                x_fit,
                y_fit,
                p0=[0.4, 0.999, 0.5],
                bounds=([-1.0, 0.90, 0.0], [1.0, 1.0002, 1.0]),
                maxfev=5000,
            )
            feedback_metric = float(
                np.clip(popt[1], 0.0, 1.0)
            )  # Return bounded decay p
        except Exception:
            # Fallback: use mean return probability if fit fails
            feedback_metric = np.mean(return_prob_array)

    # Delete intermediate RB files
    if hasattr(rb, "iname") and rb.iname and os.path.isfile(rb.iname):
        os.remove(rb.iname)
    if hasattr(rb, "fname") and rb.fname and os.path.isfile(rb.fname):
        os.remove(rb.fname)

    return feedback_metric


# ── CMA-ES evaluate wrapper ───────────────────────────────────────────────────

_eval_counter = [0]  # mutable counter accessible inside closure

LONG_RB_FIDELITY_THRESHOLD = 0.99999


# ── CRAB / physical parameter-space helper ───────────────────────────────────
# PARAM_MODE options:
#   "shared_xy"        : 2K params. One complex CRAB correction for X; Y is the
#                        physical 90-degree I/Q rotation of X. Good first pass.
#   "shared_xy_phase"  : 2K + 5 params. Same CRAB space plus low-dimensional
#                        physical knobs that can correct pulse-axis / during-pulse
#                        phase-like errors: amplitude scale, DRAG scale,
#                        pulse-local detuning phase ramp, X-axis phase, and
#                        Y-relative phase.
#   "independent_xy"   : 4K params. Original fully independent X/Y corrections.
#
# Important limitation: the waveform/config layer can correct pulse-axis phase
# and pulse-local phase ramps. It cannot apply a true post-gate virtual-Z/frame
# update after every primitive; that requires changing the QUA/RB playback macro
# to insert frame_rotation_2pi() after gates.
PARAM_MODE = "shared_xy_phase"  # "shared_xy_phase", "shared_xy", or "independent_xy"
Y_ROTATION_SIGN = +1  # +1: Y corr = i*X corr => I_Y=-Q_X, Q_Y=I_X

# Conservative physical-knob bounds. These are decoded as max*tanh(x), so CMA
# samples in an unconstrained space but hardware changes remain bounded.
AMP_FRAC_MAX = 0.03  # ±3% common amplitude scale
DRAG_FRAC_MAX = 0.60  # ±60% change of existing derivative quadrature
DETUNING_MHZ_MAX = 6.0  # pulse-local linear phase ramp, ±MHz
X_AXIS_PHASE_CYCLES_MAX = 0.035  # ±12.6 deg. Global XY phase is weakly RB-identifiable.
Y_REL_PHASE_CYCLES_MAX = 0.035  # ±12.6 deg relative X/Y axis skew
DETUNING_PHASE_SIGN = +1  # flip if your detuning convention is opposite
PHASE_KNOB_NAMES = np.array(
    [
        "amp_frac",
        "drag_frac",
        "detuning_MHz",
        "x_axis_phase_cycles",
        "y_rel_phase_cycles",
    ]
)


def n_crab_parameters(K, param_mode=PARAM_MODE):
    if param_mode == "shared_xy":
        return 2 * K
    if param_mode == "shared_xy_phase":
        return 2 * K + len(PHASE_KNOB_NAMES)
    if param_mode == "independent_xy":
        return 4 * K
    raise ValueError(f"Unknown PARAM_MODE={param_mode!r}")


def _empty_phase_knobs():
    return dict(
        amp_frac=0.0,
        drag_frac=0.0,
        detuning_MHz=0.0,
        x_axis_phase_cycles=0.0,
        y_rel_phase_cycles=0.0,
    )


def phase_knobs_to_vector(phase_knobs):
    if phase_knobs is None:
        phase_knobs = _empty_phase_knobs()
    return np.array(
        [float(phase_knobs.get(k, 0.0)) for k in PHASE_KNOB_NAMES], dtype=float
    )


def _format_phase_knobs(phase_knobs):
    v = phase_knobs_to_vector(phase_knobs)
    return (
        f"amp={v[0]:+.4f}, drag={v[1]:+.3f}, det={v[2]:+.3f} MHz, "
        f"x_phase={360 * v[3]:+.2f} deg, y_rel={360 * v[4]:+.2f} deg"
    )


def decode_crab_parameters(
    x, K, a_max, param_mode=PARAM_MODE, y_rotation_sign=Y_ROTATION_SIGN
):
    """
    Decode CMA vector into the four coefficient arrays expected by the existing
    waveform/config code, plus a dict of optional physical phase-like knobs.

    shared_xy, 2K params:
        x = [a_I_X, a_Q_X]
        Y correction is the physical 90-degree rotation of X correction:
            I_Y = -Q_X, Q_Y = I_X     for y_rotation_sign=+1
        IMPORTANT: for this to be a literal rotation, use phi_k_Q = phi_k_I.

    shared_xy_phase, 2K+5 params:
        same CRAB coefficients plus
            amp_frac                : common scale of the base pulse
            drag_frac               : scale existing derivative quadrature
            detuning_MHz            : pulse-local linear phase ramp through pulse
            x_axis_phase_cycles     : constant axis phase for X and Y
            y_rel_phase_cycles      : additional Y-axis phase relative to X

    independent_xy, 4K params:
        original parameterization: [a_I_X, a_Q_X, a_I_Y, a_Q_Y].
    """
    x = np.asarray(x)
    phase_knobs = _empty_phase_knobs()

    if param_mode in ("shared_xy", "shared_xy_phase"):
        n_expected = n_crab_parameters(K, param_mode)
        if len(x) != n_expected:
            raise ValueError(f"{param_mode} expects {n_expected} params, got {len(x)}")
        a_k_I_X = a_max * np.tanh(x[:K])
        a_k_Q_X = a_max * np.tanh(x[K : 2 * K])
        a_k_I_Y = -y_rotation_sign * a_k_Q_X
        a_k_Q_Y = y_rotation_sign * a_k_I_X

        if param_mode == "shared_xy_phase":
            raw = x[2 * K : 2 * K + len(PHASE_KNOB_NAMES)]
            phase_knobs = dict(
                amp_frac=AMP_FRAC_MAX * np.tanh(raw[0]),
                drag_frac=DRAG_FRAC_MAX * np.tanh(raw[1]),
                detuning_MHz=DETUNING_MHZ_MAX * np.tanh(raw[2]),
                x_axis_phase_cycles=X_AXIS_PHASE_CYCLES_MAX * np.tanh(raw[3]),
                y_rel_phase_cycles=Y_REL_PHASE_CYCLES_MAX * np.tanh(raw[4]),
            )

    elif param_mode == "independent_xy":
        if len(x) != 4 * K:
            raise ValueError(f"independent_xy expects 4K={4 * K} params, got {len(x)}")
        a_k_I_X = a_max * np.tanh(x[:K])
        a_k_Q_X = a_max * np.tanh(x[K : 2 * K])
        a_k_I_Y = a_max * np.tanh(x[2 * K : 3 * K])
        a_k_Q_Y = a_max * np.tanh(x[3 * K : 4 * K])
    else:
        raise ValueError(f"Unknown PARAM_MODE={param_mode!r}")

    return a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs


def _apply_base_transform(I, Q, tg, phase_knobs, *, is_y_gate=False):
    """
    Apply low-dimensional physical knobs to a base complex envelope z=I+iQ.

    This is a waveform-level parametrization of phase-like errors:
      - constant pulse-axis phase: z(t) -> z(t) exp(i phi0)
      - pulse-local detuning/ramp: z(t) -> z(t) exp(i 2pi df (t-tc))

    This is NOT a true post-gate virtual-Z. A post-Z has to be implemented in
    the QUA sequence via frame_rotation_2pi after each primitive.
    """
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    phase_knobs = _empty_phase_knobs() if phase_knobs is None else phase_knobs

    amp_frac = float(phase_knobs.get("amp_frac", 0.0))
    drag_frac = float(phase_knobs.get("drag_frac", 0.0))
    detuning_MHz = float(phase_knobs.get("detuning_MHz", 0.0))
    x_axis_phase_cycles = float(phase_knobs.get("x_axis_phase_cycles", 0.0))
    y_rel_phase_cycles = float(phase_knobs.get("y_rel_phase_cycles", 0.0))

    # Scale the derivative quadrature relative to the main quadrature.
    # X convention: base I is main, base Q is derivative.
    # Y convention in your config: physical Q is main, physical I is derivative.
    if is_y_gate:
        I2 = (1.0 + amp_frac) * (1.0 + drag_frac) * I
        Q2 = (1.0 + amp_frac) * Q
        extra_phase_cycles = y_rel_phase_cycles
    else:
        I2 = (1.0 + amp_frac) * I
        Q2 = (1.0 + amp_frac) * (1.0 + drag_frac) * Q
        extra_phase_cycles = 0.0

    z = I2 + 1j * Q2
    n = len(z)
    t_ns = np.arange(n, dtype=float)
    t_centered_ns = t_ns - 0.5 * (n - 1)
    phase_cycles = (
        x_axis_phase_cycles
        + extra_phase_cycles
        + DETUNING_PHASE_SIGN * (detuning_MHz / 1000.0) * t_centered_ns
    )
    z = z * np.exp(1j * 2 * np.pi * phase_cycles)
    return z.real, z.imag


def add_phase_knob_corrections_to_crab(
    config,
    q_name,
    tg,
    crab_wf_I_X,
    crab_wf_Q_X,
    crab_wf_I_Y,
    crab_wf_Q_Y,
    phase_knobs=None,
):
    """
    Convert physical knobs into additive waveform corrections and add them to
    the CRAB correction waveforms. The output is still suitable for the existing
    add_x90_oc_to_config(), which adds corrections to the unmodified base gates.
    """
    if phase_knobs is None:
        phase_knobs = _empty_phase_knobs()
    if np.allclose(phase_knobs_to_vector(phase_knobs), 0.0):
        return crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y

    x90_I = np.array(config["waveforms"]["X90_wf_" + q_name]["samples"], dtype=float)
    x90_Q = np.array(
        config["waveforms"]["X90_wf_der_" + q_name]["samples"], dtype=float
    )
    y90_I = np.array(
        config["waveforms"]["Y90_wf_der_" + q_name]["samples"], dtype=float
    )
    y90_Q = np.array(config["waveforms"]["Y90_wf_" + q_name]["samples"], dtype=float)

    xI_new, xQ_new = _apply_base_transform(
        x90_I, x90_Q, tg, phase_knobs, is_y_gate=False
    )
    yI_new, yQ_new = _apply_base_transform(
        y90_I, y90_Q, tg, phase_knobs, is_y_gate=True
    )

    crab_wf_I_X = (np.asarray(crab_wf_I_X, dtype=float) + (xI_new - x90_I)).tolist()
    crab_wf_Q_X = (np.asarray(crab_wf_Q_X, dtype=float) + (xQ_new - x90_Q)).tolist()
    crab_wf_I_Y = (np.asarray(crab_wf_I_Y, dtype=float) + (yI_new - y90_I)).tolist()
    crab_wf_Q_Y = (np.asarray(crab_wf_Q_Y, dtype=float) + (yQ_new - y90_Q)).tolist()
    return crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y


def make_evaluate_fast(
    qubit,
    tg,
    x90_amp,
    omega_k,
    phi_k_I,
    phi_k_Q,
    meta_dict,
    fast_circuit_depths,
    shots,
    num_circuits,
    calib_params,
    outer_folder,
    param_mode=PARAM_MODE,
    y_rotation_sign=Y_ROTATION_SIGN,
):
    """
    Fast evaluate function.  Default shared_xy uses 2K parameters and is
    better conditioned for noisy RB optimization than the original 4K space.
    - If len(fast_circuit_depths) == 1: cost = 1 - return_probability
    - If len(fast_circuit_depths) > 1: cost = 1 - decay_parameter_p (from exponential fit)
    """
    a_max = x90_amp / 30
    K = len(omega_k)

    def evaluate(x):
        a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs = decode_crab_parameters(
            x, K, a_max, param_mode=param_mode, y_rotation_sign=y_rotation_sign
        )
        _eval_counter[0] += 1
        label = f"cma_iter_{_eval_counter[0]:04d}"

        feedback_metric = run_rb_fast_feedback(
            a_k_I_X,
            a_k_Q_X,
            a_k_I_Y,
            a_k_Q_Y,
            omega_k,
            phi_k_I,
            phi_k_Q,
            qubit,
            tg,
            meta_dict,
            fast_circuit_depths,
            shots,
            num_circuits,
            calib_params,
            outer_folder,
            label,
            phase_knobs=phase_knobs,
        )

        cost = 1.0 - feedback_metric
        fidelity = 1.0 - 0.5 * cost / 2.2083  # Approximate fidelity

        metric_name = "return_prob" if len(fast_circuit_depths) == 1 else "decay_p"
        knob_msg = (
            f"  |  {_format_phase_knobs(phase_knobs)}"
            if param_mode == "shared_xy_phase"
            else ""
        )
        print(
            f"[{label}]  {metric_name} = {feedback_metric:.6f}  →  cost = {cost:.6f}  (approx F = {fidelity:.6f}){knob_msg}"
        )

        return cost

    return evaluate


def make_evaluate(
    qubit,
    tg,
    x90_amp,
    omega_k,
    phi_k_I,
    phi_k_Q,
    meta_dict,
    experiment_dict,
    calib_params,
    outer_folder,
    crab_dir=None,
    long_rb_experiment_dict=None,
    param_mode=PARAM_MODE,
    y_rotation_sign=Y_ROTATION_SIGN,
):
    """
    Returns an evaluate(x) function closed over the fixed CRAB basis.
    Parameter layout depends on param_mode.  a_k decoded as a_max*tanh(x).
    If fidelity exceeds LONG_RB_FIDELITY_THRESHOLD a long RB is run immediately.
    """
    a_max = x90_amp / 30
    K = len(omega_k)

    def evaluate(x):
        a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs = decode_crab_parameters(
            x, K, a_max, param_mode=param_mode, y_rotation_sign=y_rotation_sign
        )
        _eval_counter[0] += 1
        label = f"cma_iter_{_eval_counter[0]:04d}"
        print(
            f"[{label}]  X: I [{a_k_I_X.min():.4f}, {a_k_I_X.max():.4f}]  Q [{a_k_Q_X.min():.4f}, {a_k_Q_X.max():.4f}] | "
            f"Y: I [{a_k_I_Y.min():.4f}, {a_k_I_Y.max():.4f}]  Q [{a_k_Q_Y.min():.4f}, {a_k_Q_Y.max():.4f}]"
        )
        if param_mode == "shared_xy_phase":
            print(f"[{label}]  phase knobs: {_format_phase_knobs(phase_knobs)}")

        fidelity, _ = run_rb_fidelity(
            a_k_I_X,
            a_k_Q_X,
            a_k_I_Y,
            a_k_Q_Y,
            omega_k,
            phi_k_I,
            phi_k_Q,
            qubit,
            tg,
            meta_dict,
            experiment_dict,
            calib_params,
            outer_folder,
            label,
            phase_knobs=phase_knobs,
            crab_dir=crab_dir,
        )

        if np.isnan(fidelity):
            print("  WARNING: RB returned NaN, substituting cost=1.0")
            return 1.0

        print(f"  gate_fidelity = {fidelity:.8f}  →  cost = {1 - fidelity:.8f}")

        if (
            fidelity > LONG_RB_FIDELITY_THRESHOLD
            and long_rb_experiment_dict is not None
        ):
            print(
                f"  *** fidelity {fidelity:.8f} > {LONG_RB_FIDELITY_THRESHOLD} — running long RB ***"
            )
            long_fidelity, _ = run_rb_fidelity(
                a_k_I_X,
                a_k_Q_X,
                a_k_I_Y,
                a_k_Q_Y,
                omega_k,
                phi_k_I,
                phi_k_Q,
                qubit,
                tg,
                meta_dict,
                long_rb_experiment_dict,
                calib_params,
                outer_folder,
                f"{label}_long",
                phase_knobs=phase_knobs,
                crab_dir=crab_dir,
            )
            print(f"  Long RB gate_fidelity = {long_fidelity:.8f}")

        return 1.0 - fidelity

    return evaluate


### USAGE -----------------------------------------------------------------

if __name__ == "__main__":
    """
    Overnight RB-only CRAB information run.

    Goal: maximize information, not merely minimize one noisy orange curve.
    This script runs several short CMA campaigns with different parameter spaces,
    validates each incumbent by interleaved zero/xbest fast RB, and spends the
    last part of the night deeply remeasuring the best candidates.

    No leakage RB and no purity RB are run.
    """
    import csv
    import json
    import time
    import traceback

    # --------------------------- user knobs ---------------------------------
    qubit = "q6"
    TOTAL_HOURS = 8.0
    FAST_CIRCUIT_DEPTHS = [64, 256, 1000, 3000]

    # Fast-RB statistics.  This is still RB-only.
    FAST_SHOTS = 100
    FAST_NUM_CIRCUITS = 200

    # Validation is the key anti-selection-bias check.
    VALIDATE_EVERY = 4  # CMA generations
    VALIDATION_PAIRS = 3  # zero,xbest repeated this many times during campaigns
    FINAL_VALIDATION_PAIRS = 6  # deeper final zero/winner interleaving
    PASS_Z = 1.5  # only a rough flag; inspect the raw numbers

    RUN_SINGLE_SHOT_CAL = True
    RUN_FINAL_LONG_RB = True  # RB only; no leakage/purity

    # The overnight schedule.  The physical/phase-only knobs looked weak, so
    # spend little time there.  Spend most time comparing shared CRAB vs
    # shared CRAB+phase knobs, then one small independent-XY test.
    CAMPAIGNS = [
        dict(
            name="physical_only_5D",
            param_mode="shared_xy_phase",
            f_k=[],
            sigma0=0.25,
            maxiter=20,
            budget_hours=0.75,
            notes="5 physical knobs only: amp, DRAG, detuning-ramp, x-axis phase, y-relative phase.",
        ),
        dict(
            name="shared_xy_crab_2K",
            param_mode="shared_xy",
            f_k=[5, 15, 30, 50, 70, 90, 100, 110, 130, 160],
            sigma0=0.22,
            maxiter=90,
            budget_hours=2.25,
            notes="Most likely useful: constrained shared X/Y CRAB basis near alpha plus low-frequency distortion.",
        ),
        dict(
            name="shared_xy_crab_plus_phase_2K5",
            param_mode="shared_xy_phase",
            f_k=[5, 15, 30, 50, 70, 90, 100, 110, 130, 160],
            sigma0=0.22,
            maxiter=90,
            budget_hours=2.25,
            notes="Tests whether physical phase/amp/DRAG knobs help once CRAB waveform freedom is present.",
        ),
        dict(
            name="independent_xy_small_4K",
            param_mode="independent_xy",
            f_k=[10, 30, 70, 90, 110, 140],
            sigma0=0.18,
            maxiter=70,
            budget_hours=1.50,
            notes="Small X/Y-asymmetry test.  More expressive but noisier; do not use K=12 overnight.",
        ),
    ]

    # ---------------------- common experiment setup --------------------------
    t_global0 = time.time()
    global_deadline = t_global0 + TOTAL_HOURS * 3600.0

    meta_dict = make_meta_dict(qubit)[qubit]
    tg = meta_dict["X90"]["length"]
    x90_amp = meta_dict["X90"]["amp"]
    a_max = x90_amp / 30
    outer_folder = meta_dict["global_params"]["outer_folder"]
    os.makedirs(outer_folder, exist_ok=True)

    today = datetime.date.today().strftime("%Y_%m_%d")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    crab_dir = os.path.join(outer_folder, qubit, f"{qubit}_{today}", "crab_overnight")
    crab_run_dir = os.path.join(crab_dir, run_timestamp)
    os.makedirs(crab_run_dir, exist_ok=True)

    print("=" * 90)
    print("OVERNIGHT RB-ONLY CRAB INFORMATION RUN")
    print(f"Saving outputs to: {crab_run_dir}")
    print(f"Qubit: {qubit} | tg={tg} ns | x90_amp={x90_amp} V | a_max={a_max:.6g} V")
    print(f"Total wall-clock budget: {TOTAL_HOURS:.2f} h")
    print("No leakage RB. No purity RB.")
    print("=" * 90)

    with open(os.path.join(crab_run_dir, "overnight_config.json"), "w") as f:
        json.dump(
            dict(
                qubit=qubit,
                total_hours=TOTAL_HOURS,
                fast_circuit_depths=FAST_CIRCUIT_DEPTHS,
                fast_shots=FAST_SHOTS,
                fast_num_circuits=FAST_NUM_CIRCUITS,
                validate_every=VALIDATE_EVERY,
                validation_pairs=VALIDATION_PAIRS,
                final_validation_pairs=FINAL_VALIDATION_PAIRS,
                campaigns=CAMPAIGNS,
                amp_frac_max=AMP_FRAC_MAX,
                drag_frac_max=DRAG_FRAC_MAX,
                detuning_mhz_max=DETUNING_MHZ_MAX,
                x_axis_phase_cycles_max=X_AXIS_PHASE_CYCLES_MAX,
                y_rel_phase_cycles_max=Y_REL_PHASE_CYCLES_MAX,
            ),
            f,
            indent=2,
        )

    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {}
    if RUN_SINGLE_SHOT_CAL:
        print("Running single-shot calibration...")
        SS = SingleShot1Q(
            qubit,
            1000,
            meta_dict=meta_dict,
            outer_folder=outer_folder,
            path=qubit,
            suffix="SS_Cal_overnight",
            config=config,
        )
        calib_params = SS.calib_params
        meta_dict["read_theta"] = SS.calib_params["read_theta"]
        config = meta_dict_to_config_1Q(meta_dict)
        print("Single-shot calib_params:", SS.calib_params)

    fast_experiment_dict = {
        "RB": {
            "circuit_lengths": FAST_CIRCUIT_DEPTHS,
            "num_circuits": FAST_NUM_CIRCUITS,
            "shots": FAST_SHOTS,
        }
    }
    long_rb_experiment_dict = {
        "RB": {
            "circuit_lengths": [
                2,
                16,
                64,
                256,
                500,
                1000,
                1500,
                2000,
                2500,
                3000,
                5000,
                7000,
            ],
            "num_circuits": 100,
            "shots": 100,
        }
    }

    validation_csv = os.path.join(crab_run_dir, f"overnight_validation_{qubit}.csv")
    campaign_csv = os.path.join(crab_run_dir, f"overnight_campaign_summary_{qubit}.csv")

    def write_validation_record(record):
        exists = os.path.exists(validation_csv)
        with open(validation_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(record)

    def write_campaign_summary(record):
        exists = os.path.exists(campaign_csv)
        with open(campaign_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(record)

    def mean_sem(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = float(std / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        return mean, std, sem

    def make_strategy(c):
        f_k = np.asarray(c["f_k"], dtype=float)
        K = len(f_k)
        omega_k = f_k * 2 * np.pi / 1000.0
        rng = np.random.default_rng()  # intentionally independent per strategy
        phi_k_I = rng.uniform(0, 2 * np.pi, K)
        if c["param_mode"] in ("shared_xy", "shared_xy_phase"):
            phi_k_Q = phi_k_I.copy()
        else:
            phi_k_Q = rng.uniform(0, 2 * np.pi, K)
        n_params = n_crab_parameters(K, c["param_mode"])
        return dict(
            c, K=K, omega_k=omega_k, phi_k_I=phi_k_I, phi_k_Q=phi_k_Q, n_params=n_params
        )

    def decode_for_strategy(x, s):
        return decode_crab_parameters(
            x,
            s["K"],
            a_max,
            param_mode=s["param_mode"],
            y_rotation_sign=Y_ROTATION_SIGN,
        )

    def fast_metric_for_x(x, s, label):
        a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs = decode_for_strategy(x, s)
        metric = run_rb_fast_feedback(
            a_k_I_X,
            a_k_Q_X,
            a_k_I_Y,
            a_k_Q_Y,
            s["omega_k"],
            s["phi_k_I"],
            s["phi_k_Q"],
            qubit,
            tg,
            meta_dict,
            FAST_CIRCUIT_DEPTHS,
            FAST_SHOTS,
            FAST_NUM_CIRCUITS,
            calib_params,
            crab_run_dir,
            label,
            phase_knobs=phase_knobs,
        )
        return float(metric)

    def cost_for_x(x, s, label):
        metric = fast_metric_for_x(x, s, label)
        return 1.0 - metric

    def validate_candidate(s, x, label, n_pairs=VALIDATION_PAIRS, campaign_iter=-1):
        x = np.asarray(x, dtype=float).copy()
        zero = np.zeros_like(x)
        costs = {"zero": [], "candidate": []}
        metrics = {"zero": [], "candidate": []}
        print("\n" + "-" * 80)
        print(f"VALIDATION {label}: zero/candidate interleaved, n_pairs={n_pairs}")
        print(
            f"strategy={s['name']} mode={s['param_mode']} K={s['K']} n_params={s['n_params']}"
        )
        print("-" * 80)
        for j in range(n_pairs):
            for kind, x_use in [("zero", zero), ("candidate", x)]:
                metric = fast_metric_for_x(x_use, s, f"{label}_pair{j + 1:02d}_{kind}")
                cost = 1.0 - metric
                metrics[kind].append(metric)
                costs[kind].append(cost)
                print(
                    f"  {kind:9s} pair {j + 1:02d}: metric={metric:.8f}, cost={cost:.6e}"
                )

        z_mean, z_std, z_sem = mean_sem(costs["zero"])
        c_mean, c_std, c_sem = mean_sem(costs["candidate"])
        improvement = z_mean - c_mean
        sem_diff = (
            np.sqrt(z_sem**2 + c_sem**2)
            if np.isfinite(z_sem) and np.isfinite(c_sem)
            else np.nan
        )
        z_score = (
            improvement / sem_diff if np.isfinite(sem_diff) and sem_diff > 0 else np.nan
        )
        verdict = (
            "pass"
            if np.isfinite(z_score) and z_score >= PASS_Z and improvement > 0
            else ("inconclusive" if improvement > 0 else "fail")
        )

        _, _, _, _, phase_knobs = decode_for_strategy(x, s)
        record = dict(
            wall_time_hr=(time.time() - t_global0) / 3600.0,
            campaign=s["name"],
            param_mode=s["param_mode"],
            K=s["K"],
            n_params=s["n_params"],
            label=label,
            campaign_iter=campaign_iter,
            n_pairs=n_pairs,
            zero_mean=z_mean,
            zero_std=z_std,
            zero_sem=z_sem,
            cand_mean=c_mean,
            cand_std=c_std,
            cand_sem=c_sem,
            improvement=improvement,
            sem_diff=sem_diff,
            z_score=z_score,
            verdict=verdict,
            amp_frac=phase_knobs.get("amp_frac", 0.0),
            drag_frac=phase_knobs.get("drag_frac", 0.0),
            detuning_MHz=phase_knobs.get("detuning_MHz", 0.0),
            x_axis_phase_cycles=phase_knobs.get("x_axis_phase_cycles", 0.0),
            y_rel_phase_cycles=phase_knobs.get("y_rel_phase_cycles", 0.0),
        )
        write_validation_record(record)
        print(f"zero cost      = {z_mean:.6e} ± {z_sem:.2e} SEM (std={z_std:.2e})")
        print(f"candidate cost = {c_mean:.6e} ± {c_sem:.2e} SEM (std={c_std:.2e})")
        print(
            f"improvement zero−candidate = {improvement:+.3e}; z≈{z_score:.2f}; verdict={verdict}"
        )
        if s["param_mode"] == "shared_xy_phase":
            print("candidate physical knobs:", _format_phase_knobs(phase_knobs))
        return record

    def plot_campaign(s, history, validations=None):
        if len(history) == 0:
            return
        hist = np.asarray(history, dtype=float)
        best_so_far = np.minimum.accumulate(hist)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hist, marker="o", markersize=3, alpha=0.45, label="best in generation")
        ax.plot(best_so_far, marker="o", markersize=3, label="best so far")
        ax.set_xlabel("CMA-ES generation")
        ax.set_ylabel("Best fast-RB cost in generation (1 - decay p)")
        ax.set_yscale("log")
        ax.set_title(f"{s['name']} — {qubit}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(crab_run_dir, f"convergence_{qubit}_{s['name']}.png"), dpi=200
        )
        plt.close(fig)

    def plot_summary(candidate_records):
        if not candidate_records:
            return
        labels = [r["label"] for r in candidate_records]
        imp = [r["improvement"] for r in candidate_records]
        err = [
            r["sem_diff"] if np.isfinite(r["sem_diff"]) else 0.0
            for r in candidate_records
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.axhline(0.0, linestyle="--", color="gray")
        ax.errorbar(
            np.arange(len(labels)),
            imp,
            yerr=err,
            marker="o",
            capsize=3,
            linestyle="none",
        )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Validated fast-RB improvement: zero cost − candidate cost")
        ax.set_title(f"Overnight candidate validation summary — {qubit}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(crab_run_dir, f"overnight_validation_summary_{qubit}.png"),
            dpi=200,
        )
        plt.close(fig)

    # Baseline drift/noise monitor before doing any CMA.  Uses physical_only zero
    # because zero correction is identical independent of the basis.
    print("\nInitial baseline drift/noise monitor: repeated zero fast RB")
    baseline_s = make_strategy(
        dict(name="baseline_zero_monitor", param_mode="shared_xy_phase", f_k=[])
    )
    zero_x = np.zeros(baseline_s["n_params"])
    zero_costs = []
    for j in range(4):
        if time.time() > global_deadline:
            break
        c0 = cost_for_x(zero_x, baseline_s, f"baseline_zero_monitor_{j + 1:02d}")
        zero_costs.append(c0)
        print(f"  zero monitor {j + 1}: cost={c0:.6e}")
    zmean, zstd, zsem = mean_sem(zero_costs)
    print(f"Initial zero fast-RB cost: {zmean:.6e} ± {zsem:.2e} SEM (std={zstd:.2e})")

    all_candidates = []
    campaign_summaries = []

    def run_campaign(s):
        print("\n" + "=" * 90)
        print(f"START CAMPAIGN {s['name']}")
        print(
            f"mode={s['param_mode']} K={s['K']} n_params={s['n_params']} f_k={list(np.asarray(s['f_k'], dtype=float))}"
        )
        print(s.get("notes", ""))
        print("=" * 90)

        camp_dir = os.path.join(crab_run_dir, s["name"])
        os.makedirs(camp_dir, exist_ok=True)
        local_deadline = min(
            global_deadline, time.time() + float(s.get("budget_hours", 1.0)) * 3600.0
        )
        n_params = s["n_params"]
        popsize = int(4 + 3 * np.log(max(n_params, 2)))
        es = cma.CMAEvolutionStrategy(
            np.zeros(n_params),
            float(s.get("sigma0", 0.2)),
            dict(
                maxiter=int(s.get("maxiter", 50)),
                popsize=popsize,
                tolfun=1e-5,
                tolx=1e-6,
                verbose=3,
            ),
        )
        history = []
        sigma_history = []
        iter_idx = 0
        consecutive_failish = 0
        last_validation = None

        def evaluate_local(x):
            label = f"{s['name']}_iter{iter_idx + 1:03d}_eval{len(eval_costs) + 1:02d}"
            try:
                return cost_for_x(x, s, label)
            except Exception as exc:
                print(f"WARNING: evaluation failed for {label}: {exc}")
                traceback.print_exc()
                return 1.0

        while (
            not es.stop()
            and time.time() < local_deadline
            and time.time() < global_deadline
        ):
            solutions = es.ask()
            eval_costs = []
            for x in solutions:
                if time.time() >= local_deadline or time.time() >= global_deadline:
                    break
                eval_costs.append(evaluate_local(x))
            if len(eval_costs) != len(solutions):
                print(
                    "Campaign time budget reached mid-generation; skipping partial tell and ending campaign."
                )
                break
            es.tell(solutions, eval_costs)
            es.disp()
            iter_idx += 1
            gen_best = float(np.min(eval_costs))
            history.append(gen_best)
            sigma_history.append(float(es.sigma))
            print(
                f"[{s['name']}] gen {iter_idx}: gen_best={gen_best:.6e}, overall_best={es.result.fbest:.6e}, sigma={es.sigma:.4g}"
            )
            plot_campaign(s, history)

            if VALIDATE_EVERY and iter_idx % VALIDATE_EVERY == 0:
                rec = validate_candidate(
                    s,
                    es.result.xbest,
                    f"{s['name']}_gen{iter_idx:03d}",
                    n_pairs=VALIDATION_PAIRS,
                    campaign_iter=iter_idx,
                )
                last_validation = rec
                all_candidates.append(
                    dict(
                        strategy=s,
                        x=np.asarray(es.result.xbest).copy(),
                        record=rec,
                        history=list(history),
                    )
                )
                plot_summary([c["record"] for c in all_candidates])
                if rec["improvement"] <= 0:
                    consecutive_failish += 1
                else:
                    consecutive_failish = 0
                # If a low-info space is repeatedly not validating, move on.
                if consecutive_failish >= 2:
                    print(
                        f"[{s['name']}] Two consecutive non-improving validations; moving to next campaign."
                    )
                    break

        # End-of-campaign validation, even if no scheduled validation landed.
        if len(history) > 0 and time.time() < global_deadline:
            rec = validate_candidate(
                s,
                es.result.xbest,
                f"{s['name']}_final",
                n_pairs=VALIDATION_PAIRS,
                campaign_iter=iter_idx,
            )
            last_validation = rec
            all_candidates.append(
                dict(
                    strategy=s,
                    x=np.asarray(es.result.xbest).copy(),
                    record=rec,
                    history=list(history),
                )
            )
            plot_summary([c["record"] for c in all_candidates])

        best_x = (
            np.asarray(es.result.xbest).copy()
            if len(history) > 0
            else np.zeros(n_params)
        )
        a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs = decode_for_strategy(best_x, s)
        np.savez(
            os.path.join(camp_dir, f"best_params_{qubit}_{s['name']}.npz"),
            xbest=best_x,
            a_k_I_X=a_k_I_X,
            a_k_Q_X=a_k_Q_X,
            a_k_I_Y=a_k_I_Y,
            a_k_Q_Y=a_k_Q_Y,
            phase_knob_names=PHASE_KNOB_NAMES,
            phase_knob_values=phase_knobs_to_vector(phase_knobs),
            omega_k=s["omega_k"],
            f_k=np.asarray(s["f_k"], dtype=float),
            phi_k_I=s["phi_k_I"],
            phi_k_Q=s["phi_k_Q"],
            history=np.asarray(history),
            sigma_history=np.asarray(sigma_history),
            param_mode=s["param_mode"],
        )
        summary = dict(
            campaign=s["name"],
            param_mode=s["param_mode"],
            K=s["K"],
            n_params=s["n_params"],
            n_generations=len(history),
            cma_fbest=float(es.result.fbest) if len(history) else np.nan,
            last_val_improvement=last_validation["improvement"]
            if last_validation
            else np.nan,
            last_val_z=last_validation["z_score"] if last_validation else np.nan,
            last_val_verdict=last_validation["verdict"] if last_validation else "none",
            elapsed_hr=(time.time() - t_global0) / 3600.0,
        )
        campaign_summaries.append(summary)
        write_campaign_summary(summary)
        print(f"END CAMPAIGN {s['name']}: {summary}")
        return summary

    # Run the campaigns until the global deadline is approached.
    for c in CAMPAIGNS:
        if time.time() > global_deadline - 20 * 60:
            print(
                "Less than ~20 min left; skipping remaining campaigns for final validation."
            )
            break
        s = make_strategy(c)
        try:
            run_campaign(s)
        except Exception as exc:
            print(f"ERROR in campaign {c['name']}: {exc}")
            traceback.print_exc()

    # ------------------------ final deep validation --------------------------
    print("\n" + "=" * 90)
    print("FINAL SELECTION / DEEP RB-ONLY VALIDATION")
    print("=" * 90)
    if len(all_candidates) == 0:
        print("No candidates were produced. Exiting.")
    else:
        # Prefer candidates that validated, not candidates that only had a lucky CMA minimum.
        def rank_key(c):
            r = c["record"]
            # Primary: validated improvement. Secondary: z-score.
            z = r["z_score"] if np.isfinite(r["z_score"]) else -999.0
            return (r["improvement"], z)

        all_candidates_sorted = sorted(all_candidates, key=rank_key, reverse=True)
        top = all_candidates_sorted[:2]
        deep_records = []
        for idx, cand in enumerate(top):
            if time.time() > global_deadline - 10 * 60:
                print("Not enough time for more deep validations.")
                break
            s = cand["strategy"]
            x = cand["x"]
            rec = validate_candidate(
                s,
                x,
                f"DEEP_rank{idx + 1}_{s['name']}",
                n_pairs=FINAL_VALIDATION_PAIRS,
                campaign_iter=-1,
            )
            deep_records.append(dict(strategy=s, x=x, record=rec))
            all_candidates.append(dict(strategy=s, x=x, record=rec, history=[]))
            plot_summary([c["record"] for c in all_candidates])

        if deep_records:
            winner = max(deep_records, key=lambda c: c["record"]["improvement"])
        else:
            winner = all_candidates_sorted[0]
        sw = winner["strategy"]
        xw = winner["x"]
        rw = winner["record"]
        print("\nWINNER BY VALIDATED FAST-RB IMPROVEMENT:")
        print(
            f"  campaign={sw['name']} mode={sw['param_mode']} K={sw['K']} n_params={sw['n_params']}"
        )
        print(
            f"  improvement={rw['improvement']:+.3e}, z≈{rw['z_score']:.2f}, verdict={rw['verdict']}"
        )
        _, _, _, _, phase_knobs = decode_for_strategy(xw, sw)
        if sw["param_mode"] == "shared_xy_phase":
            print("  physical knobs:", _format_phase_knobs(phase_knobs))

        # Save winner in a top-level file for easy loading tomorrow.
        a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs = decode_for_strategy(xw, sw)
        np.savez(
            os.path.join(crab_run_dir, f"WINNER_best_params_{qubit}.npz"),
            campaign=sw["name"],
            param_mode=sw["param_mode"],
            K=sw["K"],
            f_k=np.asarray(sw["f_k"], dtype=float),
            omega_k=sw["omega_k"],
            phi_k_I=sw["phi_k_I"],
            phi_k_Q=sw["phi_k_Q"],
            xbest=xw,
            a_k_I_X=a_k_I_X,
            a_k_Q_X=a_k_Q_X,
            a_k_I_Y=a_k_I_Y,
            a_k_Q_Y=a_k_Q_Y,
            phase_knob_names=PHASE_KNOB_NAMES,
            phase_knob_values=phase_knobs_to_vector(phase_knobs),
            validated_improvement=rw["improvement"],
            validated_z_score=rw["z_score"],
        )

        if RUN_FINAL_LONG_RB and time.time() < global_deadline - 10 * 60:
            print("\nFinal long RB on zero and winner. RB-only; no leakage/purity.")
            # zero and winner under the winning strategy.  Zero waveform is the original gate.
            zero = np.zeros_like(xw)
            for kind, x_use in [("zero", zero), ("winner", xw)]:
                a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y, phase_knobs = decode_for_strategy(
                    x_use, sw
                )
                fid, _ = run_rb_fidelity(
                    a_k_I_X,
                    a_k_Q_X,
                    a_k_I_Y,
                    a_k_Q_Y,
                    sw["omega_k"],
                    sw["phi_k_I"],
                    sw["phi_k_Q"],
                    qubit,
                    tg,
                    meta_dict,
                    long_rb_experiment_dict,
                    calib_params,
                    crab_run_dir,
                    f"FINAL_LONG_RB_{kind}_{sw['name']}",
                    phase_knobs=phase_knobs,
                    crab_dir=crab_run_dir,
                )
                print(
                    f"FINAL_LONG_RB {kind:6s}: fidelity={fid:.8f}, error={1.0 - fid:.6e}"
                )

    print("\nOvernight run complete.")
    print(f"Outputs saved in: {crab_run_dir}")
