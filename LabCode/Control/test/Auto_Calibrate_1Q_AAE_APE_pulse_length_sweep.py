"""
Pulse-length / DRAG-beta sweep with keep-phase APE calibration.

This is a minimal-edit rewrite of the older DRAG autocal sweep:
    Chevron bootstrap -> SS cal -> [AAE -> APE] repeated N times -> optional RB/LRB/PRB

Default post-chevron calibration sequence for each (pulse length, beta):
    AAE -> APE -> AAE -> APE
because AAE can move the effective phase and APE can slightly move the best
amplitude condition. Set AAE_APE_CYCLES to change the number of AAE/APE pairs.

Important:
    * Uses m_amplified_phase_error_keep_phase.APE.
    * No Frequency_Calibration_XY is run after the AAE/APE calibration block.
    * APE correction convention, validated by sanity triad:
          meta_dict["IF_freq"] += int(optimal_detuning_MHz * u.MHz)
"""

import os
import time
import copy
import h5py
from datetime import datetime
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# Add LabCode to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

plt.ion()
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q

# from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Rabi.m_Rabi_Chevron_IQ import RabiChevronIQ
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage
from LabCode.Experiments.RB.m_RB_1Q import RB1Q
from LabCode.Experiments.RB.m_RB_purity_1Q import RB1Q_Purity
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2E import T2E
from qualang_tools.units import unit
from LabCode.Experiments.Gates.m_amplified_phase_error import APE
# IMPORTANT: keep-phase corrected APE. Do not silently import the old APE.


u = unit(coerce_to_integer=True)

# Gates per Clifford for the 1Q Clifford decomposition (53/24).
GATES_PER_CLIFFORD = 2.2083

# =============================================================================
# User settings
# =============================================================================
PULSE_TYPE = "X90"
QUBIT = "q6"

# RB/readout experiment flags. These are intentionally independent so you can
# run a fast calibration-only pass, then enable RB/LRB/PRB when the APE residuals
# look good.
RUN_RB1Q = True
RUN_LEAKAGE_RB = True
RUN_PURITY_RB = True          # only runs for the best-beta point per pulse length
RUN_RB_3STATE = True          # vanilla RB readout uses 3-state cal if available
RUN_COHERENCE_MEASUREMENTS = False

# For backward compatibility with older plotting helper naming.
DO_LEAKAGE_CHARACTERIZATION = RUN_LEAKAGE_RB
DO_PURITY_RB = RUN_PURITY_RB
DO_RB_3STATE = RUN_RB_3STATE
DO_COHERENCE_MEASUREMENTS = RUN_COHERENCE_MEASUREMENTS

NUM_AVERAGES = 3              # number of RB/LRB repeats where applicable

# AAE/APE calibration after the chevron bootstrap. AAE_APE_CYCLES=2 gives:
#     AAE -> APE -> AAE -> APE
AAE_APE_CYCLES = 2
FINAL_APE_SANITY_CHECK = True
FINAL_APE_RESIDUAL_TARGET_MHZ = 0.03

# Diagnostic: when True, additionally run the old AAE + Frequency_Calibration_XY
# calibration (to convergence) from the same post-chevron seed, then measure the
# leftover detuning with an APE sanity. Lets us compare the residual left by the old
# Freq-XY cal against the residual left by the APE-based cal on the same figure. This
# branch runs on a copy of the state and does NOT affect the applied calibration.
FREQ_XYCAL_CHECK = True

# Chevron amplitude bound. Rotation ~ amp*length, so the pi/2 amplitude shrinks
# as the pulse lengthens. Sweeping fast->slow, bound each chevron's amp_max by
# previous pulse length's beta=0 calibrated amplitude times this headroom.
CHEVRON_AMP_MAX_HEADROOM = 1.3
HARDWARE_AMP_MAX = 0.49

# Sweep vectors.
PULSE_LENGTH_VEC = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 60]  # ns
DRAG_beta_vec = np.array([0])
# DRAG_beta_vec = np.linspace(0, 1.8, 16)

optimizer_convergence_params = {
    "amplitude_accuracy": 0.0005,
    "ape_accuracy_MHz": 0.03,
    "aae_ape_cycles": AAE_APE_CYCLES,
    # FREQ_XYCAL_CHECK staged AAE + Frequency_Calibration_XY convergence.
    "detuning_accuracy": 0.05,  # MHz
    "freq_xycal_max_iter": 3,
}

experiment_dict = {
    "SS_Cal": {
        "shots": 1000,
        "number_pi_pulses": 1,
        "ground_threshold": 0.7,
        "object": SingleShot1Q,
    },
    "SS_Cal_3L": {
        "shots": 600,
        "object": SingleShot1Q3Level,
    },
    "Chevron_IQ": {
        "shots": 21,
        "amp_min": 0.0,
        "amp_max": 0.35,
        "amp_points": 31,
        "freq_span": 21 * u.MHz,
        "freq_points": 11,
        "object": RabiChevronIQ,
    },
    "Chevron": {
        "shots": 20,
        "amp_min": 0.0,
        "amp_max": 0.35,
        "amp_points": 31,
        "freq_span": 21 * u.MHz,
        "freq_points": 31,
        "object": RabiChevronSS,
    },
    "Amplified_Amplitude_Error": {
        "shots": 200,
        "N_min": 1,
        "N_step": 1,
        # The first two entries are used for AAE_APE_CYCLES=2.
        # If AAE_APE_CYCLES is larger, the last value is reused.
        "N_max_sequence": [11, 51, 81],
        "object": AAE,
    },
    "Amplified_Phase_Error": {
        "shots": 100,
        "center": (0.0, 0.0),
        "center_mode": "fixed",
        "min_radius": 0.15,
        "local_fit_points": 5,
        "allow_extrapolation": False,
        "keep_phase_during_detuning": True,
        # Broad sweep around current IF to find the zero.
        "broad_span_MHz": 8.0,
        "broad_points": 17,
        "broad_N_vec": np.arange(1, 21, 1),
        # Narrow sweep centered on the broad optimum.
        "narrow_span_MHz": 1.5,
        "narrow_points": 11,
        "narrow_N_vec": np.arange(1, 21, 1),
        # Fine sweep centered on the narrow optimum; this is the applied correction.
        "fine_span_MHz": 0.50,
        "fine_points": 21,
        "fine_N_vec": np.arange(1, 21, 1),
        # Final sanity around zero after the last APE correction. Check only.
        "sanity_span_MHz": 1.0,
        "sanity_points": 15,
        "sanity_N_vec": np.arange(1, 21, 1),
        "object": APE,
    },
    # Used only by FREQ_XYCAL_CHECK (the old-style comparison calibration).
    "Frequency_Calibration_XY": {
        "shots": 100,
        "pulse_sequence": [
            ("X90", "I"),
            ("Y90", "I"),
            ("X90", "Y90"),
            ("Y90", "X90"),
        ],
        "freq_span_stages": [5 * u.MHz, 1 * u.MHz],
        "freq_points": 401,
        "num_repeats": 1,
        "object": Frequency_Calibration_XY,
    },
    "Leakage_RB": {
        "shots": 25,
        "num_circuits": 20,
        "num_averages": NUM_AVERAGES,
        "circuit_lengths": [
            2, 4, 8, 16, 64, 128, 250, 300, 500, 600, 800, 1000,
            1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000,
        ],
        "object": RB1Q_Leakage,
    },
    "RB1Q": {
        "run": RUN_RB1Q,
        "simulate": False,
        "shots": 50,
        "num_circuits": 50,
        "circuit_lengths": [
            2, 4, 8, 16, 32, 64, 128, 256, 300, 400, 500, 600,
            700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000,
            5000, 6000, 7000,
        ],
        "object": RB1Q,
    },
    "Purity_RB": {
        "run": RUN_PURITY_RB,
        "shots": 100,
        "num_circuits": 100,
        "circuit_lengths": [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256,
            384, 512, 768, 1500, 2000, 3000, 4000,
        ],
        "object": RB1Q_Purity,
    },
    "T1": {
        "run": RUN_COHERENCE_MEASUREMENTS,
        "shots": 500,
        "num_pulses": [1],
        "t_min": 400 * u.ns,
        "t_max": 5000 * u.us,
        "t_points": 51,
        "object": T1,
    },
    "T2E": {
        "run": RUN_COHERENCE_MEASUREMENTS,
        "shots": 500,
        "t_min": 0.5 * u.us,
        "t_max": 5000 * u.us,
        "t_points": 51,
        "object": T2E,
    },
}


# =============================================================================
# Helpers
# =============================================================================
def _stage_value(seq, idx):
    """Return seq[idx], reusing the last element if idx exceeds len(seq)-1."""
    if len(seq) == 0:
        raise ValueError("stage sequence is empty")
    return seq[min(idx, len(seq) - 1)]


def _f_vec_MHz(center_MHz, span_MHz, points):
    """Vector in integer Hz, centered at center_MHz with total span span_MHz."""
    lo = (center_MHz - 0.5 * span_MHz) * u.MHz
    hi = (center_MHz + 0.5 * span_MHz) * u.MHz
    return np.linspace(lo, hi, points, dtype=int)


def _ape_opt_MHz(ape):
    return float(ape.data.get("optimal_detuning_MHz", np.nan))


def _ape_status(ape):
    return {
        "opt_MHz": _ape_opt_MHz(ape),
        "zero_bracketed": bool(ape.data.get("zero_bracketed", False)),
        "fit_status": ape.data.get("fit_status", "unknown"),
        "slope_over_expected": float(ape.data.get("slope_over_expected", np.nan)),
    }


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def _safe_nanstd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if np.any(np.isfinite(arr)) else np.nan


def _run_ape_once(
    qubit,
    meta_dict,
    config,
    calib_params,
    pulse_type,
    ape_cfg,
    f_vec,
    N_vec,
    outer_folder,
    path,
    suffix,
):
    print(
        f"      APE {suffix}: IF={meta_dict['IF_freq']/u.MHz:.6f} MHz, "
        f"f=[{np.min(f_vec)/u.MHz:+.3f},{np.max(f_vec)/u.MHz:+.3f}] MHz, "
        f"N=[{int(np.min(N_vec))},{int(np.max(N_vec))}], "
        f"pts={len(f_vec)}x{len(N_vec)}"
    )
    ape = APE(
        qubit,
        f_vec,
        N_vec,
        ape_cfg["shots"],
        calib_params,
        pulse_type=pulse_type,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=path,
        suffix=suffix,
        config=config,
        # The robust/keep-phase APE accepts the following arguments.
        center=ape_cfg["center"],
        center_mode=ape_cfg["center_mode"],
        min_radius=ape_cfg["min_radius"],
        local_fit_points=ape_cfg["local_fit_points"],
        allow_extrapolation=ape_cfg["allow_extrapolation"],
        keep_phase_during_detuning=ape_cfg["keep_phase_during_detuning"],
        flat_path=True,
    )
    st = _ape_status(ape)
    print(
        f"      APE result {suffix}: opt={st['opt_MHz']:+.6f} MHz, "
        f"bracketed={st['zero_bracketed']}, status={st['fit_status']}, "
        f"slope/expected={st['slope_over_expected']:.3f}"
    )
    return ape


def _run_ape_staged_and_apply(
    qubit,
    meta_dict,
    config,
    calib_params,
    pulse_type,
    exp_dict,
    save_path,
    suffix_prefix,
):
    """Run broad -> narrow -> fine APE, apply fine optimum to IF_freq, return config and status."""
    ape_cfg = exp_dict["Amplified_Phase_Error"]
    outer_folder = meta_dict["global_params"]["outer_folder"]

    # Broad around current operating point.
    f_broad = _f_vec_MHz(0.0, ape_cfg["broad_span_MHz"], ape_cfg["broad_points"])
    ape_broad = _run_ape_once(
        qubit, meta_dict, config, calib_params, pulse_type, ape_cfg,
        f_broad, ape_cfg["broad_N_vec"], outer_folder, save_path,
        f"{suffix_prefix}_APE_broad",
    )
    opt_broad = _ape_opt_MHz(ape_broad)
    if not np.isfinite(opt_broad):
        print("      WARNING: broad APE did not return a finite optimum; skipping APE apply.")
        return config, {
            "broad_MHz": opt_broad,
            "narrow_MHz": np.nan,
            "fine_MHz": np.nan,
            "applied_MHz": 0.0,
            "bracketed": False,
            "fit_status": "broad_failed",
            "slope_over_expected": np.nan,
        }

    # Narrow centered at broad optimum.
    f_narrow = _f_vec_MHz(opt_broad, ape_cfg["narrow_span_MHz"], ape_cfg["narrow_points"])
    ape_narrow = _run_ape_once(
        qubit, meta_dict, config, calib_params, pulse_type, ape_cfg,
        f_narrow, ape_cfg["narrow_N_vec"], outer_folder, save_path,
        f"{suffix_prefix}_APE_narrow",
    )
    opt_narrow = _ape_opt_MHz(ape_narrow)
    if not np.isfinite(opt_narrow):
        print("      WARNING: narrow APE did not return a finite optimum; skipping APE apply.")
        return config, {
            "broad_MHz": opt_broad,
            "narrow_MHz": opt_narrow,
            "fine_MHz": np.nan,
            "applied_MHz": 0.0,
            "bracketed": False,
            "fit_status": "narrow_failed",
            "slope_over_expected": np.nan,
        }

    # Fine centered at narrow optimum. This is the value we apply.
    f_fine = _f_vec_MHz(opt_narrow, ape_cfg["fine_span_MHz"], ape_cfg["fine_points"])
    ape_fine = _run_ape_once(
        qubit, meta_dict, config, calib_params, pulse_type, ape_cfg,
        f_fine, ape_cfg["fine_N_vec"], outer_folder, save_path,
        f"{suffix_prefix}_APE_fine",
    )
    st_fine = _ape_status(ape_fine)
    opt_fine = st_fine["opt_MHz"]
    if not np.isfinite(opt_fine):
        print("      WARNING: fine APE did not return a finite optimum; skipping APE apply.")
        return config, {
            "broad_MHz": opt_broad,
            "narrow_MHz": opt_narrow,
            "fine_MHz": opt_fine,
            "applied_MHz": 0.0,
            "bracketed": False,
            "fit_status": "fine_failed",
            "slope_over_expected": st_fine["slope_over_expected"],
        }

    old_IF = int(meta_dict["IF_freq"])
    delta_IF = int(opt_fine * u.MHz)
    meta_dict["IF_freq"] = old_IF + delta_IF
    config = meta_dict_to_config_1Q(meta_dict)
    print(
        "      Applied APE correction: "
        f"old_IF={old_IF/u.MHz:.6f} MHz, "
        f"df={delta_IF/u.MHz:+.6f} MHz, "
        f"new_IF={meta_dict['IF_freq']/u.MHz:.6f} MHz"
    )

    return config, {
        "broad_MHz": opt_broad,
        "narrow_MHz": opt_narrow,
        "fine_MHz": opt_fine,
        "applied_MHz": delta_IF / u.MHz,
        "bracketed": st_fine["zero_bracketed"],
        "fit_status": st_fine["fit_status"],
        "slope_over_expected": st_fine["slope_over_expected"],
    }


def _run_final_ape_sanity(
    qubit,
    meta_dict,
    config,
    calib_params,
    pulse_type,
    exp_dict,
    save_path,
    suffix_prefix,
):
    ape_cfg = exp_dict["Amplified_Phase_Error"]
    f_vec = _f_vec_MHz(0.0, ape_cfg["sanity_span_MHz"], ape_cfg["sanity_points"])
    ape = _run_ape_once(
        qubit, meta_dict, config, calib_params, pulse_type, ape_cfg,
        f_vec, ape_cfg["sanity_N_vec"], meta_dict["global_params"]["outer_folder"],
        save_path, f"{suffix_prefix}_APE_final_sanity",
    )
    st = _ape_status(ape)
    return st


def _run_freq_xycal_and_measure_residual(
    qubit,
    meta_dict,
    config,
    calib_params,
    pulse_type,
    exp_dict,
    conv_params,
    save_path,
    suffix_prefix,
):
    """Old-style calibration for comparison: staged AAE + Frequency_Calibration_XY run
    to convergence, then an APE sanity measurement of the leftover detuning.

    Returns the APE-measured residual detuning (MHz) so it can be compared head-to-head
    with the APE-based calibration residual. Mutates the passed meta_dict/config, so call
    it on a deepcopy of the post-chevron state."""
    aae_cfg = exp_dict["Amplified_Amplitude_Error"]
    xy_cfg = exp_dict["Frequency_Calibration_XY"]
    max_iter = int(conv_params.get("freq_xycal_max_iter", 3))
    N_max_stages = aae_cfg["N_max_sequence"]
    freq_span_stages = xy_cfg["freq_span_stages"]

    for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
        for i in range(max_iter):
            # AAE: re-pin amplitude (the XY phase fit assumes a clean pi/2).
            N_vec = np.arange(aae_cfg["N_min"], N_max, aae_cfg["N_step"])
            aae = AAE(
                qubit,
                N_vec,
                aae_cfg["shots"],
                calib_params=calib_params,
                pulse_type=pulse_type,
                meta_dict=meta_dict,
                outer_folder=meta_dict["global_params"]["outer_folder"],
                path=save_path,
                suffix=f"{suffix_prefix}_AAE_s{stage_idx}_i{i:02d}",
                config=config,
                flat_path=True,
            )
            amp_err = float(aae.data["amplitude_error"])
            meta_dict["X90"]["amp"] += amp_err
            meta_dict["Y90"]["amp"] += amp_err
            config = meta_dict_to_config_1Q(meta_dict)

            # Frequency_Calibration_XY: correct the detuning.
            freq_vec = np.linspace(
                -freq_span / 2, freq_span / 2, xy_cfg["freq_points"], dtype=int
            )
            fxy = Frequency_Calibration_XY(
                qubit,
                freq_vec,
                xy_cfg["pulse_sequence"],
                xy_cfg["shots"],
                calib_params,
                num_repeats=xy_cfg["num_repeats"],
                meta_dict=meta_dict,
                outer_folder=meta_dict["global_params"]["outer_folder"],
                path=save_path,
                suffix=f"{suffix_prefix}_FreqXY_s{stage_idx}_i{i:02d}",
                config=config,
                flat_path=True,
            )
            det_corr_MHz = float(fxy.data["optimal_detuning_MHz"])
            meta_dict["IF_freq"] += int(det_corr_MHz * u.MHz)
            config = meta_dict_to_config_1Q(meta_dict)
            print(
                f"      [XYcal s{stage_idx} i{i:02d}] amp_err={amp_err:+.3e} "
                f"det_corr={det_corr_MHz:+.4f} MHz"
            )
            if abs(amp_err) < conv_params["amplitude_accuracy"] and abs(
                det_corr_MHz
            ) < conv_params.get("detuning_accuracy", 0.05):
                break

    # APE sanity on the XY-calibrated frame: the leftover detuning APE still sees.
    st = _run_final_ape_sanity(
        qubit,
        meta_dict,
        config,
        calib_params,
        pulse_type,
        exp_dict,
        save_path,
        suffix_prefix=f"{suffix_prefix}_postXY",
    )
    return st["opt_MHz"]


def _run_rb_error_rate(
    meta_dict, config, calib_params, calib_3L, pl_folder, beta_label, suffix_prefix
):
    """Run vanilla RB NUM_AVERAGES times and return (error_rate_mean, error_rate_std).
    Uses 3-state readout when RUN_RB_3STATE and a 3-level cal is available, matching the
    main RB call so the result is directly comparable."""
    rb_3state_calib = calib_3L if (RUN_RB_3STATE and calib_3L is not None) else None
    fids = []
    for avg_idx in range(NUM_AVERAGES):
        rb1q = RB1Q(
            QUBIT,
            experiment_dict["RB1Q"]["circuit_lengths"],
            experiment_dict["RB1Q"]["num_circuits"],
            experiment_dict["RB1Q"]["shots"],
            calib_params,
            meta_dict=meta_dict,
            outer_folder=pl_folder,
            path=beta_label,
            suffix=f"{suffix_prefix}_avg{avg_idx:02d}",
            config=config,
            simulate=experiment_dict["RB1Q"]["simulate"],
            flat_path=True,
            calib_params_3state=rb_3state_calib,
        )
        fids.append(rb1q.data.get("gate_fidelity", float("nan")))
    fid_mean = _safe_nanmean(fids)
    fid_std = _safe_nanstd(fids)
    err_mean = 1.0 - fid_mean if np.isfinite(fid_mean) else np.nan
    return err_mean, fid_std


def save_drag_summary(
    pl_folder,
    qubit,
    pulse_length_ns,
    do_leakage,
    drag_results,
    rb_error_rates,
    rb_error_rate_errs,
    freq_xycal_rb_errors=None,
):
    """Rewrite per-pulse-length summary H5 + figure from accumulated beta results."""
    if len(drag_results) == 0:
        return None

    betas = np.array([r[0] for r in drag_results])
    l1s = np.array([r[1] for r in drag_results])
    l1errs = np.array([r[2] for r in drag_results])
    l2s = np.array([r[3] for r in drag_results])
    l2errs = np.array([r[4] for r in drag_results])
    lambda1s = np.array([r[5] for r in drag_results])
    amps = np.array([r[6] for r in drag_results])
    if_freqs = np.array([r[7] for r in drag_results])
    final_ape_resid = np.array([r[8] if len(r) > 8 else np.nan for r in drag_results])
    last_ape_opt = np.array([r[9] if len(r) > 9 else np.nan for r in drag_results])
    initial_ape_opt = np.array([r[10] if len(r) > 10 else np.nan for r in drag_results])
    freq_xycal_resid = np.array([r[11] if len(r) > 11 else np.nan for r in drag_results])
    rb_err_arr = np.array(rb_error_rates)
    rb_err_std_arr = np.array(rb_error_rate_errs)
    if freq_xycal_rb_errors is not None and len(freq_xycal_rb_errors) == len(betas):
        freq_xycal_rb_arr = np.array(freq_xycal_rb_errors, dtype=float)
    else:
        freq_xycal_rb_arr = np.full(len(betas), np.nan)

    drag_summary_path = os.path.join(pl_folder, f"{qubit}_autocal_summary.h5")
    with h5py.File(drag_summary_path, "w") as f:
        f.create_dataset("betas", data=betas)
        f.create_dataset("L1", data=l1s)
        f.create_dataset("L1_err", data=l1errs)
        f.create_dataset("L2", data=l2s)
        f.create_dataset("L2_err", data=l2errs)
        f.create_dataset("lambda1", data=lambda1s)
        f.create_dataset("rb_error_rate", data=rb_err_arr)
        f.create_dataset("rb_error_rate_err", data=rb_err_std_arr)
        f.create_dataset("freq_xycal_rb_error_rate", data=freq_xycal_rb_arr)
        f.create_dataset("amp", data=amps)
        f.create_dataset("IF_freq_Hz", data=if_freqs)
        f.create_dataset("final_ape_residual_MHz", data=final_ape_resid)
        f.create_dataset("last_ape_opt_MHz", data=last_ape_opt)
        f.create_dataset("initial_ape_opt_MHz", data=initial_ape_opt)
        f.create_dataset("freq_xycal_ape_resid_MHz", data=freq_xycal_resid)
        f.attrs["qubit"] = qubit
        f.attrs["pulse_length_ns"] = int(pulse_length_ns)
        f.attrs["AAE_APE_CYCLES"] = int(AAE_APE_CYCLES)

    if do_leakage:
        one_minus_lambda1s = 1.0 - lambda1s
        fig_drag, axes_drag = plt.subplots(5, 1, figsize=(7, 15), sharex=True)
        for ax_d, ys, yerrs, ylabel, ylim_bottom in zip(
            axes_drag[:3],
            [l1s, l2s, one_minus_lambda1s],
            [l1errs, l2errs, np.zeros(len(one_minus_lambda1s))],
            ["L1  (leakage per Clifford)", "L2  (leakage per Clifford)", "1 - λ1"],
            [1e-6, None, 1e-5],
        ):
            ax_d.errorbar(betas, ys, yerr=yerrs, fmt="o-", capsize=4)
            ax_d.set_yscale("log")
            ax_d.set_ylim(bottom=ylim_bottom)
            ax_d.set_ylabel(ylabel)
            ax_d.grid(True, which="both", alpha=0.3)

        axes_drag[3].errorbar(
            betas, rb_err_arr, yerr=rb_err_std_arr, fmt="o-", capsize=4,
            label="APE-cal RB error (total)",
        )
        if np.any(np.isfinite(freq_xycal_rb_arr)):
            axes_drag[3].plot(
                betas, freq_xycal_rb_arr, "^-", color="C2",
                label="Freq-XY-cal RB error (total)",
            )
        axes_drag[3].errorbar(
            betas, l1s, yerr=l1errs, fmt="s-", capsize=4,
            label="Leakage error (L1)",
        )
        axes_drag[3].set_yscale("log")
        axes_drag[3].set_ylim(bottom=1e-5)
        axes_drag[3].set_ylabel("Error rate")
        axes_drag[3].legend()
        axes_drag[3].grid(True, which="both", alpha=0.3)

        axes_drag[4].plot(
            betas, initial_ape_opt, "s-", color="C1",
            label="initial measured detuning (cycle 0)",
        )
        axes_drag[4].plot(
            betas, final_ape_resid, "o-", color="C0", label="APE-cal residual",
        )
        if np.any(np.isfinite(freq_xycal_resid)):
            axes_drag[4].plot(
                betas, freq_xycal_resid, "^-", color="C2",
                label="Freq-XY-cal residual (APE-measured)",
            )
        axes_drag[4].axhline(0, color="k", lw=1, alpha=0.5)
        axes_drag[4].axhline(+FINAL_APE_RESIDUAL_TARGET_MHZ, color="k", ls="--", lw=1, alpha=0.4)
        axes_drag[4].axhline(-FINAL_APE_RESIDUAL_TARGET_MHZ, color="k", ls="--", lw=1, alpha=0.4)
        axes_drag[4].set_ylabel("APE detuning [MHz]")
        axes_drag[4].legend(fontsize=8)
        axes_drag[4].grid(True, alpha=0.3)

        axes_drag[-1].set_xlabel("DRAG beta")
        axes_drag[0].set_title(f"{qubit} — leakage/RB vs DRAG beta  ({pulse_length_ns} ns pulse)")
        fig_drag.tight_layout()
        drag_fig_path = os.path.join(pl_folder, f"{qubit}_L1_L2_lambda1_vs_DRAG_beta.png")
        fig_drag.savefig(drag_fig_path, bbox_inches="tight")
        plt.close(fig_drag)
    else:
        fig_drag, ax_drag = plt.subplots(1, 1, figsize=(7, 5))
        ax_drag.errorbar(betas, rb_err_arr, yerr=rb_err_std_arr, fmt="o-", capsize=4)
        ax_drag.set_yscale("log")
        ax_drag.set_ylim(bottom=1e-5)
        ax_drag.set_ylabel("Total error rate (RB)")
        ax_drag.set_xlabel("DRAG beta")
        ax_drag.grid(True, which="both", alpha=0.3)
        ax_drag.set_title(f"{qubit} — RB fidelity vs DRAG beta  ({pulse_length_ns} ns pulse)")
        fig_drag.tight_layout()
        drag_fig_path = os.path.join(pl_folder, f"{qubit}_RB_vs_DRAG_beta.png")
        fig_drag.savefig(drag_fig_path, bbox_inches="tight")
        plt.close(fig_drag)
    return drag_fig_path


def _run_calibration(
    qubit,
    meta_dict,
    config,
    calib_params,
    pulse_type,
    exp_dict,
    conv_params,
    beta_label,
    save_path=None,
    chevron_amp_max=None,
):
    """
    Rabi chevron bootstrap + repeated [AAE -> APE] calibration.

    This replaces the old AAE + Frequency_Calibration_XY loop. Frequency_XY is
    deliberately not used here because APE is the correction target for the
    pulse-pair phase error.

    Modifies meta_dict in-place and returns (config, calibration_summary).
    """
    if save_path is None:
        save_path = qubit

    # Phase 1: Rabi chevron bootstrap, kept from the original autocal skeleton.
    chev = exp_dict["Chevron"]
    amp_max = chev["amp_max"] if chevron_amp_max is None else chevron_amp_max
    amp_vec = np.linspace(chev["amp_min"], amp_max, chev["amp_points"])
    f_vec = np.linspace(
        -chev["freq_span"] / 2,
        chev["freq_span"] / 2,
        chev["freq_points"],
        dtype=int,
    )

    chevron = RabiChevronSS(
        qubit,
        amp_vec,
        f_vec,
        chev["shots"],
        calib_params,
        num_pi_pulses=1,
        pulse_type=pulse_type,
        meta_dict=meta_dict,
        outer_folder=meta_dict["global_params"]["outer_folder"],
        path=save_path,
        suffix=f"Chevron_{beta_label}",
        config=config,
        flat_path=True,
    )

    opt_detuning_Hz = chevron.data["opt_detuning_Hz"]
    opt_amp = chevron.data["opt_amp"]

    print(
        f"    Chevron first π-pulse at detuning={opt_detuning_Hz / u.MHz:.3f} MHz, "
        f"amp={opt_amp:.4f} V"
    )

    meta_dict["IF_freq"] += int(opt_detuning_Hz)
    meta_dict["X90"]["amp"] = opt_amp
    meta_dict["Y90"]["amp"] = opt_amp
    config = meta_dict_to_config_1Q(meta_dict)

    # Snapshot the post-chevron state so the optional Freq-XY-cal comparison branch
    # starts from the identical seed as the AAE/APE path (fair residual comparison).
    if FREQ_XYCAL_CHECK:
        xycal_meta = copy.deepcopy(meta_dict)
        xycal_config = meta_dict_to_config_1Q(xycal_meta)

    # Phase 2: repeated AAE -> APE. For AAE_APE_CYCLES=2 this is:
    #     AAE -> APE -> AAE -> APE
    aae_cfg = exp_dict["Amplified_Amplitude_Error"]
    cycles = int(conv_params.get("aae_ape_cycles", AAE_APE_CYCLES))
    cal_history = []
    last_ape_opt_MHz = np.nan

    for cycle_idx in range(cycles):
        print(f"    ── AAE/APE cycle {cycle_idx + 1}/{cycles} ──")

        # AAE
        N_max = _stage_value(aae_cfg["N_max_sequence"], cycle_idx)
        N_vec = np.arange(aae_cfg["N_min"], N_max, aae_cfg["N_step"])
        aae = AAE(
            qubit,
            N_vec,
            aae_cfg["shots"],
            calib_params=calib_params,
            pulse_type=pulse_type,
            meta_dict=meta_dict,
            outer_folder=meta_dict["global_params"]["outer_folder"],
            path=save_path,
            suffix=f"AAE_{beta_label}_cycle{cycle_idx:02d}",
            config=config,
            flat_path=True,
        )
        amp_err = float(aae.data["amplitude_error"])
        meta_dict["X90"]["amp"] += amp_err
        meta_dict["Y90"]["amp"] += amp_err
        config = meta_dict_to_config_1Q(meta_dict)
        print(
            f"      AAE cycle {cycle_idx:02d}: amp_err={amp_err:+.6e}, "
            f"new_amp={meta_dict['X90']['amp']:.6f}"
        )

        # APE: broad -> narrow -> fine, then apply fine correction to IF_freq.
        config, ape_summary = _run_ape_staged_and_apply(
            qubit,
            meta_dict,
            config,
            calib_params,
            pulse_type,
            exp_dict,
            save_path,
            suffix_prefix=f"{beta_label}_cycle{cycle_idx:02d}",
        )
        last_ape_opt_MHz = ape_summary["fine_MHz"]

        cal_history.append({
            "cycle": cycle_idx,
            "amp_err": amp_err,
            "amp": float(meta_dict["X90"]["amp"]),
            "IF_freq_Hz": int(meta_dict["IF_freq"]),
            **ape_summary,
        })

        print(
            f"      End cycle {cycle_idx:02d}: IF={meta_dict['IF_freq']/u.MHz:.6f} MHz, "
            f"amp={meta_dict['X90']['amp']:.6f}, fine_APE={last_ape_opt_MHz:+.6f} MHz"
        )

    final_sanity = {
        "opt_MHz": np.nan,
        "zero_bracketed": False,
        "fit_status": "not_run",
        "slope_over_expected": np.nan,
    }
    if FINAL_APE_SANITY_CHECK:
        print("    ── Final APE sanity check after AAE/APE cycles ──")
        final_sanity = _run_final_ape_sanity(
            qubit,
            meta_dict,
            config,
            calib_params,
            pulse_type,
            exp_dict,
            save_path,
            suffix_prefix=f"{beta_label}_final",
        )
        print(
            f"    Final APE residual={final_sanity['opt_MHz']:+.6f} MHz "
            f"(target ±{FINAL_APE_RESIDUAL_TARGET_MHZ:.3f} MHz)"
        )

    # Detuning APE measured on the very first pass, before any APE correction was
    # applied (cycle 0 fine optimum). Compared against the final residual, this shows
    # whether APE's frequency picking actually moved the operating point.
    initial_ape_opt_MHz = cal_history[0]["fine_MHz"] if cal_history else np.nan

    # Optional comparison: old AAE + Frequency_Calibration_XY from the same post-chevron
    # seed, then APE-measure the leftover detuning. Runs on the snapshot copy so the
    # applied (APE-based) calibration above is untouched.
    freq_xycal_ape_resid_MHz = np.nan
    if FREQ_XYCAL_CHECK:
        print(
            "    ── FREQ_XYCAL_CHECK: old AAE + Frequency_Calibration_XY, then APE sanity ──"
        )
        freq_xycal_ape_resid_MHz = _run_freq_xycal_and_measure_residual(
            qubit,
            xycal_meta,
            xycal_config,
            calib_params,
            pulse_type,
            exp_dict,
            conv_params,
            save_path,
            suffix_prefix=f"{beta_label}_XYCAL",
        )
        print(
            f"    Residual comparison: Freq-XY-cal={freq_xycal_ape_resid_MHz:+.6f} MHz  "
            f"vs  APE-cal={final_sanity['opt_MHz']:+.6f} MHz (both APE-measured)"
        )

    calibration_summary = {
        "history": cal_history,
        "final_ape_residual_MHz": final_sanity["opt_MHz"],
        "final_ape_bracketed": final_sanity["zero_bracketed"],
        "final_ape_fit_status": final_sanity["fit_status"],
        "final_ape_slope_over_expected": final_sanity["slope_over_expected"],
        "last_ape_opt_MHz": last_ape_opt_MHz,
        "initial_ape_opt_MHz": initial_ape_opt_MHz,
        "freq_xycal_ape_resid_MHz": freq_xycal_ape_resid_MHz,
        # Converged Freq-XY-calibrated state (for the comparison RB). None unless checked.
        "xycal_meta": (copy.deepcopy(xycal_meta) if FREQ_XYCAL_CHECK else None),
    }

    return config, calibration_summary


# =============================================================================
# Main sweep
# =============================================================================
_base_meta = make_meta_dict(QUBIT)[QUBIT]
outer_folder_base = _base_meta["global_params"]["outer_folder"]
_date_str = datetime.now().strftime("%Y_%m_%d")
_q_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{_date_str}")
os.makedirs(_q_date_dir, exist_ok=True)

parent_folder = os.path.join(
    _q_date_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pulse_length_sweep_AAE_APE",
)
os.makedirs(parent_folder, exist_ok=True)
print(f"Pulse length sweep folder: {parent_folder}")

# (pulse_len_ns, best_beta, L1, L1_err, L2, L2_err, lambda1, amp, IF_freq_Hz,
#  rb_err, rb_err_std, r_coh, r_coh_err, r_incoh, r_incoh_err, unitarity, unitarity_err,
#  final_ape_residual_MHz)
sweep_results = []
coherence_measured = False
sweep_start = time.time()
print(f"Sweep started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Global H5 file for incremental updates.
global_h5_path = os.path.join(parent_folder, f"{QUBIT}_pulse_length_sweep_summary.h5")
with h5py.File(global_h5_path, "w") as f:
    for key in (
        "pulse_lengths_ns", "best_beta", "rb_error_rate", "rb_error_rate_std",
        "L1", "L1_err", "L2", "L2_err", "r_coh", "r_coh_err",
        "r_incoh", "r_incoh_err", "unitarity", "unitarity_err",
        "final_ape_residual_MHz", "best_amp", "best_IF_freq_Hz",
    ):
        f.create_dataset(key, (0,), maxshape=(None,), dtype=float)
    f.attrs["qubit"] = QUBIT
    f.attrs["num_averages"] = NUM_AVERAGES
    f.attrs["run_rb1q"] = RUN_RB1Q
    f.attrs["run_leakage_rb"] = RUN_LEAKAGE_RB
    f.attrs["run_purity_rb"] = RUN_PURITY_RB
    f.attrs["run_rb_3state"] = RUN_RB_3STATE
    f.attrs["aae_ape_cycles"] = AAE_APE_CYCLES

# Live/final 4x1 summary vs gate time.
fig, axes = plt.subplots(5, 1, figsize=(8, 15), sharex=True)
fig.suptitle(f"{QUBIT} — Leakage, RB, and APE Residual vs Gate Time")

pls_list, rb_errs_list, rb_stds_list = [], [], []
l1s_list, l1_errs_list, l2s_list, l2_errs_list, lambda1s_list = [], [], [], [], []
ape_resid_list = []
prev_beta0_amp = None
plot_path = None

for pulse_length_ns in PULSE_LENGTH_VEC:
    pl_label = f"{pulse_length_ns}ns"
    pl_folder = os.path.join(parent_folder, pl_label)
    os.makedirs(pl_folder, exist_ok=True)
    pl_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"Pulse length = {pulse_length_ns} ns  |  started {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")

    # IQ chevron bootstrap once per pulse length.
    _chev_iq = experiment_dict["Chevron_IQ"]
    if prev_beta0_amp is None:
        _amp_max_iq = _chev_iq["amp_max"]
    else:
        _amp_max_iq = min(HARDWARE_AMP_MAX, prev_beta0_amp * CHEVRON_AMP_MAX_HEADROOM)
        print(
            f"  Chevron amp_max set to {_amp_max_iq:.4f} V "
            f"(prev pulse beta=0 amp {prev_beta0_amp:.4f} V x {CHEVRON_AMP_MAX_HEADROOM})"
        )
    _amp_vec_iq = np.linspace(_chev_iq["amp_min"], _amp_max_iq, _chev_iq["amp_points"])
    _f_vec_iq = np.linspace(-_chev_iq["freq_span"] / 2, _chev_iq["freq_span"] / 2,
                            _chev_iq["freq_points"], dtype=int)
    _md_iq = make_meta_dict(QUBIT)[QUBIT]
    _md_iq["X90"]["length"] = pulse_length_ns
    _md_iq["Y90"]["length"] = pulse_length_ns
    _md_iq["global_params"]["outer_folder"] = pl_folder
    _cfg_iq = meta_dict_to_config_1Q(_md_iq)

    chev_iq = RabiChevronIQ(
        QUBIT,
        _amp_vec_iq,
        _f_vec_iq,
        _chev_iq["shots"],
        pulse_type=PULSE_TYPE,
        meta_dict=_md_iq,
        outer_folder=pl_folder,
        path="bootstrap_IQ",
        suffix=f"ChevronIQ_{pl_label}",
        config=_cfg_iq,
        flat_path=True,
    )

    pl_IF_freq_offset = chev_iq.data["opt_detuning_Hz"]
    pl_amp_init = chev_iq.data["opt_amp"]
    print(
        f"  IQ bootstrap first π-pulse: detuning offset={pl_IF_freq_offset / u.MHz:.3f} MHz, "
        f"amp={pl_amp_init:.4f} V"
    )

    drag_results = []
    rb_error_rates, rb_error_rate_errs = [], []
    freq_xycal_rb_errors = []  # parallel to rb_error_rates; RB error on Freq-XY-cal frame
    beta_configs = []
    beta0_amp_this_pl = None
    pl_chevron_amp_max = min(HARDWARE_AMP_MAX, pl_amp_init * CHEVRON_AMP_MAX_HEADROOM)

    for beta in DRAG_beta_vec:
        beta_label = f"beta{beta:+.3f}".replace(".", "p")
        beta_start = time.time()
        print(f"\n  DRAG beta = {beta:.3f}  |  {datetime.now().strftime('%H:%M:%S')}")

        meta_dict = make_meta_dict(QUBIT)[QUBIT]
        meta_dict["DRAG_beta"] = float(beta)
        meta_dict["X90"]["length"] = pulse_length_ns
        meta_dict["Y90"]["length"] = pulse_length_ns
        meta_dict["X90"]["amp"] = pl_amp_init
        meta_dict["Y90"]["amp"] = pl_amp_init
        meta_dict["IF_freq"] += int(pl_IF_freq_offset)
        meta_dict["global_params"]["outer_folder"] = pl_folder

        # Avoid stale waveform/gate detuning when using APE as the IF correction target.
        meta_dict["detuning"] = 0
        for gate in ("X90", "Y90", "X180", "Y180"):
            if gate in meta_dict and isinstance(meta_dict[gate], dict) and "detuning" in meta_dict[gate]:
                meta_dict[gate]["detuning"] = 0

        config = meta_dict_to_config_1Q(meta_dict)

        # 2-level SS calibration — retry up to 10 times until fidelity > 0.70.
        max_retries_2L = 10
        two_level_success = False
        for retry_idx in range(max_retries_2L):
            SS = SingleShot1Q(
                QUBIT,
                experiment_dict["SS_Cal"]["shots"],
                meta_dict=meta_dict,
                outer_folder=pl_folder,
                path=beta_label,
                suffix="SS_Cal",
                repeats=experiment_dict["SS_Cal"]["number_pi_pulses"],
                config=config,
                confidence_threshold=experiment_dict["SS_Cal"]["ground_threshold"],
                flat_path=True,
            )
            print(f"    SS Cal — fidelity {SS.max_F:.3f}")
            if SS.max_F > 0.70:
                two_level_success = True
                break
            if retry_idx < max_retries_2L - 1:
                print("    Fidelity below 0.70, retrying...")

        if not two_level_success:
            print(f"    Failed 2-level SS calibration at DRAG beta = {beta:.3f}")
            drag_results.append((beta, np.nan, np.nan, np.nan, np.nan, np.nan,
                                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            rb_error_rates.append(np.nan)
            rb_error_rate_errs.append(np.nan)
            freq_xycal_rb_errors.append(np.nan)
            continue

        CALIB_PARAMS = SS.calib_params
        meta_dict["read_theta"] = SS.calib_params["read_theta"]
        config = meta_dict_to_config_1Q(meta_dict)

        # Chevron bootstrap + AAE/APE calibration sequence.
        config, cal_summary = _run_calibration(
            QUBIT,
            meta_dict,
            config,
            CALIB_PARAMS,
            PULSE_TYPE,
            experiment_dict,
            optimizer_convergence_params,
            beta_label,
            save_path=beta_label,
            chevron_amp_max=pl_chevron_amp_max,
        )

        cal_amp = float(meta_dict["X90"]["amp"])
        cal_IF_freq = int(meta_dict["IF_freq"])
        final_ape_resid_MHz = float(cal_summary.get("final_ape_residual_MHz", np.nan))
        last_ape_opt_MHz = float(cal_summary.get("last_ape_opt_MHz", np.nan))
        initial_ape_opt_MHz = float(cal_summary.get("initial_ape_opt_MHz", np.nan))
        freq_xycal_ape_resid_MHz = float(cal_summary.get("freq_xycal_ape_resid_MHz", np.nan))

        print(
            f"    Final X90 amp={cal_amp:.6f} V  "
            f"IF_freq={cal_IF_freq / u.MHz:.6f} MHz  "
            f"final_APE_resid={final_ape_resid_MHz:+.6f} MHz"
        )

        if beta0_amp_this_pl is None:
            beta0_amp_this_pl = cal_amp

        # 3-level SS calibration if needed for leakage RB or 3-state vanilla RB.
        calib_3L = None
        need_3L = RUN_LEAKAGE_RB or (RUN_RB1Q and RUN_RB_3STATE)
        if need_3L:
            max_retries_3L = 30
            three_level_success = False
            for retry_idx in range(max_retries_3L):
                SS3 = SingleShot1Q3Level(
                    QUBIT,
                    experiment_dict["SS_Cal_3L"]["shots"],
                    meta_dict=meta_dict,
                    outer_folder=pl_folder,
                    path=beta_label,
                    suffix="SS_Cal_3L",
                    config=config,
                    flat_path=True,
                )
                print(f"    3-level SS Cal — fidelity {SS3.max_F:.3f}")
                if SS3.max_F > 0.65:
                    three_level_success = True
                    break
                if retry_idx < max_retries_3L - 1:
                    print("    Fidelity below 0.65, retrying...")
            if three_level_success:
                calib_3L = SS3.calib_params
            else:
                print(f"    Failed 3 state cal at DRAG beta = {beta:.3f}")

        # Leakage RB.
        L1 = L1_err = L2 = L2_err = lambda1 = np.nan
        if RUN_LEAKAGE_RB:
            if calib_3L is None:
                print("    RUN_LEAKAGE_RB set but no 3-level cal available — leakage entries NaN")
            else:
                lrb_l1s, lrb_l1_errs, lrb_l2s, lrb_l2_errs = [], [], [], []
                for avg_idx in range(experiment_dict["Leakage_RB"]["num_averages"]):
                    rb = RB1Q_Leakage(
                        QUBIT,
                        experiment_dict["Leakage_RB"]["circuit_lengths"],
                        experiment_dict["Leakage_RB"]["num_circuits"],
                        experiment_dict["Leakage_RB"]["shots"],
                        calib_3L,
                        meta_dict=meta_dict,
                        outer_folder=pl_folder,
                        path=beta_label,
                        suffix=f"LRB_avg{avg_idx:02d}",
                        config=config,
                        flat_path=True,
                    )
                    _L1, _L1_err = rb.data["fit_data"]["state_2_L1"]
                    _L2, _L2_err = rb.data["fit_data"]["state_2_L2"]
                    lrb_l1s.append(_L1); lrb_l1_errs.append(_L1_err)
                    lrb_l2s.append(_L2); lrb_l2_errs.append(_L2_err)
                    print(
                        f"      LRB average {avg_idx + 1}/{experiment_dict['Leakage_RB']['num_averages']}: "
                        f"L1={_L1:.4e}, L2={_L2:.4e}"
                    )
                L1 = _safe_nanmean(lrb_l1s)
                L1_err = _safe_nanmean(lrb_l1_errs)
                L2 = _safe_nanmean(lrb_l2s)
                L2_err = _safe_nanmean(lrb_l2_errs)
                lambda1 = 1.0 - L1 - L2
                print(f"    LRB done — L1={L1:.4e} ± {L1_err:.4e}  L2={L2:.4e} ± {L2_err:.4e}")

        drag_results.append((
            beta, L1, L1_err, L2, L2_err, lambda1,
            cal_amp, cal_IF_freq, final_ape_resid_MHz, last_ape_opt_MHz,
            initial_ape_opt_MHz, freq_xycal_ape_resid_MHz,
        ))

        # Optional coherence measurements once before first RB.
        if RUN_COHERENCE_MEASUREMENTS and not coherence_measured:
            print("\n  Running coherence measurements (T1, T2E)...")
            t1_cfg = experiment_dict["T1"]
            t1_vec = np.linspace(t1_cfg["t_min"], t1_cfg["t_max"], t1_cfg["t_points"], dtype=int)
            _ = T1(
                QUBIT, t1_vec, t1_cfg["shots"], CALIB_PARAMS, outer_folder=pl_folder,
                path="coherence", suffix="T1", meta_dict=meta_dict, config=config,
                num_pulses=1,
            )
            t2e_cfg = experiment_dict["T2E"]
            t2e_vec = np.linspace(t2e_cfg["t_min"], t2e_cfg["t_max"], t2e_cfg["t_points"], dtype=int)
            _ = T2E(
                QUBIT, t2e_vec, t2e_cfg["shots"], CALIB_PARAMS, outer_folder=pl_folder,
                path="coherence", suffix="T2E", meta_dict=meta_dict, config=config,
            )
            coherence_measured = True

        # Vanilla RB.
        rb_error_rate_mean = np.nan
        rb_error_rate_std = np.nan
        if RUN_RB1Q:
            rb_3state_calib = calib_3L if (RUN_RB_3STATE and calib_3L is not None) else None
            if RUN_RB_3STATE and rb_3state_calib is None:
                print("    RUN_RB_3STATE set but no 3-level cal available — using 2-state RB readout")
            rb_fidelities = []
            for avg_idx in range(NUM_AVERAGES):
                rb1q = RB1Q(
                    QUBIT,
                    experiment_dict["RB1Q"]["circuit_lengths"],
                    experiment_dict["RB1Q"]["num_circuits"],
                    experiment_dict["RB1Q"]["shots"],
                    CALIB_PARAMS,
                    meta_dict=meta_dict,
                    outer_folder=pl_folder,
                    path=beta_label,
                    suffix=f"RB1Q_{beta_label}_avg{avg_idx:02d}",
                    config=config,
                    simulate=experiment_dict["RB1Q"]["simulate"],
                    flat_path=True,
                    calib_params_3state=rb_3state_calib,
                )
                _rb_fid = rb1q.data.get("gate_fidelity", float("nan"))
                rb_fidelities.append(_rb_fid)
                print(f"      RB average {avg_idx + 1}/{NUM_AVERAGES}: fidelity = {_rb_fid:.6f}")
            rb_fidelity_mean = _safe_nanmean(rb_fidelities)
            rb_fidelity_std = _safe_nanstd(rb_fidelities)
            rb_error_rate_mean = 1.0 - rb_fidelity_mean if np.isfinite(rb_fidelity_mean) else np.nan
            rb_error_rate_std = rb_fidelity_std
        else:
            print("    RUN_RB1Q=False — RB entries set to NaN")

        rb_error_rates.append(rb_error_rate_mean)
        rb_error_rate_errs.append(rb_error_rate_std)

        # FREQ_XYCAL_CHECK: RB on the Freq-XY-calibrated frame for a direct comparison
        # against the APE-cal RB error (same RB settings / readout cal).
        freq_xycal_rb_error = np.nan
        if FREQ_XYCAL_CHECK and RUN_RB1Q:
            xycal_meta_state = cal_summary.get("xycal_meta")
            if xycal_meta_state is not None:
                xycal_meta_rb = copy.deepcopy(xycal_meta_state)
                xycal_cfg_rb = meta_dict_to_config_1Q(xycal_meta_rb)
                print("    FREQ_XYCAL_CHECK: RB on Freq-XY-calibrated frame ...")
                freq_xycal_rb_error, _ = _run_rb_error_rate(
                    xycal_meta_rb, xycal_cfg_rb, CALIB_PARAMS, calib_3L,
                    pl_folder, beta_label, suffix_prefix=f"RB1Q_XYCAL_{beta_label}",
                )
                print(
                    f"      Freq-XY-cal RB error={freq_xycal_rb_error:.6e}  "
                    f"(APE-cal RB error={rb_error_rate_mean:.6e})"
                )
        freq_xycal_rb_errors.append(freq_xycal_rb_error)

        beta_configs.append({"meta": copy.deepcopy(meta_dict), "calib": copy.deepcopy(CALIB_PARAMS)})

        save_drag_summary(pl_folder, QUBIT, pulse_length_ns, RUN_LEAKAGE_RB,
                          drag_results, rb_error_rates, rb_error_rate_errs,
                          freq_xycal_rb_errors)

        beta_elapsed = time.time() - beta_start
        print(f"    beta={beta:.3f} complete in {beta_elapsed:.0f}s")

    if beta0_amp_this_pl is not None:
        prev_beta0_amp = beta0_amp_this_pl

    # Per-pulse-length summary and best beta.
    rb_err_arr = np.array(rb_error_rates)
    final_ape_resids = np.array([r[8] for r in drag_results])
    if np.any(np.isfinite(rb_err_arr)):
        best_idx = int(np.nanargmin(rb_err_arr))
    elif np.any(np.isfinite(final_ape_resids)):
        best_idx = int(np.nanargmin(np.abs(final_ape_resids)))
    else:
        best_idx = 0
    best = drag_results[best_idx]
    best_beta = best[0]
    rb_error_rate_mean = rb_error_rates[best_idx]
    rb_error_rate_std = rb_error_rate_errs[best_idx]

    drag_fig_path = save_drag_summary(pl_folder, QUBIT, pulse_length_ns, RUN_LEAKAGE_RB,
                                      drag_results, rb_error_rates, rb_error_rate_errs,
                                      freq_xycal_rb_errors)
    print(f"  DRAG sweep figure saved to {drag_fig_path}")
    print(
        f"\n  Best beta={best_beta:.3f}  RB error={rb_error_rate_mean:.6e} ± {rb_error_rate_std:.6e}  "
        f"final_APE_resid={best[8]:+.6f} MHz"
    )

    # Purity / unitarity RB at best beta.
    u_val = u_err = r_coh_gate = r_coh_err = r_incoh_gate = r_incoh_err = np.nan
    if RUN_PURITY_RB:
        if not RUN_RB1Q or not np.isfinite(rb_error_rate_mean):
            print("  RUN_PURITY_RB=True but no valid RB error for best beta — skipping purity budget")
        else:
            print(f"\n  Running purity RB at best beta={best_beta:.3f} ...")
            best_state = beta_configs[best_idx]
            best_meta = copy.deepcopy(best_state["meta"])
            best_calib = best_state["calib"]
            best_config = meta_dict_to_config_1Q(best_meta)

            prb = RB1Q_Purity(
                QUBIT,
                experiment_dict["Purity_RB"]["circuit_lengths"],
                experiment_dict["Purity_RB"]["num_circuits"],
                experiment_dict["Purity_RB"]["shots"],
                best_calib,
                meta_dict=best_meta,
                outer_folder=pl_folder,
                path=f"beta{best_beta:+.3f}".replace(".", "p"),
                suffix="PurityRB_best",
                config=best_config,
                flat_path=True,
            )
            u_val = prb.data.get("unitarity", np.nan)
            u_err = prb.data.get("unitarity_err", np.nan)
            p_dep = 1.0 - 2.0 * GATES_PER_CLIFFORD * rb_error_rate_mean
            if np.isfinite(u_val) and u_val > 0:
                sqrt_u = np.sqrt(u_val)
                r_coh_gate = max(sqrt_u - p_dep, 0.0) / 2.0 / GATES_PER_CLIFFORD
                r_incoh_gate = max(1.0 - sqrt_u, 0.0) / 2.0 / GATES_PER_CLIFFORD
                d_sqrt_u = (u_err / (2.0 * sqrt_u)) if (np.isfinite(u_err) and sqrt_u > 0) else 0.0
                d_p = 2.0 * GATES_PER_CLIFFORD * rb_error_rate_std if np.isfinite(rb_error_rate_std) else 0.0
                r_coh_err = np.sqrt(d_sqrt_u**2 + d_p**2) / 2.0 / GATES_PER_CLIFFORD
                r_incoh_err = d_sqrt_u / 2.0 / GATES_PER_CLIFFORD
                print(
                    f"    u={u_val:.6f}±{u_err:.6f}  p={p_dep:.6f}  "
                    f"r_coh={r_coh_gate:.3e}  r_incoh={r_incoh_gate:.3e}  (per gate)"
                )
            else:
                print("    Purity RB returned no valid unitarity; budget entries set to NaN.")

    sweep_results.append((
        pulse_length_ns, best_beta, best[1], best[2], best[3], best[4], best[5],
        best[6], best[7], rb_error_rate_mean, rb_error_rate_std, r_coh_gate,
        r_coh_err, r_incoh_gate, r_incoh_err, u_val, u_err, best[8],
    ))

    # Update global H5.
    with h5py.File(global_h5_path, "a") as f:
        n = len(f["pulse_lengths_ns"]) + 1
        for key in f.keys():
            f[key].resize((n,))
        f["pulse_lengths_ns"][-1] = pulse_length_ns
        f["best_beta"][-1] = abs(best_beta)
        f["rb_error_rate"][-1] = rb_error_rate_mean
        f["rb_error_rate_std"][-1] = rb_error_rate_std
        f["L1"][-1] = best[1]
        f["L1_err"][-1] = best[2]
        f["L2"][-1] = best[3]
        f["L2_err"][-1] = best[4]
        f["r_coh"][-1] = r_coh_gate
        f["r_coh_err"][-1] = r_coh_err
        f["r_incoh"][-1] = r_incoh_gate
        f["r_incoh_err"][-1] = r_incoh_err
        f["unitarity"][-1] = u_val
        f["unitarity_err"][-1] = u_err
        f["final_ape_residual_MHz"][-1] = best[8]
        f["best_amp"][-1] = best[6]
        f["best_IF_freq_Hz"][-1] = best[7]

    # Live/final summary plot.
    pls_list.append(pulse_length_ns)
    rb_errs_list.append(rb_error_rate_mean)
    rb_stds_list.append(rb_error_rate_std)
    l1s_list.append(best[1])
    l1_errs_list.append(best[2])
    l2s_list.append(best[3])
    l2_errs_list.append(best[4])
    lambda1s_list.append(best[5])
    ape_resid_list.append(best[8])

    pls_live = np.array(pls_list)
    rb_errs_live = np.array(rb_errs_list)
    rb_stds_live = np.array(rb_stds_list)
    l1s_live = np.array(l1s_list)
    l1_errs_live = np.array(l1_errs_list)
    l2s_live = np.array(l2s_list)
    l2_errs_live = np.array(l2_errs_list)
    one_minus_lambda1_live = 1.0 - np.array(lambda1s_list)
    ape_resid_live = np.array(ape_resid_list)

    for a in axes:
        a.clear()

    for a, ys, yerrs, ylabel, ylim_bottom in zip(
        axes[:3],
        [l1s_live, l2s_live, one_minus_lambda1_live],
        [l1_errs_live, l2_errs_live, np.zeros(len(one_minus_lambda1_live))],
        ["L1  (leakage per Clifford)", "L2  (leakage per Clifford)", "1 - λ1"],
        [1e-6, None, 1e-5],
    ):
        a.errorbar(pls_live, ys, yerr=yerrs, fmt="o-", capsize=4, markersize=8)
        a.set_yscale("log")
        a.set_ylim(bottom=ylim_bottom)
        a.set_ylabel(ylabel, fontsize=12)
        a.grid(True, which="both", alpha=0.3)

    axes[3].errorbar(pls_live, rb_errs_live, yerr=rb_stds_live, fmt="o-", capsize=4,
                     markersize=8, label="Total infidelity (RB)")
    axes[3].errorbar(pls_live, l1s_live, yerr=l1_errs_live, fmt="s-", capsize=4,
                     markersize=8, label="Leakage infidelity (L1)")
    axes[3].set_yscale("log")
    axes[3].set_ylim(bottom=1e-5)
    axes[3].set_ylabel("Error rate", fontsize=12)
    axes[3].legend(fontsize=11)
    axes[3].grid(True, which="both", alpha=0.3)

    axes[4].plot(pls_live, ape_resid_live, "o-", label="final APE residual")
    axes[4].axhline(0, color="k", lw=1, alpha=0.6)
    axes[4].axhline(+FINAL_APE_RESIDUAL_TARGET_MHZ, color="k", ls="--", lw=1, alpha=0.4)
    axes[4].axhline(-FINAL_APE_RESIDUAL_TARGET_MHZ, color="k", ls="--", lw=1, alpha=0.4)
    axes[4].set_ylabel("APE resid [MHz]", fontsize=12)
    axes[4].set_xlabel("Gate time (ns)", fontsize=12)
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(f"{QUBIT} — Leakage, RB, and APE Residual vs Gate Time")
    fig.tight_layout()
    plot_path = os.path.join(parent_folder, f"{QUBIT}_summary_vs_gate_time.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"  Summary plot saved to {plot_path}")

    # Error-budget plot.
    if RUN_PURITY_RB:
        with h5py.File(global_h5_path, "r") as f:
            pls_b = f["pulse_lengths_ns"][:]
            rb_b = f["rb_error_rate"][:]
            rb_b_err = f["rb_error_rate_std"][:]
            l1_b = f["L1"][:]
            l1_b_err = f["L1_err"][:]
            rcoh_b = f["r_coh"][:]
            rcoh_b_err = f["r_coh_err"][:]
            rincoh_b = f["r_incoh"][:]
            rincoh_b_err = f["r_incoh_err"][:]

        fig_bud, ax_bud = plt.subplots(1, 1, figsize=(8, 6))
        ax_bud.errorbar(pls_b, rb_b, yerr=rb_b_err, fmt="o-", capsize=5,
                        label="Total error (RB)", linewidth=2.5, markersize=9)
        ax_bud.errorbar(pls_b, l1_b, yerr=l1_b_err, fmt="s-", capsize=5,
                        label="Leakage error (L1)", linewidth=2.5, markersize=9)
        ax_bud.errorbar(pls_b, rcoh_b, yerr=rcoh_b_err, fmt="^-", capsize=5,
                        label="Coherent error (PRB)", linewidth=2.5, markersize=9)
        ax_bud.errorbar(pls_b, rincoh_b, yerr=rincoh_b_err, fmt="v-", capsize=5,
                        label="Incoherent error (PRB)", linewidth=2.5, markersize=9)
        ax_bud.set_yscale("log")
        ax_bud.set_ylim(bottom=1e-6)
        ax_bud.set_ylabel("Error rate (per gate)", fontsize=13)
        ax_bud.set_xlabel("Gate time (ns)", fontsize=13)
        ax_bud.legend(fontsize=11, loc="best")
        ax_bud.grid(True, which="both", alpha=0.3)
        ax_bud.set_title(f"{QUBIT} — Error Budget vs Gate Time", fontsize=13)
        fig_bud.tight_layout()
        budget_path = os.path.join(parent_folder, f"{QUBIT}_error_budget_vs_gate_time.png")
        fig_bud.savefig(budget_path, bbox_inches="tight", dpi=150)
        print(f"  Error-budget plot saved to {budget_path}")
        plt.close(fig_bud)

    pl_elapsed = time.time() - pl_start
    print(f"\n  Pulse length {pulse_length_ns} ns complete in {pl_elapsed:.0f}s")
    print(
        f"  Best: beta={best[0]:.3f}  L1={best[1]:.4e}  L2={best[3]:.4e}  "
        f"λ1={best[5]:.4f}  amp={best[6]:.6f} V  "
        f"IF_freq={best[7] / u.MHz:.6f} MHz  "
        f"RB_error={rb_error_rate_mean:.6e} ± {rb_error_rate_std:.6e}  "
        f"APE_resid={best[8]:+.6f} MHz"
    )

# Sweep complete.
total_elapsed = time.time() - sweep_start
print(f"\n{'=' * 60}")
print(f"All pulse lengths complete in {total_elapsed:.0f}s.")
print(f"Summary saved to: {global_h5_path}")
if plot_path is not None:
    print(f"Live plot saved to: {plot_path}")
print(f"{'=' * 60}")

plt.close(fig)
