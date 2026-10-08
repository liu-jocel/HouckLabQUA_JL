"""
APE iterative correction runner: APE -> apply -> apply residual -> final sanity.

Purpose
-------
For a frozen-IF DRAG-beta sweep, measure the scalar phase / effective-detuning
error with APE, apply the APE correction to IF_freq, measure the residual, apply
that residual, and run one final sanity APE. This is meant to prepare clean
metadata/configs for subsequent RB and leakage-RB without running Frequency_XY
again after the IF freeze point.

Required APE implementation
---------------------------
Use the keep-phase APE class:
    LabCode/Experiments/Gates/m_amplified_phase_error_keep_phase.py

That APE implementation uses update_frequency(..., keep_phase=True) when
entering/leaving the detuned APE block. Without this, the APE correction can have
wrong sign/scale because frequency updates inject phase discontinuities.

High-level flow
---------------
1. Calibrate beta=0 normally, optionally including AAE + Frequency_XY. This is
   allowed only BEFORE the freeze point.
2. Freeze IF_freq and the beta=0 amplitude.
3. For each DRAG beta:
   a. Reset meta_dict to frozen IF / frozen amp and selected beta.
   b. Optionally run AAE only. No Frequency_XY after this point.
   c. Run staged APE: broad then narrow.
   d. Apply IF_freq += APE_APPLY_SIGN * ape_opt.
   e. Run residual/sanity APE around zero.
   f. Apply IF_freq += APE_APPLY_SIGN * residual.
   g. Run final sanity APE around zero.
4. Save H5, CSV, PNG, and a per-beta final-state table.

Interpretation
--------------
Use the final IF_freq for RB/L1. If the final APE residual is below roughly
0.03-0.05 MHz for a 20 ns X90, the residual scalar phase contribution should be
around or below ~1e-5 to a few x 1e-5, depending on the measured APE slope.
"""

import os
import csv
import copy
import inspect
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

plt.ion()

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Rabi.m_Rabi_Chevron_IQ import RabiChevronIQ
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.Gates.m_amplified_phase_error import APE

# =============================================================================
# USER SETTINGS
# =============================================================================
QUBIT = "q6"
PULSE_TYPE = "X90"
PULSE_LENGTH_NS = 20

# Start with the betas you are actively debugging; expand after the flow is stable.
DRAG_BETA_VEC = np.array([0.0, 0.3, 0.6, 0.8, 1.2])

# --- beta=0 bootstrap before IF freeze ---------------------------------------
DO_BETA0_IQ_CHEVRON = True
DO_BETA0_SS_CHEVRON = True
DO_BETA0_AAE_FREQ_LOOP = True  # allowed ONLY before freeze
BETA0_CAL_ITERS = 2

# --- per-beta calibration after IF freeze ------------------------------------
DO_AAE_PER_BETA = True  # amplitude only; no Frequency_XY after freeze
AAE_N_VEC = np.arange(1, 61, 1)
AAE_SHOTS = 300

# --- APE settings -------------------------------------------------------------
APE_SHOTS = 250
APE_CENTER_MODE = "fixed"
APE_CENTER = (0.0, 0.0)
APE_MIN_RADIUS = 0.15
APE_LOCAL_FIT_POINTS = 5
APE_ALLOW_EXTRAPOLATION = False
KEEP_PHASE_DURING_DETUNING = True

# Broad APE: only used to find a bracketed zero.
APE_BROAD_SPAN_MHZ = 8.0  # [-4,+4] MHz around current IF
APE_BROAD_POINTS = 17
APE_BROAD_N_VEC = np.arange(1, 41, 1)

# Narrow APE: centered on broad result; this is the primary correction.
APE_NARROW_SPAN_MHZ = 0.75
APE_NARROW_POINTS = 16
APE_NARROW_N_VEC = np.arange(1, 81, 1)

# Residual APE after first correction. Wide enough to catch small sign/path
# mistakes but narrow enough to get useful local slope.
APE_RESIDUAL_SPAN_MHZ = 2.0  # [-1,+1] MHz around corrected IF
APE_RESIDUAL_POINTS = 13
APE_RESIDUAL_N_VEC = np.arange(1, 81, 1)

# Final sanity APE after applying residual.
APE_FINAL_SANITY_SPAN_MHZ = 1.0  # [-0.5,+0.5] MHz around twice-corrected IF
APE_FINAL_SANITY_POINTS = 11
APE_FINAL_SANITY_N_VEC = np.arange(1, 81, 1)

# Correction convention confirmed by keep_phase APE sanity triad:
#     IF_freq <- IF_freq + optimal_detuning
APE_APPLY_SIGN = +1.0

# Apply the residual even if small? For production, it may be better to avoid
# chasing fit noise. For this diagnostic, apply if finite and above threshold.
RESIDUAL_APPLY_THRESHOLD_MHZ = 0.0
FINAL_RESIDUAL_TARGET_MHZ = 0.05  # print PASS if |final residual| <= this

# Optional sign/path test offset applied AFTER beta=0 freeze; leave 0 normally.
MANUAL_FROZEN_IF_OFFSET_MHZ = 0.0

# --- readout / chevron settings ----------------------------------------------
SS_SHOTS = 1000
SS_GROUND_THRESHOLD = 0.70

CHEVRON_IQ_SHOTS = 21
CHEVRON_SS_SHOTS = 25
CHEVRON_AMP_MIN = 0.0
CHEVRON_AMP_MAX = 0.35
CHEVRON_AMP_POINTS = 31
CHEVRON_FREQ_SPAN = 21 * u.MHz
CHEVRON_IQ_FREQ_POINTS = 11
CHEVRON_SS_FREQ_POINTS = 31

FREQ_CAL_SHOTS = 100
FREQ_CAL_POINTS = 401
FREQ_CAL_SPANS = [5 * u.MHz, 1 * u.MHz]
FREQ_CAL_SEQUENCE = [("X90", "I"), ("Y90", "I"), ("X90", "Y90"), ("Y90", "X90")]


# =============================================================================
# HELPERS
# =============================================================================
def _beta_label(beta):
    return f"beta{beta:+.3f}".replace(".", "p")


def _f_vec(center_Hz, span_MHz, points):
    half_Hz = int(0.5 * span_MHz * u.MHz)
    return np.linspace(
        int(center_Hz) - half_Hz, int(center_Hz) + half_Hz, points, dtype=int
    )


def _fresh_meta(qubit, pulse_length_ns, beta, outer_folder):
    md = make_meta_dict(qubit)[qubit]
    md["DRAG_beta"] = float(beta)
    md["X90"]["length"] = int(pulse_length_ns)
    md["Y90"]["length"] = int(pulse_length_ns)
    md["global_params"]["outer_folder"] = outer_folder

    # This APE/RB pass treats the correction as an element IF correction. Keep the
    # waveform detuning term fixed at zero unless you are deliberately testing the
    # waveform generator's internal detuning feature.
    md["detuning"] = 0
    for gate in ("X90", "Y90", "X180", "Y180"):
        if gate in md and isinstance(md[gate], dict) and "detuning" in md[gate]:
            md[gate]["detuning"] = 0
    return md


def _run_ss(qubit, meta_dict, config, outer_folder, path, suffix="SS_Cal"):
    ss = SingleShot1Q(
        qubit,
        SS_SHOTS,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=path,
        suffix=suffix,
        repeats=1,
        config=config,
        confidence_threshold=SS_GROUND_THRESHOLD,
        flat_path=True,
    )
    print(f"    SS fidelity = {ss.max_F:.3f}")
    meta_dict["read_theta"] = ss.calib_params["read_theta"]
    return ss.calib_params, meta_dict_to_config_1Q(meta_dict)


def _bootstrap_beta0(qubit, pulse_length_ns, outer_folder):
    print("\n=== beta=0 bootstrap/calibration ===")
    md = _fresh_meta(qubit, pulse_length_ns, beta=0.0, outer_folder=outer_folder)
    cfg = meta_dict_to_config_1Q(md)

    if DO_BETA0_IQ_CHEVRON:
        print("  IQ chevron bootstrap...")
        amp_vec = np.linspace(CHEVRON_AMP_MIN, CHEVRON_AMP_MAX, CHEVRON_AMP_POINTS)
        f_vec = _f_vec(0, CHEVRON_FREQ_SPAN / u.MHz, CHEVRON_IQ_FREQ_POINTS)
        ch_iq = RabiChevronIQ(
            qubit,
            amp_vec,
            f_vec,
            CHEVRON_IQ_SHOTS,
            pulse_type=PULSE_TYPE,
            meta_dict=md,
            outer_folder=outer_folder,
            path="beta0_bootstrap",
            suffix="ChevronIQ_beta0",
            config=cfg,
            flat_path=True,
        )
        md["IF_freq"] += int(ch_iq.data["opt_detuning_Hz"])
        md["X90"]["amp"] = ch_iq.data["opt_amp"]
        md["Y90"]["amp"] = ch_iq.data["opt_amp"]
        cfg = meta_dict_to_config_1Q(md)
        print(
            f"    IQ opt: df={ch_iq.data['opt_detuning_Hz'] / u.MHz:+.3f} MHz, "
            f"amp={ch_iq.data['opt_amp']:.6f}"
        )

    calib, cfg = _run_ss(qubit, md, cfg, outer_folder, path="beta0_bootstrap")

    if DO_BETA0_SS_CHEVRON:
        print("  SS chevron refinement...")
        amp_max = min(CHEVRON_AMP_MAX, max(0.02, md["X90"]["amp"] * 1.35))
        amp_vec = np.linspace(CHEVRON_AMP_MIN, amp_max, CHEVRON_AMP_POINTS)
        f_vec = _f_vec(0, CHEVRON_FREQ_SPAN / u.MHz, CHEVRON_SS_FREQ_POINTS)
        ch = RabiChevronSS(
            qubit,
            amp_vec,
            f_vec,
            CHEVRON_SS_SHOTS,
            calib,
            num_pi_pulses=1,
            pulse_type=PULSE_TYPE,
            meta_dict=md,
            outer_folder=outer_folder,
            path="beta0_bootstrap",
            suffix="ChevronSS_beta0",
            config=cfg,
            flat_path=True,
        )
        md["IF_freq"] += int(ch.data["opt_detuning_Hz"])
        md["X90"]["amp"] = ch.data["opt_amp"]
        md["Y90"]["amp"] = ch.data["opt_amp"]
        cfg = meta_dict_to_config_1Q(md)
        print(
            f"    SS opt: df={ch.data['opt_detuning_Hz'] / u.MHz:+.3f} MHz, "
            f"amp={ch.data['opt_amp']:.6f}"
        )

    if DO_BETA0_AAE_FREQ_LOOP:
        for i in range(BETA0_CAL_ITERS):
            print(f"  beta=0 AAE/frequency loop {i + 1}/{BETA0_CAL_ITERS}...")
            aae = AAE(
                qubit,
                AAE_N_VEC,
                AAE_SHOTS,
                calib_params=calib,
                pulse_type=PULSE_TYPE,
                meta_dict=md,
                outer_folder=outer_folder,
                path="beta0_bootstrap",
                suffix=f"AAE_beta0_iter{i:02d}",
                config=cfg,
                flat_path=True,
            )
            amp_err = aae.data["amplitude_error"]
            md["X90"]["amp"] += amp_err
            md["Y90"]["amp"] += amp_err
            cfg = meta_dict_to_config_1Q(md)

            # Frequency_XY is allowed here only because this is before the freeze point.
            f_span = FREQ_CAL_SPANS[min(i, len(FREQ_CAL_SPANS) - 1)]
            f_vec = _f_vec(0, f_span / u.MHz, FREQ_CAL_POINTS)
            fcal = Frequency_Calibration_XY(
                qubit,
                f_vec,
                FREQ_CAL_SEQUENCE,
                FREQ_CAL_SHOTS,
                calib,
                num_repeats=1,
                meta_dict=md,
                outer_folder=outer_folder,
                path="beta0_bootstrap",
                suffix=f"FreqCal_beta0_iter{i:02d}",
                config=cfg,
                flat_path=True,
            )
            df_MHz = fcal.data["optimal_detuning_MHz"]
            md["IF_freq"] += int(df_MHz * u.MHz)
            cfg = meta_dict_to_config_1Q(md)
            print(f"    amp_err={amp_err:+.3e}, freq_corr={df_MHz:+.4f} MHz")

    print("\n=== FREEZE POINT ===")
    print(f"  frozen IF_freq = {md['IF_freq'] / u.MHz:.6f} MHz")
    print(f"  frozen X90 amp = {md['X90']['amp']:.6f}")
    return md, cfg, calib


def _run_aae_only(qubit, meta_dict, config, calib, outer_folder, path, suffix):
    aae = AAE(
        qubit,
        AAE_N_VEC,
        AAE_SHOTS,
        calib_params=calib,
        pulse_type=PULSE_TYPE,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=path,
        suffix=suffix,
        config=config,
        flat_path=True,
    )
    amp_err = float(aae.data["amplitude_error"])
    meta_dict["X90"]["amp"] += amp_err
    meta_dict["Y90"]["amp"] += amp_err
    config = meta_dict_to_config_1Q(meta_dict)
    print(f"    AAE amp_err={amp_err:+.3e}; new amp={meta_dict['X90']['amp']:.6f}")
    return amp_err, config


def _check_ape_signature():
    sig = inspect.signature(APE.__init__)
    required = [
        "center",
        "center_mode",
        "min_radius",
        "local_fit_points",
        "allow_extrapolation",
        "keep_phase_during_detuning",
    ]
    missing = [k for k in required if k not in sig.parameters]
    if missing:
        raise RuntimeError(
            "Imported APE class is not the keep-phase robust version. Missing kwargs: "
            f"{missing}. Import path is APE.__module__={APE.__module__}."
        )


def _run_ape(qubit, meta_dict, config, calib, outer_folder, path, suffix, f_vec, N_vec):
    _check_ape_signature()
    print(
        f"\nAPE call {suffix}: "
        f"IF_freq={meta_dict['IF_freq'] / u.MHz:.6f} MHz, "
        f"f_vec=[{np.min(f_vec) / u.MHz:+.3f}, {np.max(f_vec) / u.MHz:+.3f}] MHz, "
        f"N=[{np.min(N_vec)}, {np.max(N_vec)}], points={len(f_vec)}x{len(N_vec)}"
    )
    ape = APE(
        qubit,
        f_vec,
        N_vec,
        APE_SHOTS,
        calib,
        pulse_type=PULSE_TYPE,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=path,
        suffix=suffix,
        config=config,
        center=APE_CENTER,
        center_mode=APE_CENTER_MODE,
        min_radius=APE_MIN_RADIUS,
        local_fit_points=APE_LOCAL_FIT_POINTS,
        allow_extrapolation=APE_ALLOW_EXTRAPOLATION,
        keep_phase_during_detuning=KEEP_PHASE_DURING_DETUNING,
    )
    opt = float(ape.data.get("optimal_detuning_MHz", np.nan))
    fit_slope = float(ape.data.get("fit_slope_rad_per_Hz", np.nan))
    exp_slope = float(ape.data.get("expected_slope_rad_per_Hz", np.nan))
    ratio = (
        fit_slope / exp_slope if np.isfinite(exp_slope) and exp_slope != 0 else np.nan
    )
    print(
        f"APE result {suffix}: opt={opt:+.6f} MHz, "
        f"bracketed={ape.data.get('zero_bracketed', False)}, "
        f"status={ape.data.get('fit_status', 'unknown')}, "
        f"slope/expected={ratio:.3f}"
    )
    return ape


def _ape_fields(ape, prefix=""):
    opt_MHz = float(ape.data.get("optimal_detuning_MHz", np.nan))
    opt_Hz = float(ape.data.get("optimal_detuning_Hz", np.nan))
    fit_slope_Hz = float(ape.data.get("fit_slope_rad_per_Hz", np.nan))
    exp_slope_Hz = float(ape.data.get("expected_slope_rad_per_Hz", np.nan))
    ratio = (
        fit_slope_Hz / exp_slope_Hz
        if np.isfinite(exp_slope_Hz) and exp_slope_Hz != 0
        else np.nan
    )
    return {
        f"{prefix}opt_MHz": opt_MHz,
        f"{prefix}opt_Hz": opt_Hz,
        f"{prefix}zero_bracketed": int(bool(ape.data.get("zero_bracketed", False))),
        f"{prefix}fit_status": str(ape.data.get("fit_status", "unknown")),
        f"{prefix}slope_rad_per_MHz": fit_slope_Hz * 1e6
        if np.isfinite(fit_slope_Hz)
        else np.nan,
        f"{prefix}expected_slope_rad_per_MHz": exp_slope_Hz * 1e6
        if np.isfinite(exp_slope_Hz)
        else np.nan,
        f"{prefix}slope_ratio": ratio,
    }


def _run_ape_staged(qubit, meta_dict, config, calib, outer_folder, path, beta_label):
    broad_f = _f_vec(0, APE_BROAD_SPAN_MHZ, APE_BROAD_POINTS)
    ape_broad = _run_ape(
        qubit,
        meta_dict,
        config,
        calib,
        outer_folder,
        path,
        suffix=f"APE_broad_{beta_label}",
        f_vec=broad_f,
        N_vec=APE_BROAD_N_VEC,
    )
    broad_Hz = ape_broad.data.get("optimal_detuning_Hz", np.nan)
    if not np.isfinite(broad_Hz):
        print(
            "    Broad APE did not bracket a zero; skipping narrow/apply for this beta."
        )
        return ape_broad, None

    narrow_f = _f_vec(broad_Hz, APE_NARROW_SPAN_MHZ, APE_NARROW_POINTS)
    ape_narrow = _run_ape(
        qubit,
        meta_dict,
        config,
        calib,
        outer_folder,
        path,
        suffix=f"APE_narrow_{beta_label}",
        f_vec=narrow_f,
        N_vec=APE_NARROW_N_VEC,
    )
    return ape_narrow, ape_broad


def _apply_ape_correction(meta_dict, config, opt_MHz, label):
    old_IF = int(meta_dict["IF_freq"])
    if not np.isfinite(opt_MHz):
        print(f"    {label}: no finite correction to apply")
        return config, 0, old_IF
    delta_IF = int(APE_APPLY_SIGN * opt_MHz * u.MHz)
    meta_dict["IF_freq"] = old_IF + delta_IF
    config = meta_dict_to_config_1Q(meta_dict)
    print("\n" + "=" * 70)
    print(f"{label}")
    print(f"correction opt:     {opt_MHz:+.6f} MHz")
    print(f"APE_APPLY_SIGN:     {APE_APPLY_SIGN:+.1f}")
    print(f"old IF:             {old_IF / u.MHz:.6f} MHz")
    print(f"delta IF applied:   {delta_IF / u.MHz:+.6f} MHz")
    print(f"new IF:             {meta_dict['IF_freq'] / u.MHz:.6f} MHz")
    print("=" * 70 + "\n")
    return config, delta_IF, old_IF


def _write_outputs(parent_folder, rows):
    h5_path = os.path.join(parent_folder, f"{QUBIT}_APE_iterative_apply_summary.h5")
    csv_path = os.path.join(parent_folder, f"{QUBIT}_APE_iterative_apply_summary.csv")

    # Union of all row keys in insertion-friendly order.
    ordered_keys = [
        "beta",
        "amp_final",
        "frozen_IF_Hz",
        "initial_IF_Hz",
        "IF_after_first_Hz",
        "IF_final_Hz",
        "main_opt_MHz",
        "first_delta_IF_MHz",
        "residual1_opt_MHz",
        "residual_delta_IF_MHz",
        "final_residual_MHz",
        "pass_final_target",
        "main_slope_ratio",
        "residual1_slope_ratio",
        "final_slope_ratio",
        "main_zero_bracketed",
        "residual1_zero_bracketed",
        "final_zero_bracketed",
        "main_fit_status",
        "residual1_fit_status",
        "final_fit_status",
        "aae_amp_err",
    ]
    all_keys = []
    for k in ordered_keys:
        if any(k in r for r in rows):
            all_keys.append(k)
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in all_keys})

    # Store string fields as variable-length UTF8; numeric as float arrays where possible.
    with h5py.File(h5_path, "w") as f:
        f.attrs["qubit"] = QUBIT
        f.attrs["pulse_type"] = PULSE_TYPE
        f.attrs["pulse_length_ns"] = PULSE_LENGTH_NS
        f.attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["ape_apply_sign"] = APE_APPLY_SIGN
        f.attrs["final_residual_target_MHz"] = FINAL_RESIDUAL_TARGET_MHZ
        grp = f.create_group("summary")
        for k in all_keys:
            vals = [r.get(k, np.nan) for r in rows]
            if any(isinstance(v, str) for v in vals):
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset(
                    k, data=np.array([str(v) for v in vals], dtype=object), dtype=dt
                )
            else:
                grp.create_dataset(k, data=np.array(vals, dtype=float))

    fig_path = _plot_summary(parent_folder, rows)
    return h5_path, csv_path, fig_path


def _plot_summary(parent_folder, rows):
    if not rows:
        return None
    beta = np.array([r.get("beta", np.nan) for r in rows], dtype=float)
    main = np.array([r.get("main_opt_MHz", np.nan) for r in rows], dtype=float)
    res1 = np.array([r.get("residual1_opt_MHz", np.nan) for r in rows], dtype=float)
    final = np.array([r.get("final_residual_MHz", np.nan) for r in rows], dtype=float)
    total_corr = np.array(
        [
            r.get("first_delta_IF_MHz", np.nan) + r.get("residual_delta_IF_MHz", np.nan)
            for r in rows
        ],
        dtype=float,
    )
    main_ratio = np.array(
        [r.get("main_slope_ratio", np.nan) for r in rows], dtype=float
    )
    final_ratio = np.array(
        [r.get("final_slope_ratio", np.nan) for r in rows], dtype=float
    )

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 11), sharex=True)

    axes[0].plot(beta, main, "o-", label="main APE opt")
    axes[0].axhline(0, color="k", lw=1, alpha=0.5)
    axes[0].set_ylabel("Initial APE opt [MHz]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(beta, res1, "o-", label="after first apply")
    axes[1].plot(beta, final, "s-", label="final after residual apply")
    axes[1].axhline(0, color="k", lw=1, alpha=0.5)
    axes[1].axhline(+FINAL_RESIDUAL_TARGET_MHZ, color="k", lw=0.8, ls="--", alpha=0.4)
    axes[1].axhline(-FINAL_RESIDUAL_TARGET_MHZ, color="k", lw=0.8, ls="--", alpha=0.4)
    axes[1].set_ylabel("APE residual [MHz]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(beta, total_corr, "o-", label="total IF correction")
    axes[2].axhline(0, color="k", lw=1, alpha=0.5)
    axes[2].set_ylabel("Total applied IF shift [MHz]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(beta, main_ratio, "o-", label="main slope/expected")
    axes[3].plot(beta, final_ratio, "s-", label="final slope/expected")
    axes[3].axhline(1, color="k", lw=1, alpha=0.5)
    axes[3].axhline(-1, color="k", lw=1, alpha=0.3)
    axes[3].set_ylabel("slope / expected")
    axes[3].set_xlabel("DRAG beta")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.suptitle(f"{QUBIT} {PULSE_LENGTH_NS} ns — APE apply + residual apply")
    fig.tight_layout()
    fig_path = os.path.join(parent_folder, f"{QUBIT}_APE_iterative_apply_summary.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    base_meta = make_meta_dict(QUBIT)[QUBIT]
    outer_base = base_meta["global_params"]["outer_folder"]
    day_dir = os.path.join(
        outer_base, QUBIT, f"{QUBIT}_{datetime.now().strftime('%Y_%m_%d')}"
    )
    os.makedirs(day_dir, exist_ok=True)
    parent = os.path.join(
        day_dir,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_APE_iterative_apply_{PULSE_LENGTH_NS}ns",
    )
    os.makedirs(parent, exist_ok=True)
    print(f"Output folder: {parent}")
    print(f"Using APE from: {APE.__module__}")
    _check_ape_signature()

    beta0_meta, beta0_cfg, calib = _bootstrap_beta0(QUBIT, PULSE_LENGTH_NS, parent)
    frozen_IF_before_manual = int(beta0_meta["IF_freq"])
    frozen_IF = int(frozen_IF_before_manual + MANUAL_FROZEN_IF_OFFSET_MHZ * u.MHz)
    frozen_amp = float(beta0_meta["X90"]["amp"])

    print("\n" + "=" * 70)
    print("FROZEN STATE")
    print(f"beta0 calibrated IF:     {frozen_IF_before_manual / u.MHz:.6f} MHz")
    print(f"manual frozen IF offset: {MANUAL_FROZEN_IF_OFFSET_MHZ:+.6f} MHz")
    print(f"frozen IF used:          {frozen_IF / u.MHz:.6f} MHz")
    print(f"frozen X90 amp:          {frozen_amp:.6f}")
    print(f"APE apply sign:          {APE_APPLY_SIGN:+.1f}")
    print("No Frequency_XY will be run after this point.")
    print("=" * 70 + "\n")

    rows = []

    for beta in DRAG_BETA_VEC:
        label = _beta_label(beta)
        print("\n" + "#" * 78)
        print(f"DRAG beta = {beta:+.3f}")
        print("#" * 78)

        beta_path = label
        md = _fresh_meta(QUBIT, PULSE_LENGTH_NS, beta=beta, outer_folder=parent)
        md["IF_freq"] = int(frozen_IF)
        md["X90"]["amp"] = frozen_amp
        md["Y90"]["amp"] = frozen_amp
        md["read_theta"] = beta0_meta.get("read_theta", md.get("read_theta", 0))
        cfg = meta_dict_to_config_1Q(md)

        row = {
            "beta": float(beta),
            "frozen_IF_Hz": float(frozen_IF),
            "initial_IF_Hz": float(md["IF_freq"]),
            "aae_amp_err": np.nan,
        }

        if DO_AAE_PER_BETA:
            amp_err, cfg = _run_aae_only(
                QUBIT, md, cfg, calib, parent, beta_path, f"AAE_{label}"
            )
            row["aae_amp_err"] = amp_err

        row["amp_final"] = float(md["X90"]["amp"])

        # 1) Main staged APE: broad -> narrow.
        print("\n--- Main APE: broad -> narrow ---")
        ape_main, ape_broad = _run_ape_staged(
            QUBIT, md, cfg, calib, parent, beta_path, label
        )
        row.update(_ape_fields(ape_main, prefix="main_"))
        row["main_opt_MHz"] = row.pop("main_opt_MHz")  # explicit for readability
        main_opt = row["main_opt_MHz"]

        if not np.isfinite(main_opt):
            print(
                f"    beta={beta:+.3f}: main APE failed; skipping apply/residual/final sanity"
            )
            row["pass_final_target"] = 0
            rows.append(row)
            _write_outputs(parent, rows)
            continue

        # 2) Apply main correction.
        cfg, delta1_Hz, old_if0 = _apply_ape_correction(
            md, cfg, main_opt, "APPLY MAIN APE CORRECTION"
        )
        row["first_delta_IF_MHz"] = delta1_Hz / u.MHz
        row["IF_after_first_Hz"] = float(md["IF_freq"])

        # 3) Residual APE after first correction.
        print("\n--- Residual APE after first apply ---")
        residual_f = _f_vec(0, APE_RESIDUAL_SPAN_MHZ, APE_RESIDUAL_POINTS)
        ape_resid = _run_ape(
            QUBIT,
            md,
            cfg,
            calib,
            parent,
            beta_path,
            suffix=f"APE_residual_after_first_apply_{label}",
            f_vec=residual_f,
            N_vec=APE_RESIDUAL_N_VEC,
        )
        resid_fields = _ape_fields(ape_resid, prefix="residual1_")
        row.update(resid_fields)
        residual_opt = row["residual1_opt_MHz"]

        # 4) Apply residual correction.
        delta2_Hz = 0
        if (
            np.isfinite(residual_opt)
            and abs(residual_opt) >= RESIDUAL_APPLY_THRESHOLD_MHZ
        ):
            cfg, delta2_Hz, old_if1 = _apply_ape_correction(
                md, cfg, residual_opt, "APPLY RESIDUAL APE CORRECTION"
            )
        else:
            print(
                f"    Residual not applied: residual={residual_opt:+.6f} MHz, "
                f"threshold={RESIDUAL_APPLY_THRESHOLD_MHZ:.6f} MHz"
            )
        row["residual_delta_IF_MHz"] = delta2_Hz / u.MHz
        row["IF_final_Hz"] = float(md["IF_freq"])

        # 5) Final sanity APE after residual correction.
        print("\n--- Final sanity APE after residual apply ---")
        final_f = _f_vec(0, APE_FINAL_SANITY_SPAN_MHZ, APE_FINAL_SANITY_POINTS)
        ape_final = _run_ape(
            QUBIT,
            md,
            cfg,
            calib,
            parent,
            beta_path,
            suffix=f"APE_final_sanity_{label}",
            f_vec=final_f,
            N_vec=APE_FINAL_SANITY_N_VEC,
        )
        final_fields = _ape_fields(ape_final, prefix="final_")
        row.update(final_fields)
        row["final_residual_MHz"] = row["final_opt_MHz"]
        row["pass_final_target"] = int(
            np.isfinite(row["final_residual_MHz"])
            and abs(row["final_residual_MHz"]) <= FINAL_RESIDUAL_TARGET_MHZ
        )

        print("\n" + "-" * 70)
        print(f"BETA {beta:+.3f} SUMMARY")
        print(f"main opt:          {main_opt:+.6f} MHz")
        print(f"residual after 1:  {residual_opt:+.6f} MHz")
        print(f"final residual:    {row['final_residual_MHz']:+.6f} MHz")
        print(f"total IF shift:    {(delta1_Hz + delta2_Hz) / u.MHz:+.6f} MHz")
        print(f"final IF:          {md['IF_freq'] / u.MHz:.6f} MHz")
        print(
            f"target |res| <= {FINAL_RESIDUAL_TARGET_MHZ:.3f} MHz: "
            f"{'PASS' if row['pass_final_target'] else 'CHECK'}"
        )
        print("-" * 70)

        rows.append(row)
        h5_path, csv_path, fig_path = _write_outputs(parent, rows)
        print(f"Incremental summary: {h5_path}")
        print(f"Incremental CSV:     {csv_path}")
        print(f"Incremental figure:  {fig_path}")

    h5_path, csv_path, fig_path = _write_outputs(parent, rows)
    print("\n" + "=" * 78)
    print("APE iterative apply sweep complete")
    print(f"Output folder: {parent}")
    print(f"H5 summary:    {h5_path}")
    print(f"CSV summary:   {csv_path}")
    print(f"Figure:        {fig_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
