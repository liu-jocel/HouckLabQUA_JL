"""
APE-vs-DRAG-beta diagnostic for q6.

Purpose
-------
Test whether increasing DRAG beta creates a scalar coherent phase / detuning
error. The key point is that IF_freq is frozen after the beta=0 calibration.
For beta > 0, this script optionally runs AAE only; it does NOT run a frequency
calibration before measuring APE.

Install/use
-----------
1. Replace LabCode/Experiments/Gates/m_amplified_phase_error.py with the robust
   version I provided, or import that robust APE class here.
2. Edit the knobs in the USER SETTINGS block.
3. Run this script from your LabCode environment.

Interpretation
--------------
- If APE optimal detuning moves with beta and RB recovers after applying it,
  the beta-induced error is mostly scalar Z/Stark/frame error.
- If APE moves but RB does not recover, beta is causing additional axis/shape/IQ
  errors.
- If APE does not move but RB worsens, the beta error is not simple scalar phase.
"""

import os
import time
import copy
import inspect
import h5py
from pathlib import Path
from datetime import datetime

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

# Prefer the robust APE implementation. Keep a clear failure mode if Python
# accidentally imports the old class that lacks center/local-fit kwargs.
try:
    from LabCode.Experiments.Gates.m_amplified_phase_error_robust import APE
except ImportError:
    from LabCode.Experiments.Gates.m_amplified_phase_error import APE

# Optional recovery RB. Leave False for the first diagnostic pass.
from LabCode.Experiments.RB.m_RB_1Q import RB1Q


# =============================================================================
# USER SETTINGS
# =============================================================================
QUBIT = "q6"
PULSE_TYPE = "X90"
PULSE_LENGTH_NS = 20

DRAG_BETA_VEC = np.array([0.0, 0.3, 0.6, 0.8, 1.0])

# Bootstrap/calibrate beta=0, then freeze IF_freq for all beta values.
DO_BETA0_IQ_CHEVRON = True
DO_BETA0_SS_CHEVRON = True
DO_BETA0_AAE_FREQ_LOOP = True
BETA0_CAL_ITERS = 2

# For beta > 0, correct only pulse area. This intentionally does NOT correct phase.
DO_AAE_PER_BETA = True
AAE_N_VEC = np.arange(1, 61, 1)
AAE_SHOTS = 300

# APE broad then narrow. The broad span should bracket the zero. If it does not,
# the robust APE returns NaN rather than extrapolating.
APE_SHOTS = 250
APE_CENTER_MODE = "fixed"  # q6: use fixed center, not partial-arc mean
APE_CENTER = (0.0, 0.0)
APE_MIN_RADIUS = 0.15
APE_LOCAL_FIT_POINTS = 5
APE_ALLOW_EXTRAPOLATION = False

APE_BROAD_SPAN_MHZ = 8.0  # [-4,+4] MHz around frozen IF
APE_BROAD_POINTS = 17
APE_BROAD_N_VEC = np.arange(1, 41, 1)

APE_NARROW_SPAN_MHZ = 1.5  # centered on broad estimate
APE_NARROW_POINTS = 11
APE_NARROW_N_VEC = np.arange(1, 81, 1)

# After measuring APE(beta), apply that detuning and rerun a narrow APE around zero.
# This is the best sign/self-consistency check.
DO_APPLY_APE_AND_RERUN_SANITY = True
APE_SANITY_SPAN_MHZ = (
    4.0  # wider sanity window: catches wrong-sign/not-applied corrections
)
APE_SANITY_POINTS = 17
APE_SANITY_N_VEC = np.arange(1, 81, 1)

# Optional quick RB after applying the APE correction. Off by default because APE is
# the first diagnostic and RB is slow/noisy.
DO_RB_RECOVERY_CHECK = False
RB_SHOTS = 50
RB_NUM_CIRCUITS = 30
RB_LENGTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# SS/readout and chevron settings.
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

# Sign/path debugging. This offset is applied AFTER the beta=0 calibration and
# AFTER the freeze point, so it cannot be undone by the beta=0 calibration loop.
# For a sign test, set MANUAL_FROZEN_IF_OFFSET_MHZ = +1.0 and verify that the
# beta=0 APE optimum shifts by about -1 MHz. Leave at 0.0 for normal use.
MANUAL_FROZEN_IF_OFFSET_MHZ = 0.0

# APE correction convention used by this runner:
#   measured optimum df_opt means zero phase at IF_freq + df_opt, so apply
#       IF_freq <- IF_freq + df_opt
# Do not change this unless the +1 MHz sign test proves the opposite.
APE_APPLY_SIGN = +1.0


# =============================================================================
# HELPERS
# =============================================================================
def _beta_label(beta):
    return f"beta{beta:+.3f}".replace(".", "p")


def _f_vec(center_Hz, span_MHz, points):
    half = 0.5 * span_MHz * u.MHz
    return np.linspace(center_Hz - half, center_Hz + half, points, dtype=int)


def _fresh_meta(qubit, pulse_length_ns, beta, outer_folder):
    md = make_meta_dict(qubit)[qubit]
    md["DRAG_beta"] = float(beta)
    md["X90"]["length"] = int(pulse_length_ns)
    md["Y90"]["length"] = int(pulse_length_ns)
    md["global_params"]["outer_folder"] = outer_folder

    # Avoid mixing element-frame correction with waveform detuning unless you
    # explicitly know your waveform generator needs it. APE will report an IF shift.
    for g in ("X90", "Y90", "mX90", "mY90"):
        if g in md and isinstance(md[g], dict) and "detuning" in md[g]:
            md[g]["detuning"] = 0
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
    """Calibrate beta=0 and return frozen meta/config/calib_params."""
    print("\n=== beta=0 bootstrap/calibration ===")
    md = _fresh_meta(qubit, pulse_length_ns, beta=0.0, outer_folder=outer_folder)
    cfg = meta_dict_to_config_1Q(md)

    # Optional IQ chevron to get close without readout thresholding.
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
            f"    IQ opt: df={ch_iq.data['opt_detuning_Hz'] / u.MHz:+.3f} MHz, amp={ch_iq.data['opt_amp']:.6f}"
        )

    calib, cfg = _run_ss(qubit, md, cfg, outer_folder, path="beta0_bootstrap")

    # Optional SS chevron to refine amplitude and detuning.
    if DO_BETA0_SS_CHEVRON:
        print("  SS chevron refinement...")
        amp_vec = np.linspace(
            CHEVRON_AMP_MIN,
            min(CHEVRON_AMP_MAX, md["X90"]["amp"] * 1.35),
            CHEVRON_AMP_POINTS,
        )
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
            f"    SS opt: df={ch.data['opt_detuning_Hz'] / u.MHz:+.3f} MHz, amp={ch.data['opt_amp']:.6f}"
        )

    # Normal beta=0 amplitude/frequency convergence. This is allowed only before
    # the IF freeze point.
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

            f_vec = _f_vec(
                0,
                FREQ_CAL_SPANS[min(i, len(FREQ_CAL_SPANS) - 1)] / u.MHz,
                FREQ_CAL_POINTS,
            )
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
    amp_err = aae.data["amplitude_error"]
    meta_dict["X90"]["amp"] += amp_err
    meta_dict["Y90"]["amp"] += amp_err
    config = meta_dict_to_config_1Q(meta_dict)
    return amp_err, config


def _run_ape(qubit, meta_dict, config, calib, outer_folder, path, suffix, f_vec, N_vec):
    """Run robust APE with verbose IF/f_vec diagnostics.

    The printout is intentionally noisy because the main failure mode in this
    debugging pass is that the IF correction is not actually applied to the APE
    block, or is applied with the wrong sign.
    """
    sig = inspect.signature(APE.__init__)
    required = [
        "center",
        "center_mode",
        "min_radius",
        "local_fit_points",
        "allow_extrapolation",
    ]
    missing = [k for k in required if k not in sig.parameters]
    if missing:
        raise RuntimeError(
            "Imported APE class is not the robust version. Missing kwargs: "
            f"{missing}. Import path is APE.__module__={APE.__module__}.\n"
            "Put m_amplified_phase_error_robust.py in LabCode/Experiments/Gates "
            "or replace m_amplified_phase_error.py with the robust version."
        )

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
    )

    fit_slope = ape.data.get("fit_slope_rad_per_Hz", np.nan)
    exp_slope = ape.data.get("expected_slope_rad_per_Hz", np.nan)
    ratio = (
        fit_slope / exp_slope if np.isfinite(exp_slope) and exp_slope != 0 else np.nan
    )
    print(
        f"APE result {suffix}: "
        f"opt={ape.data.get('optimal_detuning_MHz', np.nan):+.6f} MHz, "
        f"bracketed={ape.data.get('zero_bracketed', False)}, "
        f"status={ape.data.get('fit_status', 'unknown')}, "
        f"slope/expected={ratio:.3f}"
    )
    return ape


def _run_ape_staged(qubit, meta_dict, config, calib, outer_folder, path, beta_label):
    broad_f = _f_vec(0, APE_BROAD_SPAN_MHZ, APE_BROAD_POINTS)
    ape0 = _run_ape(
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
    opt0_Hz = ape0.data.get("optimal_detuning_Hz", np.nan)
    if not np.isfinite(opt0_Hz):
        return ape0, None

    narrow_f = _f_vec(opt0_Hz, APE_NARROW_SPAN_MHZ, APE_NARROW_POINTS)
    ape1 = _run_ape(
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
    return ape1, ape0


def _quick_rb(qubit, meta_dict, config, calib, outer_folder, path, suffix):
    rb = RB1Q(
        qubit,
        RB_LENGTHS,
        RB_NUM_CIRCUITS,
        RB_SHOTS,
        calib,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=path,
        suffix=suffix,
        config=config,
        simulate=False,
        flat_path=True,
    )
    fid = rb.data.get("gate_fidelity", np.nan)
    return 1.0 - fid


def _write_summary(summary_path, rows):
    dtype = [
        ("beta", "f8"),
        ("amp", "f8"),
        ("frozen_IF_Hz", "f8"),
        ("ape_opt_MHz", "f8"),
        ("ape_zero_bracketed", "i4"),
        ("ape_slope_rad_per_MHz", "f8"),
        ("ape_expected_slope_rad_per_MHz", "f8"),
        ("ape_slope_ratio", "f8"),
        ("ape_status_code", "S64"),
        ("ape_applied_correction_MHz", "f8"),
        ("ape_sanity_IF_Hz", "f8"),
        ("ape_residual_after_apply_MHz", "f8"),
        ("ape_residual_zero_bracketed", "i4"),
        ("rb_error_uncorrected", "f8"),
        ("rb_error_corrected", "f8"),
    ]
    arr = np.zeros(len(rows), dtype=dtype)
    for i, r in enumerate(rows):
        for name in arr.dtype.names:
            val = r.get(name, np.nan)
            arr[name][i] = (
                val.encode()
                if arr.dtype[name].kind == "S" and isinstance(val, str)
                else val
            )

    with h5py.File(summary_path, "w") as f:
        f.create_dataset("summary", data=arr)
        f.attrs["qubit"] = QUBIT
        f.attrs["pulse_type"] = PULSE_TYPE
        f.attrs["pulse_length_ns"] = PULSE_LENGTH_NS
        f.attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return arr


def _plot_summary(parent_folder, rows):
    if not rows:
        return None
    beta = np.array([r["beta"] for r in rows], dtype=float)
    ape = np.array([r["ape_opt_MHz"] for r in rows], dtype=float)
    resid = np.array([r["ape_residual_after_apply_MHz"] for r in rows], dtype=float)
    ratio = np.array([r["ape_slope_ratio"] for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(beta, ape, "o-")
    axes[0].axhline(0, color="k", lw=1, alpha=0.5)
    axes[0].set_ylabel("APE opt detuning [MHz]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(beta, resid, "o-")
    axes[1].axhline(0, color="k", lw=1, alpha=0.5)
    axes[1].set_ylabel("Residual after apply [MHz]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(beta, ratio, "o-")
    axes[2].axhline(1, color="k", lw=1, alpha=0.5)
    axes[2].set_ylabel("slope / expected")
    axes[2].set_xlabel("DRAG beta")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{QUBIT} {PULSE_LENGTH_NS} ns — APE vs DRAG beta; IF frozen")
    fig.tight_layout()
    path = os.path.join(parent_folder, f"{QUBIT}_ape_vs_drag_beta_summary.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


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
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_APE_vs_DRAG_beta_{PULSE_LENGTH_NS}ns",
    )
    os.makedirs(parent, exist_ok=True)
    print(f"Output folder: {parent}")

    beta0_meta, beta0_cfg, calib = _bootstrap_beta0(QUBIT, PULSE_LENGTH_NS, parent)
    frozen_IF_before_manual = int(beta0_meta["IF_freq"])
    frozen_IF = int(frozen_IF_before_manual + MANUAL_FROZEN_IF_OFFSET_MHZ * u.MHz)
    frozen_amp = float(beta0_meta["X90"]["amp"])

    print("\n" + "=" * 70)
    print("APE FROZEN IF CHECK")
    print(f"beta0 calibrated IF: {frozen_IF_before_manual / u.MHz:.6f} MHz")
    print(f"manual frozen offset: {MANUAL_FROZEN_IF_OFFSET_MHZ:+.6f} MHz")
    print(f"frozen IF used:      {frozen_IF / u.MHz:.6f} MHz")
    print(f"frozen X90 amp:      {frozen_amp:.6f}")
    print(f"APE_APPLY_SIGN:      {APE_APPLY_SIGN:+.1f}")
    print("=" * 70 + "\n")

    rows = []
    summary_path = os.path.join(parent, f"{QUBIT}_APE_vs_DRAG_beta_summary.h5")

    for beta in DRAG_BETA_VEC:
        label = _beta_label(beta)
        print(f"\n{'=' * 70}\nDRAG beta={beta:+.3f}  (IF frozen)\n{'=' * 70}")
        t0 = time.time()

        md = _fresh_meta(QUBIT, PULSE_LENGTH_NS, beta=beta, outer_folder=parent)
        md["IF_freq"] = frozen_IF
        md["X90"]["amp"] = frozen_amp
        md["Y90"]["amp"] = frozen_amp
        md["read_theta"] = beta0_meta.get("read_theta", md.get("read_theta", 0))
        cfg = meta_dict_to_config_1Q(md)

        amp_err = np.nan
        if DO_AAE_PER_BETA:
            print("  Running AAE only; no frequency correction...")
            amp_err, cfg = _run_aae_only(
                QUBIT, md, cfg, calib, parent, label, suffix=f"AAE_only_{label}"
            )
            print(f"    amp_err={amp_err:+.3e}; amp={md['X90']['amp']:.6f}")

        rb_uncorr = np.nan
        if DO_RB_RECOVERY_CHECK:
            print("  Optional RB before APE correction...")
            rb_uncorr = _quick_rb(
                QUBIT, md, cfg, calib, parent, label, suffix=f"RB_uncorrected_{label}"
            )
            print(f"    RB error uncorrected={rb_uncorr:.3e}")

        print("  Running staged APE with frozen IF...")
        ape, ape_broad = _run_ape_staged(QUBIT, md, cfg, calib, parent, label, label)
        opt_Hz = ape.data.get("optimal_detuning_Hz", np.nan)
        opt_MHz = ape.data.get("optimal_detuning_MHz", np.nan)
        bracketed = bool(ape.data.get("zero_bracketed", False))
        slope_rad_per_MHz = ape.data.get("fit_slope_rad_per_Hz", np.nan) * 1e6
        expected_rad_per_MHz = ape.data.get("expected_slope_rad_per_Hz", np.nan) * 1e6
        slope_ratio = (
            slope_rad_per_MHz / expected_rad_per_MHz
            if np.isfinite(expected_rad_per_MHz)
            else np.nan
        )
        status = ape.data.get("fit_status", "unknown")
        print(f"    APE opt={opt_MHz:+.4f} MHz, bracketed={bracketed}, status={status}")
        print(
            f"    slope={slope_rad_per_MHz:.3f} rad/MHz, expected={expected_rad_per_MHz:.3f}, ratio={slope_ratio:.2f}"
        )

        residual_MHz = np.nan
        residual_bracketed = False
        applied_corr_MHz = np.nan
        sanity_IF_Hz = np.nan
        rb_corr = np.nan
        if DO_APPLY_APE_AND_RERUN_SANITY and np.isfinite(opt_Hz):
            print("  Applying APE detuning and rerunning sanity APE around zero...")
            md_corr = copy.deepcopy(md)

            old_IF = int(md_corr["IF_freq"])
            delta_IF = int(APE_APPLY_SIGN * opt_Hz)
            md_corr["IF_freq"] = old_IF + delta_IF
            sanity_IF_Hz = float(md_corr["IF_freq"])
            applied_corr_MHz = delta_IF / u.MHz

            print("\n" + "=" * 70)
            print("APE APPLY CHECK")
            print(f"APE optimum:       {opt_MHz:+.6f} MHz")
            print(f"old IF:            {old_IF / u.MHz:.6f} MHz")
            print(f"delta IF applied:  {applied_corr_MHz:+.6f} MHz")
            print(f"new IF:            {md_corr['IF_freq'] / u.MHz:.6f} MHz")
            print(
                f"sanity f window:   +/-{APE_SANITY_SPAN_MHZ / 2:.3f} MHz around new IF"
            )
            print("=" * 70 + "\n")

            cfg_corr = meta_dict_to_config_1Q(md_corr)
            sanity_f = _f_vec(0, APE_SANITY_SPAN_MHZ, APE_SANITY_POINTS)
            ape_sanity = _run_ape(
                QUBIT,
                md_corr,
                cfg_corr,
                calib,
                parent,
                label,
                suffix=f"APE_sanity_after_apply_{label}",
                f_vec=sanity_f,
                N_vec=APE_SANITY_N_VEC,
            )
            residual_MHz = ape_sanity.data.get("optimal_detuning_MHz", np.nan)
            residual_bracketed = bool(ape_sanity.data.get("zero_bracketed", False))
            print(
                f"    residual APE optimum after apply = {residual_MHz:+.4f} MHz, "
                f"bracketed={residual_bracketed}"
            )

            if DO_RB_RECOVERY_CHECK:
                print("  Optional RB after APE correction...")
                rb_corr = _quick_rb(
                    QUBIT,
                    md_corr,
                    cfg_corr,
                    calib,
                    parent,
                    label,
                    suffix=f"RB_corrected_{label}",
                )
                print(f"    RB error corrected={rb_corr:.3e}")

        row = dict(
            beta=float(beta),
            amp=float(md["X90"]["amp"]),
            frozen_IF_Hz=float(frozen_IF),
            ape_opt_MHz=float(opt_MHz) if np.isfinite(opt_MHz) else np.nan,
            ape_zero_bracketed=int(bracketed),
            ape_slope_rad_per_MHz=float(slope_rad_per_MHz)
            if np.isfinite(slope_rad_per_MHz)
            else np.nan,
            ape_expected_slope_rad_per_MHz=float(expected_rad_per_MHz)
            if np.isfinite(expected_rad_per_MHz)
            else np.nan,
            ape_slope_ratio=float(slope_ratio) if np.isfinite(slope_ratio) else np.nan,
            ape_status_code=str(status),
            ape_applied_correction_MHz=float(applied_corr_MHz)
            if np.isfinite(applied_corr_MHz)
            else np.nan,
            ape_sanity_IF_Hz=float(sanity_IF_Hz)
            if np.isfinite(sanity_IF_Hz)
            else np.nan,
            ape_residual_after_apply_MHz=float(residual_MHz)
            if np.isfinite(residual_MHz)
            else np.nan,
            ape_residual_zero_bracketed=int(residual_bracketed),
            rb_error_uncorrected=float(rb_uncorr) if np.isfinite(rb_uncorr) else np.nan,
            rb_error_corrected=float(rb_corr) if np.isfinite(rb_corr) else np.nan,
        )
        rows.append(row)
        _write_summary(summary_path, rows)
        fig_path = _plot_summary(parent, rows)
        print(f"  beta={beta:+.3f} done in {time.time() - t0:.0f}s")
        print(f"  incremental summary: {summary_path}")
        if fig_path:
            print(f"  incremental figure: {fig_path}")

    print("\nDone.")
    print(f"Summary H5: {summary_path}")
    print(f"Figure: {_plot_summary(parent, rows)}")


if __name__ == "__main__":
    main()
