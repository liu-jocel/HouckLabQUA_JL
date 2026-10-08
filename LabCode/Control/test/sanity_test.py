"""
Standalone APE-vs-FrequencyXY IF-swap sanity runner with final AAE-after-APE diagnostic.

Purpose
-------
Diagnose whether the bad high-beta RB result after APE calibration is caused by
(1) the scalar IF chosen by APE, or (2) some other difference in the calibration
path / meta_dict / config.

For one pulse length and one DRAG beta, this script creates two calibrated states
from the same post-chevron seed:

    APE-cal state:      Chevron -> [AAE -> APE] x N -> final AAE
    FreqXY-cal state:   Chevron -> [AAE -> Frequency_Calibration_XY] staged

The final AAE is intentionally placed after the last APE correction to test the
hypothesis that the old APE branch was ending with a stale amplitude calibration.
The script also keeps an optional pre-final-AAE APE candidate so you can compare
RB before and after the final amplitude repin at the same IF.

It then runs RB on four candidate states:

    A0. APE-cal state before the final AAE  (optional diagnostic)
    A.  APE-cal state after the final AAE
    B.  FreqXY-cal state
    C.  FreqXY-cal state with ONLY IF_freq replaced by final-AAE APE-cal IF
    D.  final-AAE APE-cal state with ONLY IF_freq replaced by FreqXY-cal IF

Interpretation
--------------
If RB follows the IF value:
    A and C are bad, B and D are good  -> APE is choosing a bad scalar IF for RB.

If RB follows the calibration path:
    A and D are bad, B and C are good  -> amplitude/waveform/path differences matter.

If APE sanity is near zero for A but RB is bad:
    X90-pair APE is not sufficient to optimize the full Clifford gate set.

Notes
-----
* Requires the keep-phase-corrected APE class. The import is explicit by default.
* No Frequency_Calibration_XY is applied to the APE-cal state.
* No APE correction is applied to the FreqXY-cal state, except in the IF-swap candidate.
* The RB config is rebuilt from the candidate meta_dict immediately before every RB.
"""

import os
import time
import copy
import inspect
import h5py
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

plt.ion()

# Add LabCode to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Rabi.m_Rabi_Chevron_IQ import RabiChevronIQ
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.RB.m_RB_1Q import RB1Q
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage

# Prefer the explicit keep-phase file. If your repo overwrote m_amplified_phase_error.py
# with the keep-phase implementation, this import can be changed back, but do not do so silently.
try:
    from LabCode.Experiments.Gates.m_amplified_phase_error_keep_phase import APE
except Exception:
    from LabCode.Experiments.Gates.m_amplified_phase_error import APE

sig = inspect.signature(APE.__init__)
print("Using APE from:", APE.__module__)
print("APE signature:", sig)
if "keep_phase_during_detuning" not in sig.parameters:
    raise RuntimeError(
        "Imported APE does not expose keep_phase_during_detuning. "
        "Use the keep-phase corrected APE file before running this sanity test."
    )

# =============================================================================
# User settings
# =============================================================================
QUBIT = "q6"
PULSE_TYPE = "X90"
PULSE_LENGTH_NS = 16
DRAG_BETA = 1.2  # choose one bad point first, e.g. 1.0 or 1.2

# Run controls
RUN_RB1Q = True
RUN_LEAKAGE_RB = False
RUN_APE_SANITY_FOR_CANDIDATES = True
RUN_RB_3STATE = True
RUN_3L_CAL_PER_CANDIDATE = (
    False  # False = one 3L cal on APE state reused for all candidates
)
NUM_RB_AVERAGES = 3

# Calibration depth
AAE_APE_CYCLES = 2
FREQ_XY_MAX_ITER = 3

# New diagnostic for the amplitude-miscalibration hypothesis:
# after the final APE correction, run one more AAE and use that state for APE-cal RB.
# This changes amplitude only; it does not apply another APE correction afterward.
FINAL_AAE_AFTER_APE_BRANCH = True
KEEP_PREFINAL_AAE_CANDIDATE = (
    True  # includes A0_APEcal_preFinalAAE in the IF-swap table
)
RUN_APE_SANITY_AFTER_FINAL_AAE = True  # check only, no APE apply after the final AAE

# Targets / thresholds
FINAL_APE_RESIDUAL_TARGET_MHZ = 0.03
SS_MIN_FIDELITY = 0.70
SS3_MIN_FIDELITY = 0.65

# Chevron bound
CHEVRON_AMP_MAX_HEADROOM = 1.3
HARDWARE_AMP_MAX = 0.49

# Experiment definitions. Keep these close to the pulse-length sweep runner.
experiment_dict = {
    "SS_Cal": dict(shots=1000, number_pi_pulses=1, ground_threshold=0.7),
    "SS_Cal_3L": dict(shots=600),
    "Chevron_IQ": dict(
        shots=21,
        amp_min=0.0,
        amp_max=0.35,
        amp_points=31,
        freq_span=21 * u.MHz,
        freq_points=11,
    ),
    "Chevron": dict(
        shots=20,
        amp_min=0.0,
        amp_max=0.35,
        amp_points=31,
        freq_span=21 * u.MHz,
        freq_points=31,
    ),
    "Amplified_Amplitude_Error": dict(
        shots=200,
        N_min=1,
        N_step=1,
        N_max_sequence=[11, 51, 81],
    ),
    "Amplified_Phase_Error": dict(
        shots=100,
        center=(0.0, 0.0),
        center_mode="fixed",
        min_radius=0.15,
        local_fit_points=5,
        allow_extrapolation=False,
        keep_phase_during_detuning=True,
        broad_span_MHz=8.0,
        broad_points=17,
        broad_N_vec=np.arange(1, 11, 1),
        narrow_span_MHz=1.5,
        narrow_points=11,
        narrow_N_vec=np.arange(1, 11, 1),
        fine_span_MHz=0.50,
        fine_points=21,
        fine_N_vec=np.arange(1, 11, 1),
        sanity_span_MHz=1.0,
        sanity_points=15,
        sanity_N_vec=np.arange(1, 11, 1),
    ),
    "Frequency_Calibration_XY": dict(
        shots=100,
        pulse_sequence=[("X90", "I"), ("Y90", "I"), ("X90", "Y90"), ("Y90", "X90")],
        freq_span_stages=[5 * u.MHz, 1 * u.MHz],
        freq_points=401,
        num_repeats=1,
    ),
    "RB1Q": dict(
        simulate=False,
        shots=50,
        num_circuits=50,
        circuit_lengths=[
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1200,
            1500,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
        ],
    ),
    "Leakage_RB": dict(
        shots=25,
        num_circuits=20,
        num_averages=1,
        circuit_lengths=[
            2,
            4,
            8,
            16,
            64,
            128,
            250,
            300,
            500,
            600,
            800,
            1000,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
            6000,
            7000,
        ],
    ),
}


# =============================================================================
# Helpers
# =============================================================================
def _stage_value(seq, idx):
    return seq[min(idx, len(seq) - 1)]


def _f_vec_MHz(center_MHz, span_MHz, points):
    lo = (center_MHz - 0.5 * span_MHz) * u.MHz
    hi = (center_MHz + 0.5 * span_MHz) * u.MHz
    return np.linspace(lo, hi, points, dtype=int)


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def _safe_nanstd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if np.any(np.isfinite(arr)) else np.nan


def _zero_gate_detunings(md):
    # Keep APE/XY corrections on the element IF path, not the waveform-internal detuning path.
    md["detuning"] = 0
    for gate in ("X90", "Y90", "X180", "Y180"):
        if gate in md and isinstance(md[gate], dict) and "detuning" in md[gate]:
            md[gate]["detuning"] = 0


def _ape_opt_MHz(ape):
    return float(ape.data.get("optimal_detuning_MHz", np.nan))


def _ape_status(ape):
    return dict(
        opt_MHz=_ape_opt_MHz(ape),
        zero_bracketed=bool(ape.data.get("zero_bracketed", False)),
        fit_status=ape.data.get("fit_status", "unknown"),
        slope_over_expected=float(ape.data.get("slope_over_expected", np.nan)),
    )


def _print_config_check(label, md, cfg, extra=""):
    cfg_if = cfg["elements"][QUBIT]["intermediate_frequency"]
    print("\n" + "=" * 78)
    print(label)
    if extra:
        print(extra)
    print(f"meta IF_freq     = {md['IF_freq'] / u.MHz:.6f} MHz")
    print(f"meta detuning    = {md.get('detuning', 0) / u.MHz:.6f} MHz")
    print(f"config IF        = {cfg_if / u.MHz:.6f} MHz")
    print(f"X90 amp          = {md['X90']['amp']:.9f}")
    print(f"Y90 amp          = {md['Y90']['amp']:.9f}")
    print(f"X90 phase        = {md['X90'].get('phase', None)}")
    print(f"Y90 phase        = {md['Y90'].get('phase', None)}")
    print(f"DRAG beta        = {md['DRAG_beta']}")
    print("=" * 78 + "\n")


def _run_ss_2level(md, cfg, parent_folder, path):
    ss_cfg = experiment_dict["SS_Cal"]
    max_retries = 10
    for k in range(max_retries):
        ss = SingleShot1Q(
            QUBIT,
            ss_cfg["shots"],
            meta_dict=md,
            outer_folder=parent_folder,
            path=path,
            suffix=f"SS_Cal_try{k:02d}",
            repeats=ss_cfg["number_pi_pulses"],
            config=cfg,
            confidence_threshold=ss_cfg["ground_threshold"],
            flat_path=True,
        )
        print(f"SS Cal try {k}: fidelity {ss.max_F:.3f}")
        if ss.max_F > SS_MIN_FIDELITY:
            md["read_theta"] = ss.calib_params["read_theta"]
            cfg = meta_dict_to_config_1Q(md)
            return ss.calib_params, cfg
    raise RuntimeError("2-level SS calibration failed")


def _run_ss_3level(md, cfg, calib_parent, path, suffix="SS_Cal_3L"):
    ss3_cfg = experiment_dict["SS_Cal_3L"]
    max_retries = 30
    for k in range(max_retries):
        ss3 = SingleShot1Q3Level(
            QUBIT,
            ss3_cfg["shots"],
            meta_dict=md,
            outer_folder=calib_parent,
            path=path,
            suffix=f"{suffix}_try{k:02d}",
            config=cfg,
            flat_path=True,
        )
        print(f"3L SS Cal {suffix} try {k}: fidelity {ss3.max_F:.3f}")
        if ss3.max_F > SS3_MIN_FIDELITY:
            return ss3.calib_params
    print(f"WARNING: 3-level SS calibration failed for {suffix}")
    return None


def _run_ape_once(md, cfg, calib_params, f_vec, N_vec, parent_folder, path, suffix):
    ape_cfg = experiment_dict["Amplified_Phase_Error"]
    print(
        f"APE {suffix}: IF={md['IF_freq'] / u.MHz:.6f} MHz, "
        f"f=[{np.min(f_vec) / u.MHz:+.3f},{np.max(f_vec) / u.MHz:+.3f}] MHz, "
        f"N=[{int(np.min(N_vec))},{int(np.max(N_vec))}], pts={len(f_vec)}x{len(N_vec)}"
    )
    ape = APE(
        QUBIT,
        f_vec,
        N_vec,
        ape_cfg["shots"],
        calib_params,
        pulse_type=PULSE_TYPE,
        meta_dict=md,
        outer_folder=parent_folder,
        path=path,
        suffix=suffix,
        config=cfg,
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
        f"APE result {suffix}: opt={st['opt_MHz']:+.6f} MHz, "
        f"bracketed={st['zero_bracketed']}, status={st['fit_status']}, "
        f"slope/expected={st['slope_over_expected']:.3f}"
    )
    return ape


def _run_ape_staged_and_apply(md, cfg, calib_params, parent_folder, path, prefix):
    ape_cfg = experiment_dict["Amplified_Phase_Error"]

    f_broad = _f_vec_MHz(0.0, ape_cfg["broad_span_MHz"], ape_cfg["broad_points"])
    ape_broad = _run_ape_once(
        md,
        cfg,
        calib_params,
        f_broad,
        ape_cfg["broad_N_vec"],
        parent_folder,
        path,
        f"{prefix}_APE_broad",
    )
    opt_broad = _ape_opt_MHz(ape_broad)
    if not np.isfinite(opt_broad):
        return cfg, dict(
            broad_MHz=opt_broad,
            narrow_MHz=np.nan,
            fine_MHz=np.nan,
            applied_MHz=0.0,
            status="broad_failed",
        )

    f_narrow = _f_vec_MHz(
        opt_broad, ape_cfg["narrow_span_MHz"], ape_cfg["narrow_points"]
    )
    ape_narrow = _run_ape_once(
        md,
        cfg,
        calib_params,
        f_narrow,
        ape_cfg["narrow_N_vec"],
        parent_folder,
        path,
        f"{prefix}_APE_narrow",
    )
    opt_narrow = _ape_opt_MHz(ape_narrow)
    if not np.isfinite(opt_narrow):
        return cfg, dict(
            broad_MHz=opt_broad,
            narrow_MHz=opt_narrow,
            fine_MHz=np.nan,
            applied_MHz=0.0,
            status="narrow_failed",
        )

    f_fine = _f_vec_MHz(opt_narrow, ape_cfg["fine_span_MHz"], ape_cfg["fine_points"])
    ape_fine = _run_ape_once(
        md,
        cfg,
        calib_params,
        f_fine,
        ape_cfg["fine_N_vec"],
        parent_folder,
        path,
        f"{prefix}_APE_fine",
    )
    st_fine = _ape_status(ape_fine)
    opt_fine = st_fine["opt_MHz"]
    if not np.isfinite(opt_fine):
        return cfg, dict(
            broad_MHz=opt_broad,
            narrow_MHz=opt_narrow,
            fine_MHz=opt_fine,
            applied_MHz=0.0,
            status="fine_failed",
        )

    old_if = int(md["IF_freq"])
    delta_if = int(opt_fine * u.MHz)
    md["IF_freq"] = old_if + delta_if
    cfg = meta_dict_to_config_1Q(md)
    print(
        f"Applied APE correction: old_IF={old_if / u.MHz:.6f} MHz, "
        f"df={delta_if / u.MHz:+.6f} MHz, new_IF={md['IF_freq'] / u.MHz:.6f} MHz"
    )
    return cfg, dict(
        broad_MHz=opt_broad,
        narrow_MHz=opt_narrow,
        fine_MHz=opt_fine,
        applied_MHz=delta_if / u.MHz,
        status=st_fine["fit_status"],
        bracketed=st_fine["zero_bracketed"],
        slope_over_expected=st_fine["slope_over_expected"],
    )


def _run_final_ape_sanity(md, cfg, calib_params, parent_folder, path, prefix):
    ape_cfg = experiment_dict["Amplified_Phase_Error"]
    f_vec = _f_vec_MHz(0.0, ape_cfg["sanity_span_MHz"], ape_cfg["sanity_points"])
    ape = _run_ape_once(
        md,
        cfg,
        calib_params,
        f_vec,
        ape_cfg["sanity_N_vec"],
        parent_folder,
        path,
        f"{prefix}_APE_sanity",
    )
    return _ape_status(ape)


def _run_aae_once(md, cfg, calib_params, parent_folder, path, suffix, cycle_idx):
    aae_cfg = experiment_dict["Amplified_Amplitude_Error"]
    N_max = _stage_value(aae_cfg["N_max_sequence"], cycle_idx)
    N_vec = np.arange(aae_cfg["N_min"], N_max, aae_cfg["N_step"])
    aae = AAE(
        QUBIT,
        N_vec,
        aae_cfg["shots"],
        calib_params=calib_params,
        pulse_type=PULSE_TYPE,
        meta_dict=md,
        outer_folder=parent_folder,
        path=path,
        suffix=suffix,
        config=cfg,
        flat_path=True,
    )
    amp_err = float(aae.data["amplitude_error"])
    md["X90"]["amp"] += amp_err
    md["Y90"]["amp"] += amp_err
    cfg = meta_dict_to_config_1Q(md)
    print(f"AAE {suffix}: amp_err={amp_err:+.6e}, new_amp={md['X90']['amp']:.9f}")
    return cfg, amp_err


def _calibrate_ape_branch(seed_md, seed_cfg, calib_params, parent_folder, path):
    """APE branch calibration.

    Default sequence with AAE_APE_CYCLES=2 and FINAL_AAE_AFTER_APE_BRANCH=True:
        AAE -> APE(apply) -> AAE -> APE(apply) -> AAE(final, amplitude only)

    The pre-final-AAE state is retained in the returned summary so RB can compare
    whether the last AAE fixes the high-beta APE-cal RB blow-up.
    """
    md = copy.deepcopy(seed_md)
    cfg = meta_dict_to_config_1Q(md)
    history = []

    for c in range(AAE_APE_CYCLES):
        print(f"\n--- APE branch cycle {c + 1}/{AAE_APE_CYCLES} ---")
        cfg, amp_err = _run_aae_once(
            md,
            cfg,
            calib_params,
            parent_folder,
            path,
            f"APEbranch_AAE_cycle{c:02d}",
            c,
        )
        cfg, ape_sum = _run_ape_staged_and_apply(
            md,
            cfg,
            calib_params,
            parent_folder,
            path,
            f"APEbranch_cycle{c:02d}",
        )
        history.append(
            dict(
                cycle=c,
                kind="AAE_APE",
                amp_err=amp_err,
                amp=md["X90"]["amp"],
                IF_freq_Hz=int(md["IF_freq"]),
                **ape_sum,
            )
        )

    # Snapshot exactly where the old runner would have stopped: after final APE apply.
    pre_final_md = copy.deepcopy(md)
    pre_final_cfg = meta_dict_to_config_1Q(pre_final_md)
    pre_final_sanity = _run_final_ape_sanity(
        pre_final_md,
        pre_final_cfg,
        calib_params,
        parent_folder,
        path,
        "APEbranch_preFinalAAE",
    )

    final_aae_amp_err = np.nan
    if FINAL_AAE_AFTER_APE_BRANCH:
        print("\n--- APE branch final AAE after final APE correction ---")
        final_cycle_idx = AAE_APE_CYCLES
        cfg, final_aae_amp_err = _run_aae_once(
            md,
            cfg,
            calib_params,
            parent_folder,
            path,
            "APEbranch_FINAL_AAE_after_APE",
            final_cycle_idx,
        )
        history.append(
            dict(
                cycle=final_cycle_idx,
                kind="FINAL_AAE_AFTER_APE",
                amp_err=final_aae_amp_err,
                amp=md["X90"]["amp"],
                IF_freq_Hz=int(md["IF_freq"]),
                broad_MHz=np.nan,
                narrow_MHz=np.nan,
                fine_MHz=np.nan,
                applied_MHz=0.0,
                status="final_aae_only",
                bracketed=np.nan,
                slope_over_expected=np.nan,
            )
        )

    if RUN_APE_SANITY_AFTER_FINAL_AAE:
        sanity = _run_final_ape_sanity(
            md, cfg, calib_params, parent_folder, path, "APEbranch_finalPostAAE"
        )
    else:
        sanity = dict(
            opt_MHz=np.nan,
            zero_bracketed=False,
            fit_status="not_run",
            slope_over_expected=np.nan,
        )

    return (
        md,
        cfg,
        dict(
            history=history,
            pre_final_md=pre_final_md,
            pre_final_sanity=pre_final_sanity,
            final_aae_amp_err=final_aae_amp_err,
            final_sanity=sanity,
        ),
    )


def _calibrate_xy_branch(seed_md, seed_cfg, calib_params, parent_folder, path):
    md = copy.deepcopy(seed_md)
    cfg = meta_dict_to_config_1Q(md)
    aae_cfg = experiment_dict["Amplified_Amplitude_Error"]
    xy_cfg = experiment_dict["Frequency_Calibration_XY"]
    history = []

    for stage_idx, freq_span in enumerate(xy_cfg["freq_span_stages"]):
        for i in range(FREQ_XY_MAX_ITER):
            print(f"\n--- XY branch stage {stage_idx}, iter {i} ---")
            cfg, amp_err = _run_aae_once(
                md,
                cfg,
                calib_params,
                parent_folder,
                path,
                f"XYbranch_AAE_s{stage_idx}_i{i:02d}",
                stage_idx,
            )
            freq_vec = np.linspace(
                -freq_span / 2, freq_span / 2, xy_cfg["freq_points"], dtype=int
            )
            fxy = Frequency_Calibration_XY(
                QUBIT,
                freq_vec,
                xy_cfg["pulse_sequence"],
                xy_cfg["shots"],
                calib_params,
                num_repeats=xy_cfg["num_repeats"],
                meta_dict=md,
                outer_folder=parent_folder,
                path=path,
                suffix=f"XYbranch_FreqXY_s{stage_idx}_i{i:02d}",
                config=cfg,
                flat_path=True,
            )
            det_corr_MHz = float(fxy.data["optimal_detuning_MHz"])
            md["IF_freq"] += int(det_corr_MHz * u.MHz)
            cfg = meta_dict_to_config_1Q(md)
            history.append(
                dict(
                    stage=stage_idx,
                    iter=i,
                    amp_err=amp_err,
                    det_corr_MHz=det_corr_MHz,
                    amp=md["X90"]["amp"],
                    IF_freq_Hz=int(md["IF_freq"]),
                )
            )
            print(
                f"XY branch: amp_err={amp_err:+.6e}, det_corr={det_corr_MHz:+.6f} MHz, IF={md['IF_freq'] / u.MHz:.6f} MHz"
            )
            if abs(amp_err) < 5e-4 and abs(det_corr_MHz) < 0.05:
                break

    sanity = _run_final_ape_sanity(
        md, cfg, calib_params, parent_folder, path, "XYbranch_final"
    )
    return md, cfg, dict(history=history, final_sanity=sanity)


def _run_rb_candidate(name, md, cfg, calib_params, calib_3L, parent_folder):
    _print_config_check(f"PRE-RB CONFIG CHECK: {name}", md, cfg)
    rb_3state_calib = calib_3L if (RUN_RB_3STATE and calib_3L is not None) else None
    fids = []
    if not RUN_RB1Q:
        return dict(rb_error=np.nan, rb_std=np.nan, fids=[])

    for avg_idx in range(NUM_RB_AVERAGES):
        rb = RB1Q(
            QUBIT,
            experiment_dict["RB1Q"]["circuit_lengths"],
            experiment_dict["RB1Q"]["num_circuits"],
            experiment_dict["RB1Q"]["shots"],
            calib_params,
            meta_dict=md,
            outer_folder=parent_folder,
            path=name,
            suffix=f"RB_avg{avg_idx:02d}",
            config=cfg,
            simulate=experiment_dict["RB1Q"]["simulate"],
            flat_path=True,
            calib_params_3state=rb_3state_calib,
        )
        fid = float(rb.data.get("gate_fidelity", np.nan))
        fids.append(fid)
        print(
            f"{name} RB avg {avg_idx + 1}/{NUM_RB_AVERAGES}: fidelity={fid:.8f}, err={1 - fid:.3e}"
        )

    fid_mean = _safe_nanmean(fids)
    fid_std = _safe_nanstd(fids)
    return dict(
        rb_error=(1.0 - fid_mean if np.isfinite(fid_mean) else np.nan),
        rb_std=fid_std,
        fids=fids,
    )


def _run_lrb_candidate(name, md, cfg, calib_3L, parent_folder):
    if not RUN_LEAKAGE_RB or calib_3L is None:
        return dict(L1=np.nan, L1_err=np.nan, L2=np.nan, L2_err=np.nan, lambda1=np.nan)
    lrb_cfg = experiment_dict["Leakage_RB"]
    rb = RB1Q_Leakage(
        QUBIT,
        lrb_cfg["circuit_lengths"],
        lrb_cfg["num_circuits"],
        lrb_cfg["shots"],
        calib_3L,
        meta_dict=md,
        outer_folder=parent_folder,
        path=name,
        suffix="LRB",
        config=cfg,
        flat_path=True,
    )
    L1, L1_err = rb.data["fit_data"]["state_2_L1"]
    L2, L2_err = rb.data["fit_data"]["state_2_L2"]
    return dict(
        L1=float(L1),
        L1_err=float(L1_err),
        L2=float(L2),
        L2_err=float(L2_err),
        lambda1=float(1.0 - L1 - L2),
    )


def _make_candidate(name, base_md, forced_if_Hz=None):
    md = copy.deepcopy(base_md)
    if forced_if_Hz is not None:
        md["IF_freq"] = int(forced_if_Hz)
    cfg = meta_dict_to_config_1Q(md)
    return name, md, cfg


def _plot_results(parent_folder, results):
    names = [r["name"] for r in results]
    rb = np.array([r.get("rb_error", np.nan) for r in results], dtype=float)
    rb_std = np.array([r.get("rb_std", np.nan) for r in results], dtype=float)
    ape = np.array([r.get("ape_resid_MHz", np.nan) for r in results], dtype=float)
    L1 = np.array([r.get("L1", np.nan) for r in results], dtype=float)

    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].errorbar(x, rb, yerr=rb_std, fmt="o", capsize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RB error")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(x, ape, "o")
    axes[1].axhline(0, color="k", lw=1, alpha=0.5)
    axes[1].axhline(+FINAL_APE_RESIDUAL_TARGET_MHZ, color="k", ls="--", lw=1, alpha=0.4)
    axes[1].axhline(-FINAL_APE_RESIDUAL_TARGET_MHZ, color="k", ls="--", lw=1, alpha=0.4)
    axes[1].set_ylabel("APE resid [MHz]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, L1, "s")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("L1")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=25, ha="right")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].set_title(f"{QUBIT} {PULSE_LENGTH_NS} ns beta={DRAG_BETA}: IF-swap sanity")

    fig.tight_layout()
    out = os.path.join(
        parent_folder,
        f"{QUBIT}_{PULSE_LENGTH_NS}ns_beta{DRAG_BETA:+.3f}_if_swap_finalAAE_sanity.png".replace(
            ".", "p"
        ),
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", out)
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    base_md = make_meta_dict(QUBIT)[QUBIT]
    outer_folder_base = base_md["global_params"]["outer_folder"]
    date_str = datetime.now().strftime("%Y_%m_%d")
    q_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{date_str}")
    os.makedirs(q_date_dir, exist_ok=True)
    parent_folder = os.path.join(
        q_date_dir,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_APE_vs_XY_IF_swap_sanity",
    )
    os.makedirs(parent_folder, exist_ok=True)
    print("Parent folder:", parent_folder)

    # Initial pulse-length / beta seed and IQ chevron bootstrap.
    md0 = make_meta_dict(QUBIT)[QUBIT]
    md0["DRAG_beta"] = float(DRAG_BETA)
    md0["X90"]["length"] = PULSE_LENGTH_NS
    md0["Y90"]["length"] = PULSE_LENGTH_NS
    md0["global_params"]["outer_folder"] = parent_folder
    _zero_gate_detunings(md0)
    cfg0 = meta_dict_to_config_1Q(md0)

    iq_cfg = experiment_dict["Chevron_IQ"]
    amp_vec_iq = np.linspace(iq_cfg["amp_min"], iq_cfg["amp_max"], iq_cfg["amp_points"])
    f_vec_iq = np.linspace(
        -iq_cfg["freq_span"] / 2,
        iq_cfg["freq_span"] / 2,
        iq_cfg["freq_points"],
        dtype=int,
    )
    chev_iq = RabiChevronIQ(
        QUBIT,
        amp_vec_iq,
        f_vec_iq,
        iq_cfg["shots"],
        pulse_type=PULSE_TYPE,
        meta_dict=md0,
        outer_folder=parent_folder,
        path="bootstrap_IQ",
        suffix=f"ChevronIQ_{PULSE_LENGTH_NS}ns_beta{DRAG_BETA:+.3f}".replace(".", "p"),
        config=cfg0,
        flat_path=True,
    )
    pl_IF_freq_offset = int(chev_iq.data["opt_detuning_Hz"])
    pl_amp_init = float(chev_iq.data["opt_amp"])
    md0["IF_freq"] += pl_IF_freq_offset
    md0["X90"]["amp"] = pl_amp_init
    md0["Y90"]["amp"] = pl_amp_init
    cfg0 = meta_dict_to_config_1Q(md0)
    print(
        f"IQ bootstrap: IF offset={pl_IF_freq_offset / u.MHz:+.6f} MHz, amp={pl_amp_init:.9f}"
    )

    # 2-level SS calibration shared by both branches.
    calib_params, cfg0 = _run_ss_2level(md0, cfg0, parent_folder, path="shared_SS")

    # RabiChevronSS bootstrap from same seed, as in your main autocal runner.
    chev_cfg = experiment_dict["Chevron"]
    amp_max = min(HARDWARE_AMP_MAX, pl_amp_init * CHEVRON_AMP_MAX_HEADROOM)
    amp_vec = np.linspace(chev_cfg["amp_min"], amp_max, chev_cfg["amp_points"])
    f_vec = np.linspace(
        -chev_cfg["freq_span"] / 2,
        chev_cfg["freq_span"] / 2,
        chev_cfg["freq_points"],
        dtype=int,
    )
    chevron = RabiChevronSS(
        QUBIT,
        amp_vec,
        f_vec,
        chev_cfg["shots"],
        calib_params,
        num_pi_pulses=1,
        pulse_type=PULSE_TYPE,
        meta_dict=md0,
        outer_folder=parent_folder,
        path="post_iq_chevron",
        suffix="ChevronSS_seed",
        config=cfg0,
        flat_path=True,
    )
    opt_detuning_Hz = int(chevron.data["opt_detuning_Hz"])
    opt_amp = float(chevron.data["opt_amp"])
    md0["IF_freq"] += opt_detuning_Hz
    md0["X90"]["amp"] = opt_amp
    md0["Y90"]["amp"] = opt_amp
    cfg0 = meta_dict_to_config_1Q(md0)
    seed_md = copy.deepcopy(md0)
    seed_cfg = meta_dict_to_config_1Q(seed_md)
    print(
        f"ChevronSS seed: IF offset={opt_detuning_Hz / u.MHz:+.6f} MHz, amp={opt_amp:.9f}"
    )
    _print_config_check("POST-CHEVRON SEED", seed_md, seed_cfg)

    # Calibrate branches from identical seed.
    start = time.time()
    ape_md, ape_cfg, ape_summary = _calibrate_ape_branch(
        seed_md, seed_cfg, calib_params, parent_folder, path="APE_branch"
    )
    xy_md, xy_cfg, xy_summary = _calibrate_xy_branch(
        seed_md, seed_cfg, calib_params, parent_folder, path="XY_branch"
    )
    print(f"Branch calibrations done in {time.time() - start:.0f}s")

    ape_pre_md = ape_summary.get("pre_final_md")
    ape_pre_cfg = meta_dict_to_config_1Q(ape_pre_md) if ape_pre_md is not None else None

    if ape_pre_md is not None:
        _print_config_check(
            "APE-CAL STATE BEFORE FINAL AAE",
            ape_pre_md,
            ape_pre_cfg,
            extra=f"pre-final APE sanity={ape_summary.get('pre_final_sanity')}",
        )
    _print_config_check(
        "FINAL APE-CAL STATE AFTER FINAL AAE",
        ape_md,
        ape_cfg,
        extra=(
            f"post-final-AAE APE sanity={ape_summary['final_sanity']}; "
            f"final_AAE_amp_err={ape_summary.get('final_aae_amp_err', np.nan):+.6e}"
        ),
    )
    _print_config_check(
        "FINAL FREQ-XY-CAL STATE",
        xy_md,
        xy_cfg,
        extra=f"XY sanity={xy_summary['final_sanity']}",
    )

    ape_if = int(ape_md["IF_freq"])
    ape_pre_if = int(ape_pre_md["IF_freq"]) if ape_pre_md is not None else ape_if
    xy_if = int(xy_md["IF_freq"])
    print("\n" + "-" * 78)
    print(f"APE IF before final AAE = {ape_pre_if / u.MHz:.6f} MHz")
    print(f"APE IF after final AAE  = {ape_if / u.MHz:.6f} MHz")
    print(f"XY IF                   = {xy_if / u.MHz:.6f} MHz")
    print(f"IF diff after-finalAAE  = {(ape_if - xy_if) / u.MHz:+.6f} MHz  (APE - XY)")
    print(
        f"Final AAE amp_err       = {ape_summary.get('final_aae_amp_err', np.nan):+.6e}"
    )
    print("-" * 78 + "\n")

    # Shared 3L readout cal by default. Use APE state because it is the branch being questioned.
    calib_3L_shared = None
    if RUN_RB_3STATE or RUN_LEAKAGE_RB:
        calib_3L_shared = _run_ss_3level(
            ape_md,
            ape_cfg,
            parent_folder,
            path="shared_3L",
            suffix="APE_state_shared_3L",
        )

    # Build the decisive candidates. A0 is the old stopping point before the final AAE.
    candidates = []
    if KEEP_PREFINAL_AAE_CANDIDATE and ape_pre_md is not None:
        candidates.append(
            _make_candidate("A0_APEcal_preFinalAAE", ape_pre_md, forced_if_Hz=None)
        )
    candidates.extend(
        [
            _make_candidate("A_APEcal_postFinalAAE", ape_md, forced_if_Hz=None),
            _make_candidate("B_XYcal", xy_md, forced_if_Hz=None),
            _make_candidate("C_XYcal_IFtoAPE", xy_md, forced_if_Hz=ape_if),
            _make_candidate("D_APEcal_IFtoXY", ape_md, forced_if_Hz=xy_if),
        ]
    )

    results = []
    for name, md, cfg in candidates:
        print("\n" + "#" * 78)
        print(f"RUNNING CANDIDATE: {name}")
        print("#" * 78)

        sanity = dict(
            opt_MHz=np.nan,
            zero_bracketed=False,
            fit_status="not_run",
            slope_over_expected=np.nan,
        )
        if RUN_APE_SANITY_FOR_CANDIDATES:
            sanity = _run_final_ape_sanity(
                md, cfg, calib_params, parent_folder, path=name, prefix=name
            )

        if RUN_3L_CAL_PER_CANDIDATE and (RUN_RB_3STATE or RUN_LEAKAGE_RB):
            calib_3L = _run_ss_3level(
                md, cfg, parent_folder, path=name, suffix=f"{name}_3L"
            )
        else:
            calib_3L = calib_3L_shared

        rb_res = _run_rb_candidate(name, md, cfg, calib_params, calib_3L, parent_folder)
        lrb_res = _run_lrb_candidate(name, md, cfg, calib_3L, parent_folder)

        row = dict(
            name=name,
            IF_freq_Hz=int(md["IF_freq"]),
            IF_freq_MHz=md["IF_freq"] / u.MHz,
            amp=float(md["X90"]["amp"]),
            drag_beta=float(md["DRAG_beta"]),
            ape_resid_MHz=sanity["opt_MHz"],
            ape_bracketed=sanity["zero_bracketed"],
            ape_fit_status=sanity["fit_status"],
            ape_slope_over_expected=sanity["slope_over_expected"],
            **rb_res,
            **lrb_res,
        )
        results.append(row)

        print("\nRESULT", name)
        for k, v in row.items():
            if k != "fids":
                print(f"  {k}: {v}")

    # Save H5 and text summary.
    out_h5 = os.path.join(
        parent_folder,
        f"{QUBIT}_{PULSE_LENGTH_NS}ns_beta{DRAG_BETA:+.3f}_if_swap_finalAAE_sanity.h5".replace(
            ".", "p"
        ),
    )
    with h5py.File(out_h5, "w") as f:
        f.attrs["qubit"] = QUBIT
        f.attrs["pulse_length_ns"] = PULSE_LENGTH_NS
        f.attrs["drag_beta"] = DRAG_BETA
        f.attrs["ape_if_Hz"] = ape_if
        f.attrs["ape_pre_final_aae_if_Hz"] = ape_pre_if
        f.attrs["xy_if_Hz"] = xy_if
        f.attrs["ape_pre_final_sanity_opt_MHz"] = ape_summary.get(
            "pre_final_sanity", {}
        ).get("opt_MHz", np.nan)
        f.attrs["ape_final_aae_amp_err"] = ape_summary.get("final_aae_amp_err", np.nan)
        f.attrs["ape_final_sanity_opt_MHz"] = ape_summary["final_sanity"]["opt_MHz"]
        f.attrs["xy_final_sanity_opt_MHz"] = xy_summary["final_sanity"]["opt_MHz"]
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "candidate_names",
            data=np.array([r["name"] for r in results], dtype=object),
            dtype=dt,
        )
        for key in [
            "IF_freq_Hz",
            "IF_freq_MHz",
            "amp",
            "drag_beta",
            "ape_resid_MHz",
            "ape_slope_over_expected",
            "rb_error",
            "rb_std",
            "L1",
            "L1_err",
            "L2",
            "L2_err",
            "lambda1",
        ]:
            f.create_dataset(
                key, data=np.array([r.get(key, np.nan) for r in results], dtype=float)
            )
    print("Saved H5:", out_h5)

    plot_path = _plot_results(parent_folder, results)

    print("\n" + "=" * 78)
    print("IF-SWAP SANITY SUMMARY")
    for r in results:
        print(
            f"{r['name']:18s}  IF={r['IF_freq_MHz']:+.6f} MHz  "
            f"APE={r['ape_resid_MHz']:+.6f} MHz  "
            f"RBerr={r['rb_error']:.6e}  L1={r.get('L1', np.nan):.3e}"
        )
    print("\nInterpretation guide:")
    print(
        "  If A and C are bad, B and D good: RB follows IF; APE picked a bad scalar IF for RB."
    )
    print(
        "  If A and D are bad, B and C good: RB follows calibration path; non-IF differences matter."
    )
    print(
        "  If A has near-zero APE but bad RB: X90-pair APE is not sufficient for full Clifford RB."
    )
    print("Saved plot:", plot_path)
    print("Saved H5:", out_h5)
    print("=" * 78)


if __name__ == "__main__":
    main()
