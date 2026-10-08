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
from LabCode.Experiments.Gates.m_pulse_distortion_tomography import PulseDistortionTomography
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage
from LabCode.Experiments.RB.m_RB_1Q import RB1Q
from LabCode.Experiments.RB.m_RB_purity_1Q import RB1Q_Purity
from LabCode.Experiments.RB.m_RB_1Q_Interleaved import RB1Q_Interleaved
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2E import T2E
from qualang_tools.units import unit


u = unit(coerce_to_integer=True)

# Gates per Clifford for the 1Q Clifford decomposition (53/24).
GATES_PER_CLIFFORD = 2.2083

PULSE_TYPE = "Y90" # Needs to be a 90 in order to run properly
QUBIT = "q2"
DO_LEAKAGE_CHARACTERIZATION = True  # If True: run ss_cal_3l and leakage RB; if False: only sscal, tune, and regular RB
DO_PURITY_RB = True # If True: after best beta is chosen, run purity/unitarity RB once and add a coherent/incoherent error-budget figure
DO_IRB = True  # If True: after best beta is chosen, run interleaved RB for each gate (X90, mX90, Y90, mY90, I) and add a per-gate-fidelity-vs-gate-time figure
DO_RB_3STATE = False# If True: vanilla RB uses 3-state readout (SingleShot1Q3Level) and computes return
# probability renormalized over {0,1} (|2> shots discarded), instead of naive 2-state thresholding
# which misassigns |2> leakage into |1>. Reuses the 3-level cal; active reset is kept.
NUM_AVERAGES = 3  # Number of times to repeat vanilla RB at best beta for each pulse length to measure variance
DO_COHERENCE_MEASUREMENTS = False  #BUGGY If True: measure T1 and T2E once before first RB
# Chevron amplitude bound. Rotation ~ amp*length, so the pi/2 amplitude shrinks as the
# pulse lengthens. Sweeping fast->slow, we bound each chevron's amp_max by the previous
# (faster) pulse length's beta=0 Rabi amplitude times this headroom, keeping the pi lobe
# well sampled. The fastest pulse (no predecessor) uses the full Chevron amp_max. Set to
# 1.0 to use the previous amplitude exactly as the max.
CHEVRON_AMP_MAX_HEADROOM = 1.3
HARDWARE_AMP_MAX = 0.45  # OPX per-port ceiling used for the sweep
# ─── Sweep vectors ────────────────────────────────────────────────────────────
# Edit these to set the pulse lengths (ns) and DRAG betas to sweep.
PULSE_LENGTH_VEC = [16,20,24,28,32,36,40]#  # ns
DRAG_beta_vec = np.array([0.0])#,0.2,0.4,0.6,0.8,0.9,1.0,1.1,1.2,1.4])
#DRAG_beta_vec = np.linspace(0, 1.8,16)
tomo_pulse_vec = np.linspace(1,500,50)
optimizer_convergence_params = {
    "amplitude_accuracy": 0.0005,
    "detuning_accuracy": 0.05,  # MHz
    "max_iter": 4,
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
        "amp_max": 0.35,  # 0.45,
        "amp_points": 31,
        "freq_span": 21 * u.MHz,
        "freq_points": 31,
        "object": RabiChevronSS,
    },
    "Amplified_Amplitude_Error": {
        "shots": 300,
        "N_min": 1,
        "N_step": 1,
        "N_max_stages": [11, 51, 81],
        "object": AAE,
    },
    "Frequency_Calibration_XY": {
        "shots": 100,
        "pulse_sequence": [
            ("X90", "I"),
            ("Y90", "I"),
            ("X90", "Y90"),
            ("Y90", "X90"),
        ],
        "freq_span_stages": [5 * u.MHz, 1 * u.MHz],
        "freq_points": 201,
        "num_repeats": 1,
        "object": Frequency_Calibration_XY,
    },
    "Pulse_Distortion_Tomography": {
        "shots": 200,
        "object": PulseDistortionTomography,
    },
    "Leakage_RB": {
        "shots": 50,
        "num_circuits": 50,
        "num_averages": NUM_AVERAGES,
        "circuit_lengths": [
            2,
            4,
            8,
            16,
            32,
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
        "object": RB1Q_Leakage,
    },
    "RB1Q": {
        "run": True,
        "simulate": False,
        "shots": 50,
        "num_circuits": 50,
        "circuit_lengths": [
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
        "object": RB1Q,
    },
    # Purity / unitarity RB. Shorter, denser depth vector than survival RB
    # (purity decays ~2x faster in the exponent, 1 - u ~ 2(1 - p)), and more
    # shots/circuits because the purity is a squared (noisier) quantity.
    # Tune circuit_lengths so the purity decays by roughly an e-fold across the range.
    "Purity_RB": {
        "shots": 100,
        "num_circuits": 100,
        "circuit_lengths": [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1500,
            2000,
            3000,
            4000,
        ],
        "object": RB1Q_Purity,
    },
    # Interleaved RB. circuit_depths are the reference depths; IRB runs at half
    # those depths so the total gate count (Clifford + interleaved) matches the
    # reference.  Gates: X90, mX90, Y90, mY90, I (identity wait).
    "Interleaved_RB": {
        "shots": 50,
        "num_circuits": 50,
        "circuit_depths": [
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1000,
            2000,
            4000,
            7000,
        ],
        "gates": ["X90", "mX90", "Y90", "mY90", "I"],
        "object": RB1Q_Interleaved,
    },
    "T1": {
        "run": True,
        "shots": 500,
        "num_pulses": [1],
        "t_min": 400 * u.ns,
        "t_max": 5000 * u.us,
        "t_points": 51,
        "object": T1,
    },
    "T2E": {
        "run": True,
        "shots": 500,
        "t_min": 0.5 * u.us,
        "t_max": 5000 * u.us,
        "t_points": 51,
        "object": T2E,
    },
}


def save_drag_summary(
    pl_folder,
    qubit,
    pulse_length_ns,
    do_leakage,
    drag_results,
    rb_error_rates,
    rb_error_rate_errs,
):
    """(Re)write the per-pulse-length DRAG-sweep H5 + figure from the results
    accumulated so far. Called after every beta so both outputs update in real time
    and can be monitored mid-sweep; the same files are overwritten each call."""
    if len(drag_results) == 0:
        return None

    betas = np.array([r[0] for r in drag_results])
    l1s = np.array([r[1] for r in drag_results])
    l1errs = np.array([r[2] for r in drag_results])
    l2s = np.array([r[3] for r in drag_results])
    l2errs = np.array([r[4] for r in drag_results])
    lambda1s = np.array([r[5] for r in drag_results])
    rb_err_arr = np.array(rb_error_rates)
    rb_err_std_arr = np.array(rb_error_rate_errs)

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
        f.attrs["qubit"] = qubit
        f.attrs["pulse_length_ns"] = int(pulse_length_ns)

    if do_leakage:
        one_minus_lambda1s = 1.0 - lambda1s
        fig_drag, axes_drag = plt.subplots(4, 1, figsize=(7, 13), sharex=True)
        for ax_d, ys, yerrs, ylabel, ylim_bottom in zip(
            axes_drag[:3],
            [l1s, l2s, one_minus_lambda1s],
            [l1errs, l2errs, np.zeros(len(one_minus_lambda1s))],
            [
                "L1  (leakage per Clifford)",
                "L2  (leakage per Clifford)",
                "1 - λ1",
            ],
            [1e-6, None, 1e-5],
        ):
            ax_d.errorbar(betas, ys, yerr=yerrs, fmt="o-", capsize=4)
            ax_d.set_yscale("log")
            ax_d.set_ylim(bottom=ylim_bottom)
            ax_d.set_ylabel(ylabel)
            ax_d.grid(True, which="both", alpha=0.3)

        # 4th subplot: total RB error and leakage error (L1) on same plot
        axes_drag[3].errorbar(
            betas,
            rb_err_arr,
            yerr=rb_err_std_arr,
            fmt="o-",
            capsize=4,
            label="Total error (RB)",
        )
        axes_drag[3].errorbar(
            betas, l1s, yerr=l1errs, fmt="s-", capsize=4, label="Leakage error (L1)"
        )
        axes_drag[3].set_yscale("log")
        axes_drag[3].set_ylim(bottom=1e-5)
        axes_drag[3].set_ylabel("Error rate")
        axes_drag[3].legend()
        axes_drag[3].grid(True, which="both", alpha=0.3)

        axes_drag[-1].set_xlabel("DRAG beta")
        axes_drag[0].set_title(
            f"{qubit} — leakage vs DRAG beta  ({pulse_length_ns} ns pulse)"
        )
        fig_drag.tight_layout()
        drag_fig_path = os.path.join(
            pl_folder, f"{qubit}_L1_L2_lambda1_vs_DRAG_beta.png"
        )
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
        ax_drag.set_title(
            f"{qubit} — RB fidelity vs DRAG beta  ({pulse_length_ns} ns pulse)"
        )
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
    Rabi chevron bootstrap + iterative AAE + Frequency_Calibration_XY convergence.
    Modifies meta_dict in-place and returns the final config.

    chevron_amp_max overrides the Chevron sweep's upper amplitude (e.g. anchored to
    this pulse length's bootstrap amplitude) so the pi lobe stays well sampled.
    """
    if save_path is None:
        save_path = qubit

    # Phase 1: Rabi chevron bootstrap
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
        f"    Chevron first π-pulse at detuning={opt_detuning_Hz / u.MHz:.3f} MHz, amp={opt_amp:.4f} V"
    )

    meta_dict["IF_freq"] += int(opt_detuning_Hz)
    meta_dict["X90"]["amp"] = opt_amp
    meta_dict["Y90"]["amp"] = opt_amp
    # meta_dict["ef_amp"] = opt_amp
    config = meta_dict_to_config_1Q(meta_dict)

    # Phase 2: Staged AAE + Frequency calibration
    max_iter = conv_params["max_iter"]
    N_max_stages = exp_dict["Amplified_Amplitude_Error"]["N_max_stages"]
    freq_span_stages = exp_dict["Frequency_Calibration_XY"]["freq_span_stages"]

    for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
        print(
            f"    ── Stage {stage_idx} | AAE N_max={N_max}  freq_span={freq_span / u.MHz:.1f} MHz ──"
        )
        calibrated = dict(amplitude=False, detuning=False)

        for i in range(max_iter):
            N_vec = np.arange(
                exp_dict["Amplified_Amplitude_Error"]["N_min"],
                N_max,
                exp_dict["Amplified_Amplitude_Error"]["N_step"],
            )
            aae = AAE(
                qubit,
                N_vec,
                exp_dict["Amplified_Amplitude_Error"]["shots"],
                calib_params=calib_params,
                pulse_type=pulse_type,
                meta_dict=meta_dict,
                outer_folder=meta_dict["global_params"]["outer_folder"],
                path=save_path,
                suffix=f"AAE_{beta_label}_s{stage_idx}_iter{i:02d}",
                config=config,
                flat_path=True,
            )
            amp_err = aae.data["amplitude_error"]
            meta_dict["X90"]["amp"] += amp_err
            meta_dict["Y90"]["amp"] += amp_err
            # meta_dict["ef_amp"] = meta_dict["X90"]["amp"]
            config = meta_dict_to_config_1Q(meta_dict)

            freq_vec = np.linspace(
                -freq_span / 2,
                freq_span / 2,
                exp_dict["Frequency_Calibration_XY"]["freq_points"],
                dtype=int,
            )
            freq_cal = Frequency_Calibration_XY(
                qubit,
                freq_vec,
                exp_dict["Frequency_Calibration_XY"]["pulse_sequence"],
                exp_dict["Frequency_Calibration_XY"]["shots"],
                calib_params,
                num_repeats=exp_dict["Frequency_Calibration_XY"]["num_repeats"],
                meta_dict=meta_dict,
                outer_folder=meta_dict["global_params"]["outer_folder"],
                path=save_path,
                suffix=f"FreqCal_{beta_label}_s{stage_idx}_iter{i:02d}",
                config=config,
                flat_path=True,
            )
            det_correction_MHz = freq_cal.data["optimal_detuning_MHz"]
            meta_dict["IF_freq"] += int(det_correction_MHz * u.MHz)

            calibrated["amplitude"] = abs(amp_err) < conv_params["amplitude_accuracy"]
            calibrated["detuning"] = (
                abs(det_correction_MHz) < conv_params["detuning_accuracy"]
            )

            config = meta_dict_to_config_1Q(meta_dict)

            print(
                f"      Iter {i:02d} | amp_err={amp_err:+.3e}  "
                f"det_correction={det_correction_MHz:+.4f} MHz  | "
                f"calibrated={calibrated}"
            )

            if all(calibrated.values()):
                print(f"    Stage {stage_idx} converged after {i + 1} iterations")
                break

    return config


# ─── Create parent folder ─────────────────────────────────────────────────────

_base_meta = make_meta_dict(QUBIT)[QUBIT]
outer_folder_base = _base_meta["global_params"]["outer_folder"]
_date_str = datetime.now().strftime("%Y_%m_%d")
_q1_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{_date_str}")
os.makedirs(_q1_date_dir, exist_ok=True)

parent_folder = os.path.join(
    _q1_date_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pulse_length_sweep",
)
os.makedirs(parent_folder, exist_ok=True)
print(f"Pulse length sweep folder: {parent_folder}")

# ─── Main sweep ───────────────────────────────────────────────────────────────

# (pulse_len_ns, best_beta, L1, L1_err, L2, L2_err, lambda1, amp, IF_freq_Hz, rb_err, rb_err_std,
#  r_coh, r_coh_err, r_incoh, r_incoh_err, unitarity, unitarity_err)
sweep_results = []
coherence_measured = False  # Track if T1/T2E measurements have been done

sweep_start = time.time()
print(f"Sweep started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Gates measured by interleaved RB (when DO_IRB).
_IRB_GATES = experiment_dict["Interleaved_RB"]["gates"]

# ─── Create global H5 file for incremental updates ────────────────────────────
global_h5_path = os.path.join(parent_folder, f"{QUBIT}_pulse_length_sweep_summary.h5")
with h5py.File(global_h5_path, "w") as f:
    f.create_dataset("pulse_lengths_ns", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("best_beta", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("rb_error_rate", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("rb_error_rate_std", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("L1", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("L1_err", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("L2", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("L2_err", (0,), maxshape=(None,), dtype=float)
    # Purity-RB-derived error budget (NaN-filled when DO_PURITY_RB is False)
    f.create_dataset("r_coh", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("r_coh_err", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("r_incoh", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("r_incoh_err", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("unitarity", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("unitarity_err", (0,), maxshape=(None,), dtype=float)
    # IRB gate fidelities (NaN-filled when DO_IRB is False)
    for _g in _IRB_GATES:
        f.create_dataset(f"irb_F_{_g}", (0,), maxshape=(None,), dtype=float)
        f.create_dataset(f"irb_F_{_g}_err", (0,), maxshape=(None,), dtype=float)
    f.attrs["qubit"] = QUBIT
    f.attrs["num_averages"] = NUM_AVERAGES
    f.attrs["do_purity_rb"] = DO_PURITY_RB
    f.attrs["do_irb"] = DO_IRB

# Create figure for the live / final 4x1 summary vs gate time
# (L1, L2, 1-lambda1, and total+leakage error overlaid — same panel structure as the
#  per-pulse-length DRAG-beta figure, but with gate time on the x-axis).
fig, axes = plt.subplots(4, 1, figsize=(8, 13), sharex=True)
fig.suptitle(f"{QUBIT} — Leakage and RB Error vs Gate Time")

# Track plot data in Python lists to avoid HDF5 file locking during live updates
pls_list = []
rb_errs_list = []
rb_stds_list = []
l1s_list = []
l1_errs_list = []
l2s_list = []
l2_errs_list = []
lambda1s_list = []

# beta=0 Rabi amplitude from the previous (faster) pulse length; bounds the next
# chevron's amp_max. None for the first pulse length -> full Chevron amp_max is used.
prev_beta0_amp = None

# Pulse-distortion tomography heatmap accumulators. One column per pulse length that
# completes (last beta run for that length); rows are tomo_pulse_vec (N).
tomo_N_vec = np.round(tomo_pulse_vec).astype(int)
tomo_pop_Z_cols = []
tomo_pop_X_cols = []
tomo_pop_Y_cols = []
tomo_pulse_lengths_completed = []

for pulse_length_ns in PULSE_LENGTH_VEC:
    pl_label = f"{pulse_length_ns}ns"
    pl_folder = os.path.join(parent_folder, pl_label)
    os.makedirs(pl_folder, exist_ok=True)
    pl_start = time.time()

    print(f"\n{'=' * 60}")
    print(
        f"Pulse length = {pulse_length_ns} ns  |  started {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"{'=' * 60}")

    # ── IQ chevron bootstrap (once per pulse length, no SS required) ──────────
    _chev_iq = experiment_dict["Chevron_IQ"]
    # Bound amp_max by the previous (faster) pulse length's beta=0 Rabi amplitude;
    # the fastest pulse (prev_beta0_amp is None) keeps the full configured range.
    if prev_beta0_amp is None:
        _amp_max_iq = _chev_iq["amp_max"]
    else:
        _amp_max_iq = min(HARDWARE_AMP_MAX, prev_beta0_amp * CHEVRON_AMP_MAX_HEADROOM)
        print(
            f"  Chevron amp_max set to {_amp_max_iq:.4f} V "
            f"(prev pulse beta=0 amp {prev_beta0_amp:.4f} V x {CHEVRON_AMP_MAX_HEADROOM})"
        )
    _amp_vec_iq = np.linspace(_chev_iq["amp_min"], _amp_max_iq, _chev_iq["amp_points"])
    _f_vec_iq = np.linspace(
        -_chev_iq["freq_span"] / 2,
        _chev_iq["freq_span"] / 2,
        _chev_iq["freq_points"],
        dtype=int,
    )
    _md_iq = make_meta_dict(QUBIT)[QUBIT]
    _md_iq["X90"]["length"] = pulse_length_ns
    _md_iq["Y90"]["length"] = pulse_length_ns
    _md_iq["global_params"]["outer_folder"] = pl_folder
    # set_LO_parameters([_md_iq["q_LO"], _md_iq["r_LO"]])
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

    # (beta, L1, L1_err, L2, L2_err, lambda1, amp, IF_freq_Hz)
    drag_results = []
    lrb_l1_stds = []  # Track std of L1 across averages
    lrb_l2_stds = []  # Track std of L2 across averages
    rb_error_rates = []
    rb_error_rate_errs = []
    beta_configs = []  # per-beta calibrated state, aligned with rb_error_rates (for purity RB at best beta)
    beta0_amp_this_pl = (
        None  # beta=0 calibrated amp; carried to bound the next pulse length
    )

    # amp_max for the per-beta chevron, anchored to this pulse length's bootstrap amp
    pl_chevron_amp_max = min(HARDWARE_AMP_MAX, pl_amp_init * CHEVRON_AMP_MAX_HEADROOM)

    # This pulse length's tomography column (last completed beta); None if every
    # beta failed 2-level SS calibration.
    pl_tomo_pop_Z = None
    pl_tomo_pop_X = None
    pl_tomo_pop_Y = None

    for beta in DRAG_beta_vec:
        beta_label = f"beta{beta:+.3f}".replace(".", "p")
        beta_start = time.time()
        print(f"\n  DRAG beta = {beta:.3f}  |  {datetime.now().strftime('%H:%M:%S')}")

        meta_dict = make_meta_dict(QUBIT)[QUBIT]
        meta_dict["DRAG_beta"] = beta
        meta_dict["X90"]["length"] = pulse_length_ns
        meta_dict["Y90"]["length"] = pulse_length_ns
        meta_dict["X90"]["amp"] = pl_amp_init
        meta_dict["Y90"]["amp"] = pl_amp_init
        # meta_dict["ef_amp"] = pl_amp_init
        meta_dict["IF_freq"] += pl_IF_freq_offset
        meta_dict["global_params"]["outer_folder"] = pl_folder

        config = meta_dict_to_config_1Q(meta_dict)
        # set_LO_parameters([meta_dict["q_LO"], meta_dict["r_LO"]])

        # 2-level SS calibration — retry up to 10 times until fidelity > 0.70
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
            print(
                f"    Failed 2-level SS calibration at DRAG beta = {beta:.3f} after {max_retries_2L} attempts"
            )
            continue

        CALIB_PARAMS = SS.calib_params
        meta_dict["read_theta"] = SS.calib_params["read_theta"]
        config = meta_dict_to_config_1Q(meta_dict)

        # Chevron bootstrap + iterative calibration
        config = _run_calibration(
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

        print(
            f"    Final X90 amp={meta_dict['X90']['amp']:.6f} V  "
            f"IF_freq={meta_dict['IF_freq'] / u.MHz:.6f} MHz"
        )

        # ── Pulse-distortion tomography sweep ──────────────────────────────────
        pdt = PulseDistortionTomography(
            QUBIT,
            tomo_N_vec,
            experiment_dict["Pulse_Distortion_Tomography"]["shots"],
            CALIB_PARAMS,
            pulse_type=PULSE_TYPE, # X90
            meta_dict=meta_dict,
            outer_folder=pl_folder,
            path=beta_label,
            suffix=f"PulseDistTomo_{beta_label}",
            config=config,
            flat_path=True,
        )
        pl_tomo_pop_Z = pdt.data["pop_Z"]
        pl_tomo_pop_X = pdt.data["pop_X"]
        pl_tomo_pop_Y = pdt.data["pop_Y"]

    # ── Append this pulse length's tomography column to the cross-length heatmap ──
    if pl_tomo_pop_Z is not None:
        tomo_pop_Z_cols.append(pl_tomo_pop_Z)
        tomo_pop_X_cols.append(pl_tomo_pop_X)
        tomo_pop_Y_cols.append(pl_tomo_pop_Y)
        tomo_pulse_lengths_completed.append(pulse_length_ns)
    else:
        print(
            f"  No tomography column for pulse length {pulse_length_ns} ns "
            "(every beta failed 2-level SS calibration)"
        )

# ── Combined pulse-distortion tomography heatmap across all pulse lengths ─────
# 3 panels side-by-side (Z, X, Y); each is (len(tomo_pulse_vec) rows) x
# (len(pulse_lengths_completed) columns), colored by measured population.
tomo_pop_Z_matrix = np.array(tomo_pop_Z_cols).T
tomo_pop_X_matrix = np.array(tomo_pop_X_cols).T
tomo_pop_Y_matrix = np.array(tomo_pop_Y_cols).T

fig_tomo, axes_tomo = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
basis_matrices = {"Z": tomo_pop_Z_matrix, "X": tomo_pop_X_matrix, "Y": tomo_pop_Y_matrix}
im = None
for ax_t, (basis_label, matrix) in zip(axes_tomo, basis_matrices.items()):
    im = ax_t.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
        origin="lower",
        extent=[-0.5, len(tomo_pulse_lengths_completed) - 0.5, tomo_N_vec[0], tomo_N_vec[-1]],
    )
    ax_t.set_xticks(range(len(tomo_pulse_lengths_completed)))
    ax_t.set_xticklabels([f"{pl}ns" for pl in tomo_pulse_lengths_completed])
    ax_t.set_xlabel("Pulse length")
    ax_t.set_ylabel(f"N  ({PULSE_TYPE} pairs)")
    ax_t.set_title(f"{basis_label} basis")
fig_tomo.colorbar(im, ax=axes_tomo, label="Measured population")
fig_tomo.suptitle(f"{QUBIT} — pulse-distortion tomography vs pulse length")
tomo_heatmap_path = os.path.join(parent_folder, f"{QUBIT}_tomography_heatmap.png")
fig_tomo.savefig(tomo_heatmap_path, bbox_inches="tight")
plt.close(fig_tomo)
print(f"Pulse-distortion tomography heatmap saved to {tomo_heatmap_path}")
