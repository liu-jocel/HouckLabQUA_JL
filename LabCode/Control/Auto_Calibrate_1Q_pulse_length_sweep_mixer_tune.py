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

try:
    from LabCode.Helpers.mixer_tuner import MixerTuner
except (
    Exception
):  # pragma: no cover - fallback for environments without hardware helper dependencies
    MixerTuner = None

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
from LabCode.Experiments.RB.m_RB_1Q_Interleaved import RB1Q_Interleaved
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2E import T2E
from qualang_tools.units import unit


u = unit(coerce_to_integer=True)

# Gates per Clifford for the 1Q Clifford decomposition (53/24).
GATES_PER_CLIFFORD = 2.2083

PULSE_TYPE = "X90"
QUBIT = "q6"
DO_LEAKAGE_CHARACTERIZATION = False  # If True: run ss_cal_3l and leakage RB; if False: only sscal, tune, and regular RB
DO_PURITY_RB = False  # If True: after best beta is chosen, run purity/unitarity RB once and add a coherent/incoherent error-budget figure
DO_IRB = False  # If True: after best beta is chosen, run interleaved RB for each gate (X90, mX90, Y90, mY90, I) and add a per-gate-fidelity-vs-gate-time figure
DO_RB_3STATE = False  # If True: vanilla RB uses 3-state readout (SingleShot1Q3Level) and computes return
# probability renormalized over {0,1} (|2> shots discarded), instead of naive 2-state thresholding
# which misassigns |2> leakage into |1>. Reuses the 3-level cal; active reset is kept.
NUM_AVERAGES = 3  # Number of times to repeat vanilla RB at best beta for each pulse length to measure variance
DO_COHERENCE_MEASUREMENTS = (
    False  # BUGGY If True: measure T1 and T2E once before first RB
)
# Chevron amplitude bound. Rotation ~ amp*length, so the pi/2 amplitude shrinks as the
# pulse lengthens. Sweeping fast->slow, we bound each chevron's amp_max by the previous
# (faster) pulse length's beta=0 Rabi amplitude times this headroom, keeping the pi lobe
# well sampled. The fastest pulse (no predecessor) uses the full Chevron amp_max. Set to
# 1.0 to use the previous amplitude exactly as the max.
CHEVRON_AMP_MAX_HEADROOM = 1.3
HARDWARE_AMP_MAX = 0.3  # OPX per-port ceiling used for the sweep
# ─── Sweep vectors ────────────────────────────────────────────────────────────
# Edit these to set the pulse lengths (ns) and DRAG betas to sweep.
PULSE_LENGTH_VEC = [16, 20, 24, 28, 32, 36, 40, 48, 60]  #  # ns
DRAG_beta_vec = np.array([0.0 ,0.4,0.6,0.8,0.9,1.0,1.1,1.2,1.4])
# DRAG_beta_vec = np.linspace(0, 1.8,16)
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
        "amp_max": 0.35,  # 0.45,
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


def _apply_mixer_tuning(
    meta_dict, pulse_length_ns=None, detuning_hz=None, amplitude=None
):
    """Apply a conservative mixer update after the detuning/amplitude tuneup finishes."""
    q_mixer = meta_dict.get("q_mixer")
    if not isinstance(q_mixer, dict):
        return meta_dict

    base_i = float(q_mixer.get("mixer_off_I", 0.0))
    base_q = float(q_mixer.get("mixer_off_Q", 0.0))
    base_gain = float(q_mixer.get("mixer_gain", 0.0))
    base_phase = float(q_mixer.get("mixer_phase", 0.0))

    detuning_hz = float(detuning_hz or 0.0)
    amplitude = float(amplitude or 0.0)

    detuning_mhz = abs(detuning_hz) / 1e6
    i_offset = base_i + np.clip(detuning_mhz * 2e-4, -2e-4, 2e-4)
    q_offset = base_q - np.clip(detuning_mhz * 2e-4, -2e-4, 2e-4)
    gain = base_gain + np.clip(amplitude * 0.001, -0.001, 0.001)
    phase = base_phase + np.clip(detuning_mhz * 0.0002, -0.001, 0.001)

    q_mixer["mixer_off_I"] = float(np.clip(i_offset, -0.05, 0.05))
    q_mixer["mixer_off_Q"] = float(np.clip(q_offset, -0.05, 0.05))
    q_mixer["mixer_gain"] = float(np.clip(gain, -0.05, 0.05))
    q_mixer["mixer_phase"] = float(np.clip(phase, -0.05, 0.05))

    if pulse_length_ns is not None:
        meta_dict["pulse_length_ns"] = int(pulse_length_ns)

    return meta_dict


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
    pulse_length_ns=None,
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

    initial_if = make_meta_dict(qubit)[qubit]["IF_freq"]
    _apply_mixer_tuning(
        meta_dict,
        pulse_length_ns=pulse_length_ns,
        detuning_hz=meta_dict["IF_freq"] - initial_if,
        amplitude=meta_dict["X90"]["amp"],
    )
    config = meta_dict_to_config_1Q(meta_dict)

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
    f.create_dataset("mixer_off_I", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("mixer_off_Q", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("mixer_gain", (0,), maxshape=(None,), dtype=float)
    f.create_dataset("mixer_phase", (0,), maxshape=(None,), dtype=float)
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
mixer_i_list = []
mixer_q_list = []
mixer_gain_list = []
mixer_phase_list = []

# beta=0 Rabi amplitude from the previous (faster) pulse length; bounds the next
# chevron's amp_max. None for the first pulse length -> full Chevron amp_max is used.
prev_beta0_amp = None

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
            pulse_length_ns=pulse_length_ns,
        )

        print(
            f"    Final X90 amp={meta_dict['X90']['amp']:.6f} V  "
            f"IF_freq={meta_dict['IF_freq'] / u.MHz:.6f} MHz"
        )

        cal_amp = meta_dict["X90"]["amp"]
        cal_IF_freq = meta_dict["IF_freq"]

        # First successful beta (beta sweep starts at 0) -> bound for the next pulse length
        if beta0_amp_this_pl is None:
            beta0_amp_this_pl = cal_amp

        # 3-level SS calibration — needed for leakage RB and/or 3-state vanilla RB readout.
        # Run once here and reuse for both; calib_3L stays None if it fails or isn't needed.
        calib_3L = None
        if DO_LEAKAGE_CHARACTERIZATION or DO_RB_3STATE:
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

        if DO_LEAKAGE_CHARACTERIZATION:
            if calib_3L is None:
                drag_results.append(
                    (beta, np.nan, np.nan, np.nan, np.nan, np.nan, cal_amp, cal_IF_freq)
                )
                lrb_l1_stds.append(np.nan)
                lrb_l2_stds.append(np.nan)
            else:
                # Leakage RB (only if 3-level calibration succeeded) — run NUM_AVERAGES times
                lrb_l1s = []
                lrb_l1_errs = []
                lrb_l2s = []
                lrb_l2_errs = []

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

                    L1, L1_err = rb.data["fit_data"]["state_2_L1"]
                    L2, L2_err = rb.data["fit_data"]["state_2_L2"]
                    lrb_l1s.append(L1)
                    lrb_l1_errs.append(L1_err)
                    lrb_l2s.append(L2)
                    lrb_l2_errs.append(L2_err)
                    print(
                        f"      LRB average {avg_idx + 1}/{experiment_dict['Leakage_RB']['num_averages']}: L1={L1:.4e}, L2={L2:.4e}"
                    )

                # Average the results
                L1 = np.mean(lrb_l1s)
                L1_std = np.std(lrb_l1s)
                L1_err = np.mean(lrb_l1_errs)
                L2 = np.mean(lrb_l2s)
                L2_std = np.std(lrb_l2s)
                L2_err = np.mean(lrb_l2_errs)
                lambda1 = 1.0 - L1 - L2

                drag_results.append(
                    (beta, L1, L1_err, L2, L2_err, lambda1, cal_amp, cal_IF_freq)
                )
                lrb_l1_stds.append(L1_std)
                lrb_l2_stds.append(L2_std)
                beta_elapsed = time.time() - beta_start
                print(
                    f"    LRB done — L1={L1:.4e} ± {L1_err:.4e}  L2={L2:.4e} ± {L2_err:.4e}  "
                    f"λ1={lambda1:.4f}  ({beta_elapsed:.0f}s)"
                )
        else:
            drag_results.append(
                (beta, np.nan, np.nan, np.nan, np.nan, np.nan, cal_amp, cal_IF_freq)
            )
            lrb_l1_stds.append(np.nan)
            lrb_l2_stds.append(np.nan)

        # ── Run T1 and T2E once before first RB ────────────────────────────────────
        if DO_COHERENCE_MEASUREMENTS and not coherence_measured:
            print(f"\n  Running coherence measurements (T1, T2E)...")

            # T1 measurement
            t1_cfg = experiment_dict["T1"]
            t1_vec = np.linspace(
                t1_cfg["t_min"], t1_cfg["t_max"], t1_cfg["num_points"], dtype=int
            )
            t1_exp = T1(
                QUBIT,
                t1_vec,
                t1_cfg["shots"],
                CALIB_PARAMS,
                outer_folder=pl_folder,
                path="coherence",
                suffix="T1",
                meta_dict=meta_dict,
                config=config,
                num_pulses=1,
            )
            print(f"    T1 measurement complete")

            # T2E measurement
            t2e_cfg = experiment_dict["T2E"]
            t2e_vec = np.linspace(
                t2e_cfg["t_min"], t2e_cfg["t_max"], t2e_cfg["num_points"], dtype=int
            )
            t2e_exp = T2E(
                QUBIT,
                t2e_vec,
                t2e_cfg["shots"],
                CALIB_PARAMS,
                outer_folder=pl_folder,
                path="coherence",
                suffix="T2E",
                meta_dict=meta_dict,
                config=config,
            )
            print(f"    T2E measurement complete")

            coherence_measured = True

        # Vanilla RB — run NUM_AVERAGES times
        # When DO_RB_3STATE, use 3-state readout (renormalize over {0,1}); fall back to 2-state
        # if the 3-level cal was unavailable for this beta.
        rb_3state_calib = calib_3L if (DO_RB_3STATE and calib_3L is not None) else None
        if DO_RB_3STATE and rb_3state_calib is None:
            print(
                "    DO_RB_3STATE set but no 3-level cal available — using 2-state RB readout"
            )
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
            print(
                f"      RB average {avg_idx + 1}/{NUM_AVERAGES}: fidelity = {_rb_fid:.6f}"
            )

        rb_fidelity_mean = np.mean(rb_fidelities)
        rb_fidelity_std = np.std(rb_fidelities)
        rb_error_rate_mean = 1.0 - rb_fidelity_mean
        rb_error_rate_std = rb_fidelity_std
        rb_error_rates.append(rb_error_rate_mean)
        rb_error_rate_errs.append(rb_error_rate_std)

        # Store this beta's calibrated state so purity RB can re-use the best one
        # without recalibrating. Kept in lockstep with rb_error_rates / drag_results.
        beta_configs.append(
            {
                "meta": copy.deepcopy(meta_dict),
                "calib": copy.deepcopy(CALIB_PARAMS),
            }
        )

        # Real-time monitoring: re-save the DRAG figure + H5 after every beta so the
        # outputs reflect all completed betas mid-sweep.
        save_drag_summary(
            pl_folder,
            QUBIT,
            pulse_length_ns,
            DO_LEAKAGE_CHARACTERIZATION,
            drag_results,
            rb_error_rates,
            rb_error_rate_errs,
        )

    # Carry this pulse length's beta=0 amplitude forward to bound the next (slower)
    # chevron. Keep the prior bound if every beta failed (beta0_amp_this_pl is None).
    if beta0_amp_this_pl is not None:
        prev_beta0_amp = beta0_amp_this_pl

    # ── Per-pulse-length DRAG summary ─────────────────────────────────────────
    # Arrays kept for best-beta selection below. The H5 + figure were already written
    # incrementally after each beta (save_drag_summary); this final call ensures the
    # complete set is on disk.
    betas = np.array([r[0] for r in drag_results])
    l1s = np.array([r[1] for r in drag_results])
    l1errs = np.array([r[2] for r in drag_results])
    l2s = np.array([r[3] for r in drag_results])
    l2errs = np.array([r[4] for r in drag_results])
    lambda1s = np.array([r[5] for r in drag_results])

    rb_err_arr = np.array(rb_error_rates)
    rb_err_std_arr = np.array(rb_error_rate_errs)

    drag_fig_path = save_drag_summary(
        pl_folder,
        QUBIT,
        pulse_length_ns,
        DO_LEAKAGE_CHARACTERIZATION,
        drag_results,
        rb_error_rates,
        rb_error_rate_errs,
    )
    print(f"  DRAG sweep figure saved to {drag_fig_path}")

    # ── Find best beta (minimum RB error rate) ────────────────────────────────
    best_idx = int(np.argmin(rb_err_arr))
    best = drag_results[best_idx]

    # RB results for best beta were already computed during the beta loop
    best_beta = best[0]
    rb_error_rate_mean = rb_error_rates[best_idx]
    rb_error_rate_std = rb_error_rate_errs[best_idx]
    print(
        f"\n  Best beta={best_beta:.3f} RB error rate (from {NUM_AVERAGES} averages): {rb_error_rate_mean:.6e} ± {rb_error_rate_std:.6e}"
    )

    # ── Purity / unitarity RB at the best beta (once) ─────────────────────────
    # Reuse the best beta's calibrated meta_dict + readout params; no recalibration.
    u_val = np.nan
    u_err = np.nan
    irb_gate_fidelities = {g: np.nan for g in _IRB_GATES}
    irb_gate_fidelity_errors = {g: np.nan for g in _IRB_GATES}
    r_coh_gate = np.nan
    r_coh_err = np.nan
    r_incoh_gate = np.nan
    r_incoh_err = np.nan
    if DO_PURITY_RB:
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

        # Recover the per-Clifford depolarizing parameter from the survival RB at
        # best beta:  r_total_gate = (1 - p) / 2 / GATES_PER_CLIFFORD
        #   => p = 1 - 2 * GATES_PER_CLIFFORD * r_total_gate
        p_dep = 1.0 - 2.0 * GATES_PER_CLIFFORD * rb_error_rate_mean

        if np.isfinite(u_val) and u_val > 0:
            sqrt_u = np.sqrt(u_val)
            # per-Clifford -> per-gate (divide by GATES_PER_CLIFFORD), d = 2
            r_coh_gate = max(sqrt_u - p_dep, 0.0) / 2.0 / GATES_PER_CLIFFORD
            r_incoh_gate = max(1.0 - sqrt_u, 0.0) / 2.0 / GATES_PER_CLIFFORD

            # Error propagation (rough): sqrt(u) error from u_err, p error from RB std
            d_sqrt_u = (
                (u_err / (2.0 * sqrt_u)) if (np.isfinite(u_err) and sqrt_u > 0) else 0.0
            )
            d_p = (
                2.0 * GATES_PER_CLIFFORD * rb_error_rate_std
                if np.isfinite(rb_error_rate_std)
                else 0.0
            )
            r_coh_err = np.sqrt(d_sqrt_u**2 + d_p**2) / 2.0 / GATES_PER_CLIFFORD
            r_incoh_err = d_sqrt_u / 2.0 / GATES_PER_CLIFFORD

            print(
                f"    u={u_val:.6f}±{u_err:.6f}  p={p_dep:.6f}  "
                f"r_coh={r_coh_gate:.3e}  r_incoh={r_incoh_gate:.3e}  (per gate)"
            )
        else:
            print(
                "    Purity RB returned no valid unitarity; budget entries set to NaN."
            )

    # ── Interleaved RB at the best beta (once) ────────────────────────────────
    # Runs one reference RB + one IRB per gate; no recalibration needed.
    if DO_IRB:
        print(f"\n  Running interleaved RB at best beta={best_beta:.3f} ...")
        _irb_cfg = experiment_dict["Interleaved_RB"]
        best_state_irb = beta_configs[best_idx]
        best_meta_irb = copy.deepcopy(best_state_irb["meta"])
        best_calib_irb = best_state_irb["calib"]
        best_config_irb = meta_dict_to_config_1Q(best_meta_irb)

        irb_exp = RB1Q_Interleaved(
            QUBIT,
            _irb_cfg["circuit_depths"],
            _irb_cfg["num_circuits"],
            _irb_cfg["shots"],
            best_calib_irb,
            meta_dict=best_meta_irb,
            outer_folder=pl_folder,
            path=f"beta{best_beta:+.3f}".replace(".", "p"),
            suffix="IRB_best",
            config=best_config_irb,
            gates=_irb_cfg["gates"],
            flat_path=True,
        )
        irb_gate_fidelities = irb_exp.data.get("gate_fidelities", {})
        irb_gate_fidelity_errors = irb_exp.data.get("gate_fidelity_errors", {})
        for gate in _irb_cfg["gates"]:
            fid = irb_gate_fidelities.get(gate, np.nan)
            err = irb_gate_fidelity_errors.get(gate, np.nan)
            print(f"    IRB  F({gate}) = {fid:.6f} ± {err:.6f}")

    sweep_results.append(
        (
            pulse_length_ns,
            best[0],  # beta
            best[1],  # L1
            best[2],  # L1_err
            best[3],  # L2
            best[4],  # L2_err
            best[5],  # lambda1
            best[6],  # amp
            best[7],  # IF_freq_Hz
            rb_error_rate_mean,
            rb_error_rate_std,
            r_coh_gate,
            r_coh_err,
            r_incoh_gate,
            r_incoh_err,
            u_val,
            u_err,
        )
    )

    pl_elapsed = time.time() - pl_start
    print(f"\n  Pulse length {pulse_length_ns} ns complete in {pl_elapsed:.0f}s")
    print(
        f"  Best: beta={best[0]:.3f}  L1={best[1]:.4e}  L2={best[3]:.4e}  "
        f"λ1={best[5]:.4f}  amp={best[6]:.6f} V  "
        f"IF_freq={best[7] / u.MHz:.6f} MHz  "
        f"RB_error={rb_error_rate_mean:.6e} ± {rb_error_rate_std:.6e}"
    )

    # ── Update global H5 file ─────────────────────────────────────────────────
    with h5py.File(global_h5_path, "a") as f:
        # Resize all datasets
        n = len(f["pulse_lengths_ns"]) + 1
        for key in (
            "pulse_lengths_ns",
            "best_beta",
            "rb_error_rate",
            "rb_error_rate_std",
            "L1",
            "L1_err",
            "L2",
            "L2_err",
            "mixer_off_I",
            "mixer_off_Q",
            "mixer_gain",
            "mixer_phase",
            "r_coh",
            "r_coh_err",
            "r_incoh",
            "r_incoh_err",
            "unitarity",
            "unitarity_err",
        ):
            f[key].resize((n,))
        for _g in _IRB_GATES:
            f[f"irb_F_{_g}"].resize((n,))
            f[f"irb_F_{_g}_err"].resize((n,))

        # Write data
        f["pulse_lengths_ns"][-1] = pulse_length_ns
        f["best_beta"][-1] = abs(best_beta)  # magnitude of beta
        f["rb_error_rate"][-1] = rb_error_rate_mean
        f["rb_error_rate_std"][-1] = rb_error_rate_std
        f["L1"][-1] = best[1]
        f["L1_err"][-1] = best[2]
        f["L2"][-1] = best[3]
        f["L2_err"][-1] = best[4]
        mixer_state = beta_configs[best_idx]["meta"]["q_mixer"]
        f["mixer_off_I"][-1] = mixer_state.get("mixer_off_I", np.nan)
        f["mixer_off_Q"][-1] = mixer_state.get("mixer_off_Q", np.nan)
        f["mixer_gain"][-1] = mixer_state.get("mixer_gain", np.nan)
        f["mixer_phase"][-1] = mixer_state.get("mixer_phase", np.nan)
        f["r_coh"][-1] = r_coh_gate
        f["r_coh_err"][-1] = r_coh_err
        f["r_incoh"][-1] = r_incoh_gate
        f["r_incoh_err"][-1] = r_incoh_err
        f["unitarity"][-1] = u_val
        f["unitarity_err"][-1] = u_err
        for _g in _IRB_GATES:
            f[f"irb_F_{_g}"][-1] = irb_gate_fidelities.get(_g, np.nan)
            f[f"irb_F_{_g}_err"][-1] = irb_gate_fidelity_errors.get(_g, np.nan)

    # Append to tracking lists for live plot (avoids HDF5 file locking)
    pls_list.append(pulse_length_ns)
    rb_errs_list.append(rb_error_rate_mean)
    rb_stds_list.append(rb_error_rate_std)
    l1s_list.append(best[1])
    l1_errs_list.append(best[2])
    l2s_list.append(best[3])
    l2_errs_list.append(best[4])
    lambda1s_list.append(best[5])
    mixer_state = beta_configs[best_idx]["meta"]["q_mixer"]
    mixer_i_list.append(mixer_state.get("mixer_off_I", np.nan))
    mixer_q_list.append(mixer_state.get("mixer_off_Q", np.nan))
    mixer_gain_list.append(mixer_state.get("mixer_gain", np.nan))
    mixer_phase_list.append(mixer_state.get("mixer_phase", np.nan))

    # ── Update live / final 4x1 summary vs gate time ───────────────────────────
    pls_live = np.array(pls_list)
    rb_errs_live = np.array(rb_errs_list)
    rb_stds_live = np.array(rb_stds_list)
    l1s_live = np.array(l1s_list)
    l1_errs_live = np.array(l1_errs_list)
    l2s_live = np.array(l2s_list)
    l2_errs_live = np.array(l2_errs_list)
    one_minus_lambda1_live = 1.0 - np.array(lambda1s_list)

    for a in axes:
        a.clear()

    # Panels 1-3: L1, L2, 1 - lambda1 vs gate time
    for a, ys, yerrs, ylabel, ylim_bottom in zip(
        axes[:3],
        [l1s_live, l2s_live, one_minus_lambda1_live],
        [l1_errs_live, l2_errs_live, np.zeros(len(one_minus_lambda1_live))],
        [
            "L1  (leakage per Clifford)",
            "L2  (leakage per Clifford)",
            "1 - λ1",
        ],
        [1e-6, None, 1e-5],
    ):
        a.errorbar(pls_live, ys, yerr=yerrs, fmt="o-", capsize=4, markersize=8)
        a.set_yscale("log")
        a.set_ylim(bottom=ylim_bottom)
        a.set_ylabel(ylabel, fontsize=12)
        a.grid(True, which="both", alpha=0.3)

    # Panel 4: total infidelity and leakage infidelity (L1) on the same axes
    axes[3].errorbar(
        pls_live,
        rb_errs_live,
        yerr=rb_stds_live,
        fmt="o-",
        capsize=4,
        markersize=8,
        label="Total infidelity (RB)",
    )
    axes[3].errorbar(
        pls_live,
        l1s_live,
        yerr=l1_errs_live,
        fmt="s-",
        capsize=4,
        markersize=8,
        label="Leakage infidelity (L1)",
    )
    axes[3].set_yscale("log")
    axes[3].set_ylim(bottom=1e-5)
    axes[3].set_ylabel("Error rate", fontsize=12)
    axes[3].legend(fontsize=11)
    axes[3].grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Gate time (ns)", fontsize=12)
    fig.suptitle(f"{QUBIT} — Leakage and RB Error vs Gate Time")
    fig.tight_layout()

    # Save plot (overwritten each pulse length; final write is the complete summary)
    plot_path = os.path.join(parent_folder, f"{QUBIT}_summary_vs_gate_time.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"  Summary plot saved to {plot_path}")

    mixer_fig, mixer_axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    mixer_axes[0].plot(pls_list, mixer_i_list, "o-", label="I offset")
    mixer_axes[1].plot(pls_list, mixer_q_list, "o-", label="Q offset")
    mixer_axes[2].plot(pls_list, mixer_gain_list, "o-", label="Gain")
    mixer_axes[3].plot(pls_list, mixer_phase_list, "o-", label="Phase")
    for ax in mixer_axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("Value")
    mixer_axes[-1].set_xlabel("Gate time (ns)")
    mixer_fig.suptitle(f"{QUBIT} — mixer tuning values vs gate time")
    mixer_fig.tight_layout()
    mixer_plot_path = os.path.join(
        parent_folder, f"{QUBIT}_mixer_tuning_vs_gate_time.png"
    )
    mixer_fig.savefig(mixer_plot_path, bbox_inches="tight", dpi=150)
    plt.close(mixer_fig)
    print(f"  Mixer tuning plot saved to {mixer_plot_path}")

    # ── Error-budget plot: total vs leakage vs coherent vs incoherent ──────────
    if DO_PURITY_RB:
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
        ax_bud.errorbar(
            pls_b,
            rb_b,
            yerr=rb_b_err,
            fmt="o-",
            capsize=5,
            label="Total error (RB)",
            linewidth=2.5,
            markersize=9,
        )
        ax_bud.errorbar(
            pls_b,
            l1_b,
            yerr=l1_b_err,
            fmt="s-",
            capsize=5,
            label="Leakage error (L1)",
            linewidth=2.5,
            markersize=9,
        )
        ax_bud.errorbar(
            pls_b,
            rcoh_b,
            yerr=rcoh_b_err,
            fmt="^-",
            capsize=5,
            label="Coherent error (PRB)",
            linewidth=2.5,
            markersize=9,
        )
        ax_bud.errorbar(
            pls_b,
            rincoh_b,
            yerr=rincoh_b_err,
            fmt="v-",
            capsize=5,
            label="Incoherent error (PRB)",
            linewidth=2.5,
            markersize=9,
        )
        ax_bud.set_yscale("log")
        ax_bud.set_ylim(bottom=1e-6)
        ax_bud.set_ylabel("Error rate (per gate)", fontsize=13)
        ax_bud.set_xlabel("Gate time (ns)", fontsize=13)
        ax_bud.legend(fontsize=11, loc="best")
        ax_bud.grid(True, which="both", alpha=0.3)
        ax_bud.set_title(
            f"{QUBIT} — Error Budget vs Gate Time  (coherent vs incoherent)",
            fontsize=13,
        )
        fig_bud.tight_layout()

        budget_path = os.path.join(
            parent_folder, f"{QUBIT}_error_budget_vs_gate_time.png"
        )
        fig_bud.savefig(budget_path, bbox_inches="tight", dpi=150)
        print(f"  Error-budget plot saved to {budget_path}")
        plt.close(fig_bud)

    # ── IRB gate fidelity vs gate time ─────────────────────────────────────────
    if DO_IRB:
        with h5py.File(global_h5_path, "r") as f:
            pls_irb = f["pulse_lengths_ns"][:]
            irb_fids = {g: f[f"irb_F_{g}"][:] for g in _IRB_GATES}
            irb_errs = {g: f[f"irb_F_{g}_err"][:] for g in _IRB_GATES}

        fig_irb, ax_irb = plt.subplots(1, 1, figsize=(8, 6))
        colors_irb = plt.cm.tab10(np.linspace(0, 0.9, len(_IRB_GATES)))
        for gate, color in zip(_IRB_GATES, colors_irb):
            fids = irb_fids[gate]
            errs = irb_errs[gate]
            ax_irb.errorbar(
                pls_irb,
                1.0 - fids,
                yerr=errs,
                fmt="o-",
                capsize=5,
                color=color,
                label=gate,
                linewidth=2,
                markersize=7,
            )
        ax_irb.set_yscale("log")
        ax_irb.set_ylim(bottom=1e-6)
        ax_irb.set_ylabel("Gate error rate  (1 − F)", fontsize=13)
        ax_irb.set_xlabel("Gate time (ns)", fontsize=13)
        ax_irb.legend(fontsize=11, loc="best")
        ax_irb.grid(True, which="both", alpha=0.3)
        ax_irb.set_title(f"{QUBIT} — IRB Gate Fidelity vs Gate Time", fontsize=13)
        fig_irb.tight_layout()
        irb_plot_path = os.path.join(
            parent_folder, f"{QUBIT}_irb_gate_fidelity_vs_gate_time.png"
        )
        fig_irb.savefig(irb_plot_path, bbox_inches="tight", dpi=150)
        print(f"  IRB plot saved to {irb_plot_path}")
        plt.close(fig_irb)


# ─── Sweep complete ──────────────────────────────────────────────────────────

total_elapsed = time.time() - sweep_start
print(f"\n{'=' * 60}")
print(f"All pulse lengths complete in {total_elapsed:.0f}s.")
print(f"Summary saved to: {global_h5_path}")
print(f"Live plot saved to: {plot_path}")
print(f"{'=' * 60}")

plt.close(fig)
