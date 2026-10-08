"""
1Q Gate Stability Monitor

Runs repeated gate calibration + coherence + RB measurements over many iterations
to monitor qubit stability and gate performance over time.

For each iteration:
  - Gate calibration (pulse_length=32ns, beta=0)
  - T1 measurement
  - T2E measurement
  - Vanilla RB (NUM_AVERAGES=3)
  - Leakage RB (NUM_AVERAGES=3)

Results saved to HDF5 and plotted in real-time (3x1: T1, T2E, RB+LRB).
"""

import os
import time
import traceback
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
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage
from LabCode.Experiments.RB.m_RB_1Q import RB1Q
from LabCode.Experiments.Coherence.m_T1 import T1
from LabCode.Experiments.Coherence.m_T2E import T2E
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

# ─── Configuration ────────────────────────────────────────────────────────────
PULSE_TYPE = "X90"
QUBIT = "q2"
NUM_ITERATIONS = 1000
NUM_AVERAGES = 3

PULSE_LENGTH_NS = 24
DRAG_BETA = 0.0

optimizer_convergence_params = {
    "amplitude_accuracy": 0.0005,
    "detuning_accuracy": 0.05,  # MHz
    "max_iter": 3,
}

experiment_dict = {
    "SS_Cal": {
        "shots": 1000,
        "number_pi_pulses": 1,
        "ground_threshold": 0.7,
    },
    "SS_Cal_3L": {
        "shots": 1000,
    },
    "Chevron": {
        "shots": 20,
        "amp_min": 0.0,
        "amp_max": 0.25,
        "amp_points": 31,
        "freq_span": 10 * u.MHz,
        "freq_points": 31,
    },
    "Amplified_Amplitude_Error": {
        "shots": 300,
        "N_min": 1,
        "N_step": 1,
        "N_max_stages": [11, 31],
    },
    "Frequency_Calibration_XY": {
        "shots": 100,
        "pulse_sequence": [
            ("X90", "I"),
            ("Y90", "I"),
            ("X90", "Y90"),
            ("Y90", "X90"),
        ],
        "freq_span_stages": [3 * u.MHz, 1 * u.MHz],
        "freq_points": 301,
        "num_repeats": 1,
    },
    "Leakage_RB": {
        "shots": 25,
        "num_circuits": 25,
        "num_averages": NUM_AVERAGES,
        "circuit_lengths": [
            2,
            16,
            64,
            128,
            500,
            600,
            1000,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
        ],
    },
    "RB1Q": {
        "shots": 50,
        "num_circuits": 50,
        "circuit_lengths": [
            8,
            32,
            128,
            256,
            500,
            700,
            900,
            1200,
            1500,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
        ],
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


# ─── Calibration routine ──────────────────────────────────────────────────────
def _run_calibration(
    qubit,
    meta_dict,
    config,
    calib_params,
    pulse_type,
    exp_dict,
    conv_params,
    label,
    save_path,
):
    """Gate calibration: chevron + staged AAE + frequency calibration."""
    chev = exp_dict["Chevron"]
    amp_vec = np.linspace(chev["amp_min"], chev["amp_max"], chev["amp_points"])
    f_vec = np.linspace(
        -chev["freq_span"] / 2, chev["freq_span"] / 2, chev["freq_points"], dtype=int
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
        suffix=f"Chevron_{label}",
        config=config,
        flat_path=True,
    )
    opt_detuning_Hz = chevron.data["opt_detuning_Hz"]
    opt_amp = chevron.data["opt_amp"]
    meta_dict["IF_freq"] += int(opt_detuning_Hz)
    meta_dict["X90"]["amp"] = opt_amp
    meta_dict["Y90"]["amp"] = opt_amp
    config = meta_dict_to_config_1Q(meta_dict)

    max_iter = conv_params["max_iter"]
    N_max_stages = exp_dict["Amplified_Amplitude_Error"]["N_max_stages"]
    freq_span_stages = exp_dict["Frequency_Calibration_XY"]["freq_span_stages"]

    for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
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
                suffix=f"AAE_{label}_s{stage_idx}_iter{i:02d}",
                config=config,
                flat_path=True,
            )
            amp_err = aae.data["amplitude_error"]
            meta_dict["X90"]["amp"] += amp_err
            meta_dict["Y90"]["amp"] += amp_err
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
                suffix=f"FreqCal_{label}_s{stage_idx}_iter{i:02d}",
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

            if all(calibrated.values()):
                break

    return config


# ─── Create parent folder ─────────────────────────────────────────────────────
_base_meta = make_meta_dict(QUBIT)[QUBIT]
outer_folder_base = _base_meta["global_params"]["outer_folder"]
_date_str = datetime.now().strftime("%Y_%m_%d")
_q_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{_date_str}")
os.makedirs(_q_date_dir, exist_ok=True)
parent_folder = os.path.join(
    _q_date_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gate_time_stats"
)
os.makedirs(parent_folder, exist_ok=True)
print(f"Gate time stats folder: {parent_folder}")

# ─── Initialize results storage ────────────────────────────────────────────────
iterations = []
wall_clock_times = []  # seconds since start
t1_values = []
t1_errors = []
t2e_values = []
t2e_errors = []
rb_error_rates = []
rb_error_stds = []
lrb_l1_rates = []
lrb_l1_stds = []

summary_path = os.path.join(parent_folder, f"{QUBIT}_gate_time_stats.h5")


# ─── Helper functions for live saving ───────────────────────────────────────
def _save_summary_h5(
    summary_path,
    iterations,
    wall_clock_times,
    t1_values,
    t1_errors,
    t2e_values,
    t2e_errors,
    rb_error_rates,
    rb_error_stds,
    lrb_l1_rates,
    lrb_l1_stds,
):
    with h5py.File(summary_path, "w") as f:
        f.create_dataset("iterations", data=np.array(iterations))
        f.create_dataset("wall_clock_s", data=np.array(wall_clock_times))
        f.create_dataset("T1_us", data=np.array(t1_values))
        f.create_dataset("T1_err_us", data=np.array(t1_errors))
        f.create_dataset("T2E_us", data=np.array(t2e_values))
        f.create_dataset("T2E_err_us", data=np.array(t2e_errors))
        f.create_dataset("RB_error_rate", data=np.array(rb_error_rates))
        f.create_dataset("RB_error_rate_std", data=np.array(rb_error_stds))
        f.create_dataset("LRB_L1_error_rate", data=np.array(lrb_l1_rates))
        f.create_dataset("LRB_L1_error_rate_std", data=np.array(lrb_l1_stds))
        f.attrs["qubit"] = QUBIT
        f.attrs["num_iterations"] = NUM_ITERATIONS
        f.attrs["num_averages"] = NUM_AVERAGES
        f.attrs["pulse_length_ns"] = PULSE_LENGTH_NS
        f.attrs["drag_beta"] = DRAG_BETA
        f.attrs["last_updated"] = datetime.now().isoformat()
        f.flush()


def _save_live_figure(
    fig,
    axes,
    parent_folder,
    time_arr,
    t1_values,
    t1_errors,
    t2e_values,
    t2e_errors,
    rb_error_rates,
    rb_error_stds,
    lrb_l1_rates,
    lrb_l1_stds,
):
    def _plot_series(ax, x, y, yerr, ylabel, title, marker, label=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            x_valid = x[valid]
            y_valid = y[valid]
            yerr_valid = np.where(np.isfinite(yerr[valid]), yerr[valid], 0.0)
            ax.errorbar(
                x_valid,
                y_valid,
                yerr=yerr_valid,
                fmt=marker,
                linestyle="-",
                linewidth=1.2,
                markersize=5,
                capsize=4,
                capthick=1.0,
                elinewidth=1.0,
                label=label,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Wall clock time (hours)")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlim(left=0)

    axes[0].clear()
    _plot_series(
        axes[0],
        time_arr,
        t1_values,
        t1_errors,
        "T1 (µs)",
        "T1 Relaxation Time",
        "o-",
    )

    axes[1].clear()
    _plot_series(
        axes[1],
        time_arr,
        t2e_values,
        t2e_errors,
        "T2E (µs)",
        "T2E Dephasing Time",
        "s-",
    )

    axes[2].clear()
    _plot_series(
        axes[2],
        time_arr,
        rb_error_rates,
        rb_error_stds,
        "Error rate (per physical gate)",
        "RB & Leakage RB Error Rates",
        "o-",
        label="RB total error",
    )
    _plot_series(
        axes[2],
        time_arr,
        lrb_l1_rates,
        lrb_l1_stds,
        "Error rate (per physical gate)",
        "RB & Leakage RB Error Rates",
        "s-",
        label="LRB L1 error",
    )
    axes[2].set_yscale("log")
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Wall clock time (hours)")

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
    fig.savefig(
        os.path.join(parent_folder, f"{QUBIT}_gate_time_stats.png"),
        bbox_inches="tight",
        dpi=150,
    )


# ─── Main iteration loop ───────────────────────────────────────────────────────
sweep_start = time.time()
print(f"\nGate time stats started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create figure for live plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle(
    f"{QUBIT} — Gate Stability Monitor (β={DRAG_BETA}, pulse_len={PULSE_LENGTH_NS}ns)"
)

for iteration in range(NUM_ITERATIONS):
    iter_start = time.time()
    wall_clock_s = iter_start - sweep_start
    iter_folder = os.path.join(parent_folder, f"iter_{iteration:03d}")
    os.makedirs(iter_folder, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"Iteration {iteration + 1}/{NUM_ITERATIONS}  |  {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"{'=' * 60}")

    try:
        # ── Gate calibration ──────────────────────────────────────────────────────
        meta_dict = make_meta_dict(QUBIT)[QUBIT]
        meta_dict["DRAG_beta"] = DRAG_BETA
        meta_dict["X90"]["length"] = PULSE_LENGTH_NS
        meta_dict["Y90"]["length"] = PULSE_LENGTH_NS
        meta_dict["global_params"]["outer_folder"] = iter_folder
        config = meta_dict_to_config_1Q(meta_dict)

        # 2-level SS calibration
        max_retries_2L = 10
        two_level_success = False
        for retry_idx in range(max_retries_2L):
            SS = SingleShot1Q(
                QUBIT,
                experiment_dict["SS_Cal"]["shots"],
                meta_dict=meta_dict,
                outer_folder=iter_folder,
                path="cal",
                suffix="SS_Cal",
                repeats=experiment_dict["SS_Cal"]["number_pi_pulses"],
                config=config,
                confidence_threshold=experiment_dict["SS_Cal"]["ground_threshold"],
                flat_path=True,
            )
            print(f"  SS Cal — fidelity {SS.max_F:.3f}")
            if SS.max_F > 0.70:
                two_level_success = True
                break
            if retry_idx < max_retries_2L - 1:
                print("  Fidelity below 0.70, retrying...")
        if not two_level_success:
            raise RuntimeError("2-level SS calibration failed")

        CALIB_PARAMS = SS.calib_params
        meta_dict["read_theta"] = SS.calib_params["read_theta"]
        config = meta_dict_to_config_1Q(meta_dict)

        # Gate calibration
        config = _run_calibration(
            QUBIT,
            meta_dict,
            config,
            CALIB_PARAMS,
            PULSE_TYPE,
            experiment_dict,
            optimizer_convergence_params,
            label="cal",
            save_path="cal",
        )

        # 3-level SS calibration
        max_retries_3L = 30
        three_level_success = False
        for retry_idx in range(max_retries_3L):
            SS3 = SingleShot1Q3Level(
                QUBIT,
                experiment_dict["SS_Cal_3L"]["shots"],
                meta_dict=meta_dict,
                outer_folder=iter_folder,
                path="cal",
                suffix="SS_Cal_3L",
                config=config,
                flat_path=True,
            )
            if SS3.max_F > 0.65:
                three_level_success = True
                break
        if not three_level_success:
            raise RuntimeError("3-level SS calibration failed")

        calib_3L = SS3.calib_params

        # ── T1 measurement ────────────────────────────────────────────────────────
        t1_cfg = experiment_dict["T1"]
        t1_vec = np.logspace(
            np.log10(t1_cfg["t_min"]),
            np.log10(t1_cfg["t_max"]),
            t1_cfg["t_points"],
            dtype=int,
        )
        t1_exp = T1(
            QUBIT,
            t1_vec,
            t1_cfg["shots"],
            CALIB_PARAMS,
            outer_folder=iter_folder,
            path="coherence",
            suffix="T1",
            meta_dict=meta_dict,
            config=config,
            num_pulses=1,
        )
        t1_val, t1_err = t1_exp.data.get("T1", (np.nan, np.nan))
        print(f"  T1 = {t1_val:.3f} ± {t1_err:.3f} µs")

        # ── T2E measurement ───────────────────────────────────────────────────────
        t2e_cfg = experiment_dict["T2E"]
        t2e_vec = np.logspace(
            np.log10(t2e_cfg["t_min"]),
            np.log10(t2e_cfg["t_max"]),
            t2e_cfg["t_points"],
            dtype=int,
        )
        t2e_exp = T2E(
            QUBIT,
            t2e_vec,
            t2e_cfg["shots"],
            CALIB_PARAMS,
            outer_folder=iter_folder,
            path="coherence",
            suffix="T2E",
            meta_dict=meta_dict,
            config=config,
        )
        t2e_val, t2e_err = t2e_exp.data.get("T2E", (np.nan, np.nan))
        print(f"  T2E = {t2e_val:.3f} ± {t2e_err:.3f} µs")

        # ── Vanilla RB × NUM_AVERAGES ────────────────────────────────────────────
        rb_fidelities = []
        for avg_idx in range(NUM_AVERAGES):
            rb1q = RB1Q(
                QUBIT,
                experiment_dict["RB1Q"]["circuit_lengths"],
                experiment_dict["RB1Q"]["num_circuits"],
                experiment_dict["RB1Q"]["shots"],
                CALIB_PARAMS,
                meta_dict=meta_dict,
                outer_folder=iter_folder,
                path="rb",
                suffix=f"RB1Q_avg{avg_idx:02d}",
                config=config,
                simulate=False,
                flat_path=True,
            )
            _fid = rb1q.data.get("gate_fidelity", float("nan"))
            rb_fidelities.append(_fid)
        rb_fid_mean = np.mean(rb_fidelities)
        rb_fid_std = np.std(rb_fidelities)
        rb_err_mean = 1.0 - rb_fid_mean
        rb_err_std = rb_fid_std
        print(f"  RB error rate = {rb_err_mean:.6e} ± {rb_err_std:.6e}")

        # ── Leakage RB × NUM_AVERAGES ────────────────────────────────────────────
        lrb_l1s = []
        for avg_idx in range(NUM_AVERAGES):
            rb = RB1Q_Leakage(
                QUBIT,
                experiment_dict["Leakage_RB"]["circuit_lengths"],
                experiment_dict["Leakage_RB"]["num_circuits"],
                experiment_dict["Leakage_RB"]["shots"],
                calib_3L,
                meta_dict=meta_dict,
                outer_folder=iter_folder,
                path="lrb",
                suffix=f"LRB_avg{avg_idx:02d}",
                config=config,
                flat_path=True,
            )
            fit_data = rb.data.get("fit_data", {})
            if "state_2_L1" in fit_data:
                L1, L1_err = fit_data["state_2_L1"]
            else:
                L1, L1_err = fit_data.get("state_2_L1_per_gate", (np.nan, np.nan))
            lrb_l1s.append(L1)
        lrb_l1_mean = np.nanmean(lrb_l1s)
        lrb_l1_std = np.nanstd(lrb_l1s)
        print(f"  LRB L1 error rate = {lrb_l1_mean:.6e} ± {lrb_l1_std:.6e}")

        # ── Store results ─────────────────────────────────────────────────────────
        iterations.append(iteration)
        wall_clock_times.append(wall_clock_s)
        t1_values.append(t1_val if not np.isnan(t1_val) else np.nan)
        t1_errors.append(t1_err if not np.isnan(t1_err) else np.nan)
        t2e_values.append(t2e_val if not np.isnan(t2e_val) else np.nan)
        t2e_errors.append(t2e_err if not np.isnan(t2e_err) else np.nan)
        rb_error_rates.append(rb_err_mean)
        rb_error_stds.append(rb_err_std)
        lrb_l1_rates.append(lrb_l1_mean)
        lrb_l1_stds.append(lrb_l1_std)

        print(
            f"  SAVED ITER {iteration}: T1={t1_val:.3f}±{t1_err:.3f}, "
            f"T2E={t2e_val:.3f}±{t2e_err:.3f}, "
            f"RB={rb_err_mean:.3e}±{rb_err_std:.3e}, "
            f"LRB={lrb_l1_mean:.3e}±{lrb_l1_std:.3e}"
        )

        # ── Save HDF5 and update live plot ────────────────────────────────────
        time_arr = np.array(wall_clock_times) / 3600  # convert to hours
        _save_summary_h5(
            summary_path,
            iterations,
            wall_clock_times,
            t1_values,
            t1_errors,
            t2e_values,
            t2e_errors,
            rb_error_rates,
            rb_error_stds,
            lrb_l1_rates,
            lrb_l1_stds,
        )
        _save_live_figure(
            fig,
            axes,
            parent_folder,
            time_arr,
            t1_values,
            t1_errors,
            t2e_values,
            t2e_errors,
            rb_error_rates,
            rb_error_stds,
            lrb_l1_rates,
            lrb_l1_stds,
        )

        iter_elapsed = time.time() - iter_start
        print(f"Iteration {iteration + 1} completed in {iter_elapsed:.0f}s")

    except Exception as e:
        # If iteration fails, fill results with NaN and continue
        print(
            f"  ERROR in iteration {iteration + 1}: {type(e).__name__}: {str(e)[:100]}"
        )
        print(traceback.format_exc())
        print(f"  Skipping iteration, filling with NaN values and continuing...")

        iterations.append(iteration)
        wall_clock_times.append(wall_clock_s)
        t1_values.append(np.nan)
        t1_errors.append(np.nan)
        t2e_values.append(np.nan)
        t2e_errors.append(np.nan)
        rb_error_rates.append(np.nan)
        rb_error_stds.append(np.nan)
        lrb_l1_rates.append(np.nan)
        lrb_l1_stds.append(np.nan)

        # Save the current state to disk even when an iteration fails
        time_arr = np.array(wall_clock_times) / 3600
        _save_summary_h5(
            summary_path,
            iterations,
            wall_clock_times,
            t1_values,
            t1_errors,
            t2e_values,
            t2e_errors,
            rb_error_rates,
            rb_error_stds,
            lrb_l1_rates,
            lrb_l1_stds,
        )
        _save_live_figure(
            fig,
            axes,
            parent_folder,
            time_arr,
            t1_values,
            t1_errors,
            t2e_values,
            t2e_errors,
            rb_error_rates,
            rb_error_stds,
            lrb_l1_rates,
            lrb_l1_stds,
        )

# ─── Final summary ────────────────────────────────────────────────────────────
total_elapsed = time.time() - sweep_start
print(f"\n{'=' * 60}")
print(f"All iterations complete in {total_elapsed:.0f}s ({total_elapsed / 3600:.2f}h).")
print(f"Results saved to: {summary_path}")
print(f"Figure saved to: {os.path.join(parent_folder, f'{QUBIT}_gate_time_stats.png')}")
print(f"{'=' * 60}")

plt.ioff()
plt.show()
