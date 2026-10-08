import os
import time
import h5py
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage

from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


PULSE_TYPE = "X90"
QUBIT = "q6"
# ─── DRAG beta sweep vector ───────────────────────────────────────────────────
# Edit this list to set which DRAG beta values to calibrate and benchmark.
DRAG_beta_vec = np.linspace(0, 1.5, 11)

optimizer_convergence_params = {
    "amplitude_accuracy": 0.0001,
    "detuning_accuracy": 0.05,  # MHz — matches the unit returned by optimal_detuning_MHz
    "max_iter": 5,
}

experiment_dict = {
    "SS_Cal": {
        "shots": 1000,
        "number_pi_pulses": 1,
        "ground_threshold": 0.7,
        "object": SingleShot1Q,
    },
    "SS_Cal_3L": {
        "shots": 1000,
        "object": SingleShot1Q3Level,
    },
    "Chevron": {
        "shots": 50,
        "amp_min": 0.0,
        "amp_max": 0.45,
        "amp_points": 21,
        "freq_span": 20 * u.MHz,
        "freq_points": 21,
        "object": RabiChevronSS,
    },
    "Amplified_Amplitude_Error": {
        "shots": 301,
        "N_min": 1,
        "N_step": 1,
        "N_max_stages": [11,101],  # N_max used at each calibration stage
        "object": AAE,
    },
    "Frequency_Calibration_XY": {
        "shots": 301,
        "pulse_sequence": [
            ("X90", "I"),
            ("Y90", "I"),
            ("X90", "Y90"),
            ("Y90", "X90"),
        ],
        "freq_span_stages": [10 * u.MHz, 3 * u.MHz],  # freq_span at each stage
        "freq_points": 51,
        "num_repeats": 1,
        "object": Frequency_Calibration_XY,
    },
    "Leakage_RB": {
        "shots": 10,
        "num_circuits": 150,
        "circuit_lengths": [2, 4, 8, 16, 64, 128, 250, 500, 600, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        "object": RB1Q_Leakage,
    },
}


def _run_calibration(qubit, meta_dict, config, calib_params, pulse_type,
                     exp_dict, conv_params, beta_label, save_path=None):
    """
    Runs the Rabi chevron bootstrap followed by the iterative
    AAE + Frequency_Calibration_XY convergence loop.

    Modifies meta_dict in-place and returns the final config.
    save_path overrides the path= argument for all experiment saves (default: qubit).
    """
    if save_path is None:
        save_path = qubit

    # ── Phase 1: Rabi chevron bootstrap ──────────────────────────────────────
    chev = exp_dict["Chevron"]
    amp_vec = np.linspace(chev["amp_min"], chev["amp_max"], chev["amp_points"])
    f_vec   = np.linspace(
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
    opt_amp         = chevron.data["opt_amp"]

    print(f"  Chevron first π-pulse at detuning={opt_detuning_Hz / u.MHz:.3f} MHz, "
          f"amp={opt_amp:.4f} V")

    meta_dict["IF_freq"]    += int(opt_detuning_Hz)
    meta_dict["X90"]["amp"] = opt_amp
    meta_dict["Y90"]["amp"] = opt_amp
    meta_dict["ef_amp"]     = opt_amp
    config = meta_dict_to_config_1Q(meta_dict)

    # ── Phase 2: Staged AAE + Frequency calibration ──────────────────────────
    max_iter = conv_params["max_iter"]

    N_max_stages     = exp_dict["Amplified_Amplitude_Error"]["N_max_stages"]
    freq_span_stages = exp_dict["Frequency_Calibration_XY"]["freq_span_stages"]

    for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
        print(f"  ── Stage {stage_idx} | AAE N_max={N_max}  freq_span={freq_span / u.MHz:.1f} MHz ──")
        prev = dict(amplitude=0.0, detuning=0.0)
        calibrated = dict(amplitude=False, detuning=False)

        for i in range(max_iter):

            # AAE
            N_vec = np.arange(
                exp_dict["Amplified_Amplitude_Error"]["N_min"],
                N_max,
                exp_dict["Amplified_Amplitude_Error"]["N_step"],
            )
            aae = AAE(
                qubit, N_vec,
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
            meta_dict["ef_amp"]      = meta_dict["X90"]["amp"]
            config = meta_dict_to_config_1Q(meta_dict)

            # Frequency_Calibration_XY
            freq_vec = np.linspace(
                -freq_span / 2,
                 freq_span / 2,
                exp_dict["Frequency_Calibration_XY"]["freq_points"],
                dtype=int,
            )
            freq_cal = Frequency_Calibration_XY(
                qubit, freq_vec,
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

            calibrated["amplitude"] = (
                abs(amp_err) < conv_params["amplitude_accuracy"]
            )
            calibrated["detuning"] = (
                abs(det_correction_MHz) < conv_params["detuning_accuracy"]
            )

            prev["amplitude"] = amp_err
            prev["detuning"]  = det_correction_MHz

            config = meta_dict_to_config_1Q(meta_dict)

            print(
                f"    Iter {i:02d} | amp_err={amp_err:+.3e}  "
                f"det_correction={det_correction_MHz:+.4f} MHz  | "
                f"calibrated={calibrated}"
            )

            if all(calibrated.values()):
                print(f"  Stage {stage_idx} converged after {i + 1} iterations")
                break

    return config


# ─── Create parent Autocal folder ────────────────────────────────────────────

_base_meta = make_meta_dict(QUBIT)[QUBIT]
pulse_length_ns  = _base_meta["X90"]["length"]
outer_folder_base = _base_meta["global_params"]["outer_folder"]

# Mirror the ExperimentClass convention: outer_folder/q1/q1_YYYY_MM_DD/
_date_str    = datetime.now().strftime('%Y_%m_%d')
_q1_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{_date_str}")
os.makedirs(_q1_date_dir, exist_ok=True)

parent_folder = os.path.join(
    _q1_date_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pulse_length_ns}ns_Autocal",
)
os.makedirs(parent_folder, exist_ok=True)
print(f"Autocal folder: {parent_folder}")

# ─── Main sweep ───────────────────────────────────────────────────────────────

results = []   # list of (beta, L1, L1_err, L2, L2_err, lambda1)

sweep_start = time.time()
print(f"Sweep started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

for beta in DRAG_beta_vec:
    beta_label = f"beta{beta:+.3f}".replace(".", "p")
    beta_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"DRAG beta = {beta:.3f}  |  started {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")

    # Fresh meta_dict for each beta; all saves route to parent_folder/q1/
    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    meta_dict["DRAG_beta"] = beta
    meta_dict["global_params"]["outer_folder"] = parent_folder

    config = meta_dict_to_config_1Q(meta_dict)
    # set_LO_parameters([meta_dict["q_LO"], meta_dict["r_LO"]])

    # 2-level SS calibration
    SS = SingleShot1Q(
        QUBIT,
        experiment_dict["SS_Cal"]["shots"],
        meta_dict=meta_dict,
        outer_folder=parent_folder,
        path=beta_label,
        suffix="SS_Cal",
        repeats=experiment_dict["SS_Cal"]["number_pi_pulses"],
        config=config,
        confidence_threshold=experiment_dict["SS_Cal"]["ground_threshold"],
        flat_path=True,
    )
    CALIB_PARAMS = SS.calib_params
    meta_dict["read_theta"] = SS.calib_params["read_theta"]
    config = meta_dict_to_config_1Q(meta_dict)
    print(f"  SS Cal complete — fidelity {SS.max_F:.3f}")

    # Chevron bootstrap + iterative calibration
    config = _run_calibration(
        QUBIT, meta_dict, config, CALIB_PARAMS,
        PULSE_TYPE, experiment_dict,
        optimizer_convergence_params, beta_label,
        save_path=beta_label,
    )

    print(f"  Final X90 amp={meta_dict['X90']['amp']:.6f} V  "
          f"IF_freq={meta_dict['IF_freq'] / u.MHz:.6f} MHz")

    # 3-level SS calibration — retry until fidelity > 75%
    while True:
        SS3 = SingleShot1Q3Level(
            QUBIT,
            experiment_dict["SS_Cal_3L"]["shots"],
            meta_dict=meta_dict,
            outer_folder=parent_folder,
            path=beta_label,
            suffix="SS_Cal_3L",
            config=config,
            flat_path=True,
        )
        print(f"  3-level SS Cal — fidelity {SS3.max_F:.3f}")
        if SS3.max_F > 0.75:
            break
        print("  Fidelity below 0.75, retrying...")
    calib_3L = SS3.calib_params

    # Leakage RB
    rb = RB1Q_Leakage(
        QUBIT,
        experiment_dict["Leakage_RB"]["circuit_lengths"],
        experiment_dict["Leakage_RB"]["num_circuits"],
        experiment_dict["Leakage_RB"]["shots"],
        calib_3L,
        meta_dict=meta_dict,
        outer_folder=parent_folder,
        path=beta_label,
        suffix="LRB",
        config=config,
        flat_path=True,
    )

    L1,  L1_err  = rb.data["fit_data"]["state_2_L1"]
    L2,  L2_err  = rb.data["fit_data"]["state_2_L2"]
    lambda1      = 1.0 - L1 - L2
    results.append((beta, L1, L1_err, L2, L2_err, lambda1))
    beta_elapsed  = time.time() - beta_start
    sweep_elapsed = time.time() - sweep_start
    print(f"  Leakage RB done — L1={L1:.4e} ± {L1_err:.4e}  L2={L2:.4e} ± {L2_err:.4e}  λ1={lambda1:.4f}")
    print(f"  Beta elapsed: {beta_elapsed:.0f}s  |  Total elapsed: {sweep_elapsed:.0f}s")


# ─── Summary ─────────────────────────────────────────────────────────────────

total_elapsed = time.time() - sweep_start
print(f"\n{'=' * 60}")
print(f"All betas complete in {total_elapsed:.0f}s.")
print(f"{'=' * 60}")

betas    = np.array([r[0] for r in results])
l1s      = np.array([r[1] for r in results])
l1errs   = np.array([r[2] for r in results])
l2s      = np.array([r[3] for r in results])
l2errs   = np.array([r[4] for r in results])
lambda1s = np.array([r[5] for r in results])

# Save summary HDF5
summary_path = os.path.join(parent_folder, f"{QUBIT}_autocal_summary.h5")
with h5py.File(summary_path, "w") as f:
    f.create_dataset("betas",   data=betas)
    f.create_dataset("L1",      data=l1s)
    f.create_dataset("L1_err",  data=l1errs)
    f.create_dataset("L2",      data=l2s)
    f.create_dataset("L2_err",  data=l2errs)
    f.create_dataset("lambda1", data=lambda1s)
    f.attrs["qubit"]           = QUBIT
    f.attrs["pulse_length_ns"] = int(pulse_length_ns)
print(f"Summary HDF5 saved to {summary_path}")

# Three-panel log-scale figure
one_minus_lambda1s = 1.0 - lambda1s
fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
for ax, ys, yerrs, ylabel, ylim_bottom in zip(
    axes,
    [l1s,    l2s,    one_minus_lambda1s],
    [l1errs, l2errs, np.zeros(len(one_minus_lambda1s))],
    ["L1  (leakage per Clifford)", "L2  (leakage per Clifford)", "1 - λ1"],
    [1e-6, None, 1e-5],
):
    ax.errorbar(betas, ys, yerr=yerrs, fmt="o-", capsize=4)
    ax.set_yscale("log")
    ax.set_ylim(bottom=ylim_bottom)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
axes[-1].set_xlabel("DRAG beta")
axes[0].set_title(f"{QUBIT} — leakage parameters vs DRAG beta  ({pulse_length_ns} ns pulse)")
plt.tight_layout()

fig_path = os.path.join(parent_folder, f"{QUBIT}_L1_L2_lambda1_vs_DRAG_beta.png")
fig.savefig(fig_path, bbox_inches="tight")
print(f"Figure saved to {fig_path}")
plt.ioff()
plt.show()
