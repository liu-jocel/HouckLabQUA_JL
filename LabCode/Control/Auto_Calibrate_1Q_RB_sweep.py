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
from LabCode.Experiments.Rabi.m_Rabi_Chevron_IQ import RabiChevronIQ
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.RB.m_RB_1Q import RB1Q

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


PULSE_TYPE = "X90"
QUBIT = "q6"
PULSE_LENGTH_VEC = [20]  # ns
DRAG_beta_vec = np.linspace(0.6, 1.2, 11)

optimizer_convergence_params = {
    "amplitude_accuracy": 0.0001,
    "detuning_accuracy": 0.08,  # MHz
    "max_iter": 3,
}

experiment_dict = {
    "SS_Cal": {
        "shots": 1000,
        "number_pi_pulses": 1,
        "ground_threshold": 0.7,
        "object": SingleShot1Q,
    },
    "Chevron_IQ": {
        "shots": 21,
        "amp_min": 0.0,
        "amp_max": 0.45,
        "amp_points": 21,
        "freq_span": 20 * u.MHz,
        "freq_points": 21,
        "object": RabiChevronIQ,
    },
    "Chevron": {
        "shots": 50,
        "amp_min": 0.0,
        "amp_max": 0.45,
        "amp_points": 31,
        "freq_span": 20 * u.MHz,
        "freq_points": 31,
        "object": RabiChevronSS,
    },
    "Amplified_Amplitude_Error": {
        "shots": 300,
        "N_min": 1,
        "N_step": 1,
        "N_max_stages": [11, 51],
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
    "RB": {
        "shots": 100,
        "num_circuits": 250,
        "circuit_lengths": [2, 4, 8, 16, 32, 64, 128, 256, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000],
        "object": RB1Q,
    },
}


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
):
    """
    Rabi chevron bootstrap + iterative AAE + Frequency_Calibration_XY convergence.
    Modifies meta_dict in-place and returns the final config.
    """
    if save_path is None:
        save_path = qubit

    # Phase 1: Rabi chevron bootstrap
    chev = exp_dict["Chevron"]
    amp_vec = np.linspace(chev["amp_min"], chev["amp_max"], chev["amp_points"])
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
        f"    Chevron first Ï€-pulse at detuning={opt_detuning_Hz / u.MHz:.3f} MHz, amp={opt_amp:.4f} V"
    )

    meta_dict["IF_freq"] += int(opt_detuning_Hz)
    meta_dict["X90"]["amp"] = opt_amp
    meta_dict["Y90"]["amp"] = opt_amp
    meta_dict["ef_amp"] = opt_amp
    config = meta_dict_to_config_1Q(meta_dict)

    # Phase 2: Staged AAE + Frequency calibration
    max_iter = conv_params["max_iter"]
    N_max_stages = exp_dict["Amplified_Amplitude_Error"]["N_max_stages"]
    freq_span_stages = exp_dict["Frequency_Calibration_XY"]["freq_span_stages"]

    for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
        print(
            f"Stage {stage_idx} | AAE N_max={N_max}  freq_span={freq_span / u.MHz:.1f} MHz"
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
            meta_dict["ef_amp"] = meta_dict["X90"]["amp"]
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


# Create parent folder 

_base_meta = make_meta_dict(QUBIT)[QUBIT]
outer_folder_base = _base_meta["global_params"]["outer_folder"]
_date_str = datetime.now().strftime("%Y_%m_%d")
_q1_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{_date_str}")
os.makedirs(_q1_date_dir, exist_ok=True)

parent_folder = os.path.join(
    _q1_date_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pulse_length_RB_sweep",
)
os.makedirs(parent_folder, exist_ok=True)
print(f"Pulse length RB sweep folder: {parent_folder}")

# Main sweep 

# (pulse_len_ns, best_beta, best_fidelity, best_fidelity_err, best_amp, best_IF_freq_Hz)
sweep_results = []

sweep_start = time.time()
print(f"Sweep started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    # IQ chevron bootstrap (once per pulse length, no SS required) 
    _chev_iq = experiment_dict["Chevron_IQ"]
    _amp_vec_iq = np.linspace(
        _chev_iq["amp_min"], _chev_iq["amp_max"], _chev_iq["amp_points"]
    )
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
        f"  IQ bootstrap first Ï€-pulse: detuning offset={pl_IF_freq_offset / u.MHz:.3f} MHz, "
        f"amp={pl_amp_init:.4f} V"
    )

    # (beta, gate_fidelity, gate_fidelity_err, amp, IF_freq_Hz)
    drag_results = []

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
        meta_dict["ef_amp"] = pl_amp_init
        meta_dict["IF_freq"] += pl_IF_freq_offset
        meta_dict["global_params"]["outer_folder"] = pl_folder

        config = meta_dict_to_config_1Q(meta_dict)
        # set_LO_parameters([meta_dict["q_LO"], meta_dict["r_LO"]])

        # SS calibration
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
        CALIB_PARAMS = SS.calib_params
        meta_dict["read_theta"] = SS.calib_params["read_theta"]
        config = meta_dict_to_config_1Q(meta_dict)
        print(f"    SS Cal complete  €” fidelity {SS.max_F:.3f}")

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
        )

        print(
            f"    Final X90 amp={meta_dict['X90']['amp']:.6f} V  "
            f"IF_freq={meta_dict['IF_freq'] / u.MHz:.6f} MHz"
        )

        # Standard RB
        rb = RB1Q(
            QUBIT,
            experiment_dict["RB"]["circuit_lengths"],
            experiment_dict["RB"]["num_circuits"],
            experiment_dict["RB"]["shots"],
            CALIB_PARAMS,
            meta_dict=meta_dict,
            outer_folder=pl_folder,
            path=beta_label,
            suffix="RB",
            config=config,
        )

        gate_fidelity = rb.data.get("gate_fidelity", float("nan"))
        gate_fidelity_err = rb.data.get("gate_fidelity_error", float("nan"))
        cal_amp = meta_dict["X90"]["amp"]
        cal_IF_freq = meta_dict["IF_freq"]

        drag_results.append(
            (beta, gate_fidelity, gate_fidelity_err, cal_amp, cal_IF_freq)
        )
        beta_elapsed = time.time() - beta_start
        print(
            f"    RB done  €” F={gate_fidelity:.6f}  ± {gate_fidelity_err:.6f}  ({beta_elapsed:.0f}s)"
        )

    # Per-pulse-length DRAG summary 
    betas = np.array([r[0] for r in drag_results])
    fidelities = np.array([r[1] for r in drag_results])
    fidelity_errs = np.array([r[2] for r in drag_results])

    drag_summary_path = os.path.join(pl_folder, f"{QUBIT}_RB_summary.h5")
    with h5py.File(drag_summary_path, "w") as f:
        f.create_dataset("betas", data=betas)
        f.create_dataset("gate_fidelity", data=fidelities)
        f.create_dataset("gate_fidelity_err", data=fidelity_errs)
        f.attrs["qubit"] = QUBIT
        f.attrs["pulse_length_ns"] = int(pulse_length_ns)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.errorbar(betas, fidelities, yerr=fidelity_errs, fmt="o-", capsize=4)
    ax.set_xlabel("DRAG beta")
    ax.set_ylabel("Gate fidelity")
    ax.set_title(f"{QUBIT}  €” gate fidelity vs DRAG beta  ({pulse_length_ns} ns pulse)")
    ax.grid(True, alpha=0.3)
    drag_fig_path = os.path.join(pl_folder, f"{QUBIT}_gate_fidelity_vs_DRAG_beta.png")
    fig.savefig(drag_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  DRAG sweep figure saved to {drag_fig_path}")

    # Find best beta (maximum gate fidelity) 
    best_idx = int(np.nanargmax(fidelities))
    best = drag_results[best_idx]
    sweep_results.append(
        (
            pulse_length_ns,
            best[0],  # beta
            best[1],  # gate_fidelity
            best[2],  # gate_fidelity_err
            best[3],  # amp
            best[4],  # IF_freq_Hz
        )
    )

    pl_elapsed = time.time() - pl_start
    print(f"\n  Pulse length {pulse_length_ns} ns complete in {pl_elapsed:.0f}s")
    print(
        f"  Best: beta={best[0]:.3f}  F={best[1]:.6f}  ± {best[2]:.6f}  "
        f"amp={best[3]:.6f} V  IF_freq={best[4] / u.MHz:.6f} MHz"
    )


# Final summary 

total_elapsed = time.time() - sweep_start
print(f"\n{'=' * 60}")
print(f"All pulse lengths complete in {total_elapsed:.0f}s.")
print(f"{'=' * 60}")

pls = np.array([r[0] for r in sweep_results])
best_betas = np.array([r[1] for r in sweep_results])
best_fidelities = np.array([r[2] for r in sweep_results])
best_fidel_errs = np.array([r[3] for r in sweep_results])
best_amps = np.array([r[4] for r in sweep_results])
best_IF_freqs = np.array([r[5] for r in sweep_results])

# Save summary HDF5
summary_path = os.path.join(parent_folder, f"{QUBIT}_best_vs_gate_time.h5")
with h5py.File(summary_path, "w") as f:
    f.create_dataset("pulse_lengths_ns", data=pls)
    f.create_dataset("best_beta", data=best_betas)
    f.create_dataset("gate_fidelity", data=best_fidelities)
    f.create_dataset("gate_fidelity_err", data=best_fidel_errs)
    f.create_dataset("X90_amp", data=best_amps)
    f.create_dataset("IF_freq_Hz", data=best_IF_freqs)
    f.attrs["qubit"] = QUBIT
print(f"Summary HDF5 saved to {summary_path}")

fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
ax.errorbar(pls, best_fidelities, yerr=best_fidel_errs, fmt="o-", capsize=4)
ax.set_xlabel("Gate time (ns)")
ax.set_ylabel("Gate fidelity (best DRAG beta)")
ax.set_title(f"{QUBIT}  €” best gate fidelity vs gate time")
ax.grid(True, alpha=0.3)
fig_path = os.path.join(parent_folder, f"{QUBIT}_gate_fidelity_vs_gate_time.png")
fig.savefig(fig_path, bbox_inches="tight")
print(f"Figure saved to {fig_path}")
plt.ioff()
plt.show()

