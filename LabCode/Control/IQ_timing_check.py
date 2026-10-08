"""
DRAG derivative-timing sweep via baseband zero-padding.

Goal: test whether a relative timing offset between the main (cosine) envelope
and the DRAG derivative term changes total RB error and leakage-RB L1.

Geometry (per spec):
  - Gate length 32 ns, DRAG beta = 1 (frozen).
  - Main envelope padded by 10 zeros on each side -> 52 ns total, starts at t=10.
  - Derivative term padded with 20 zeros total -> also 52 ns; its start slides
    from t=0 to t=20 in 1 ns steps.
  - delay = Q_start - I_start = derivative_start - 10, i.e. sweep is -10 .. +10 ns.
    (52 ns is a multiple of 4, satisfying the OPX arbitrary-waveform constraint;
    the played pulse length is identical at every sweep point.)

The DERIVATIVE term is the thing that slides on every gate, regardless of which
physical channel it lives on: for X-type gates the derivative is on the Q channel,
for Y-type gates it is on the I channel (the config swaps them). This uniformly
emulates a DRAG-timing miscalibration across the Clifford set. Identification is
by the '_der' substring in the waveform name.

Only the four 90-degree gates (X90/mX90/Y90/mY90) are padded — these are the only
operations played by the RB/LRB Clifford decompositions (verified against
m_RB_leakage_1Q.play_sequence). X180/X180_ef are untouched; SS cals run on the
unpadded config.

Calibration (SS cal + chevron + AAE + Frequency_Calibration_XY) runs ONCE,
unpadded, at the start; amp/IF are then frozen and every sweep point is the same
calibrated config with only the derivative offset changed. The constant 20 ns of
per-gate dead time shifts absolute error rates slightly (uniformly) but not the
shape of error-vs-delay.
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
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q_3level import SingleShot1Q3Level
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.RB.m_RB_leakage_1Q import RB1Q_Leakage
from LabCode.Experiments.RB.m_RB_1Q import RB1Q
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

# ─── Globals ──────────────────────────────────────────────────────────────────
PULSE_TYPE = "X90"
QUBIT = "q6"
NUM_AVERAGES = 3          # repeats of RB and LRB at every delay point

GATE_LENGTH_NS = 16       # bare DRAG gate length (ns)
DRAG_BETA_VEC = [0., 0.5, 0.9, 1.0, 1.2,0., 0.5, 0.9, 1.0, 1.2,0., 0.5, 0.9, 1.0, 1.2]  # array of betas to sweep; each gets its own folder
PAD_EACH_SIDE_NS = 6    # main envelope: 10 zeros each side -> starts at t=10
TOTAL_LENGTH_NS = GATE_LENGTH_NS + 2 * PAD_EACH_SIDE_NS  # 52 ns (multiple of 4)
I_START = PAD_EACH_SIDE_NS                               # = 10
# Derivative start slides 0..20 -> delay = der_start - I_START in [-10, +10]
DELAY_VEC = np.arange(-PAD_EACH_SIDE_NS, PAD_EACH_SIDE_NS + 1, 1)

assert TOTAL_LENGTH_NS % 4 == 0, "Padded pulse length must be a multiple of 4 ns"

# Gate pulses played by the RB Clifford decomposition (the only ones padded)
RB_GATE_PULSES = ["X90_pulse_", "Xm90_pulse_", "Y90_pulse_", "Ym90_pulse_"]

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
        "amp_max": 0.45,
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
        "shots": 30,
        "num_circuits": 25,
        "circuit_lengths": [
            2, 16, 64, 128, 500, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
        ],
    },
    "RB1Q": {
        "simulate": False,
        "shots": 50,
        "num_circuits": 50,
        "circuit_lengths": [
            8, 32, 128, 256, 500, 700, 900, 1200, 1500, 2000,
            3000, 4000, 5000, 6000, 7000,
        ],
    },
}


# ─── Padding helper ─────────────────────────────────────────────────────────--
def apply_derivative_delay(config, delay_ns):
    """
    Deep-copy `config`; for each 90-degree gate pulse, zero-pad both channels to
    TOTAL_LENGTH_NS. The MAIN envelope always starts at I_START (=10); the
    DERIVATIVE waveform (name contains '_der') starts at I_START + delay_ns.
    delay_ns in [-PAD_EACH_SIDE_NS, +PAD_EACH_SIDE_NS]; positive = derivative late.
    """
    if not (-PAD_EACH_SIDE_NS <= delay_ns <= PAD_EACH_SIDE_NS):
        raise ValueError(f"delay_ns={delay_ns} outside ±{PAD_EACH_SIDE_NS} ns padding range")

    cfg = copy.deepcopy(config)
    wfs = cfg["waveforms"]
    target_pulses = [p + QUBIT for p in RB_GATE_PULSES]

    for pname in target_pulses:
        pdef = cfg["pulses"][pname]
        i_name = pdef["waveforms"]["I"]
        q_name = pdef["waveforms"]["Q"]

        # Identify derivative vs main by waveform name; exactly one must be '_der'.
        i_is_der = "_der" in i_name
        q_is_der = "_der" in q_name
        if i_is_der == q_is_der:
            raise ValueError(
                f"Pulse {pname}: cannot identify derivative channel "
                f"(I='{i_name}', Q='{q_name}')"
            )

        for wf_name in (i_name, q_name):
            samples = np.asarray(wfs[wf_name]["samples"], dtype=float)
            if len(samples) != GATE_LENGTH_NS:
                raise ValueError(
                    f"{wf_name}: expected {GATE_LENGTH_NS} samples, got {len(samples)}"
                )
            start = I_START + (delay_ns if "_der" in wf_name else 0)
            padded = np.zeros(TOTAL_LENGTH_NS)
            padded[start:start + GATE_LENGTH_NS] = samples
            wfs[wf_name]["samples"] = padded.tolist()

        pdef["length"] = TOTAL_LENGTH_NS

    return cfg


# ─── Calibration routine (ported from m_pulse_length_sweep) ─────────────────--
def _run_calibration(qubit, meta_dict, config, calib_params, pulse_type,
                     exp_dict, conv_params, label, save_path):
    """Rabi chevron bootstrap + staged AAE + Frequency_Calibration_XY convergence.
    Modifies meta_dict in-place and returns the final (unpadded) config."""
    chev = exp_dict["Chevron"]
    amp_vec = np.linspace(chev["amp_min"], chev["amp_max"], chev["amp_points"])
    f_vec = np.linspace(-chev["freq_span"] / 2, chev["freq_span"] / 2,
                        chev["freq_points"], dtype=int)

    chevron = RabiChevronSS(
        qubit, amp_vec, f_vec, chev["shots"], calib_params,
        num_pi_pulses=1, pulse_type=pulse_type, meta_dict=meta_dict,
        outer_folder=meta_dict["global_params"]["outer_folder"],
        path=save_path, suffix=f"Chevron_{label}", config=config, flat_path=True,
    )
    opt_detuning_Hz = chevron.data["opt_detuning_Hz"]
    opt_amp = chevron.data["opt_amp"]
    print(f"    Chevron first π-pulse at detuning={opt_detuning_Hz / u.MHz:.3f} MHz, amp={opt_amp:.4f} V")

    meta_dict["IF_freq"] += int(opt_detuning_Hz)
    meta_dict["X90"]["amp"] = opt_amp
    meta_dict["Y90"]["amp"] = opt_amp
    config = meta_dict_to_config_1Q(meta_dict)

    max_iter = conv_params["max_iter"]
    N_max_stages = exp_dict["Amplified_Amplitude_Error"]["N_max_stages"]
    freq_span_stages = exp_dict["Frequency_Calibration_XY"]["freq_span_stages"]

    for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
        print(f"    ── Stage {stage_idx} | AAE N_max={N_max}  freq_span={freq_span / u.MHz:.1f} MHz ──")
        calibrated = dict(amplitude=False, detuning=False)

        for i in range(max_iter):
            N_vec = np.arange(
                exp_dict["Amplified_Amplitude_Error"]["N_min"], N_max,
                exp_dict["Amplified_Amplitude_Error"]["N_step"],
            )
            aae = AAE(
                qubit, N_vec, exp_dict["Amplified_Amplitude_Error"]["shots"],
                calib_params=calib_params, pulse_type=pulse_type, meta_dict=meta_dict,
                outer_folder=meta_dict["global_params"]["outer_folder"],
                path=save_path, suffix=f"AAE_{label}_s{stage_idx}_iter{i:02d}",
                config=config, flat_path=True,
            )
            amp_err = aae.data["amplitude_error"]
            meta_dict["X90"]["amp"] += amp_err
            meta_dict["Y90"]["amp"] += amp_err
            config = meta_dict_to_config_1Q(meta_dict)

            freq_vec = np.linspace(-freq_span / 2, freq_span / 2,
                                   exp_dict["Frequency_Calibration_XY"]["freq_points"], dtype=int)
            freq_cal = Frequency_Calibration_XY(
                qubit, freq_vec,
                exp_dict["Frequency_Calibration_XY"]["pulse_sequence"],
                exp_dict["Frequency_Calibration_XY"]["shots"], calib_params,
                num_repeats=exp_dict["Frequency_Calibration_XY"]["num_repeats"],
                meta_dict=meta_dict,
                outer_folder=meta_dict["global_params"]["outer_folder"],
                path=save_path, suffix=f"FreqCal_{label}_s{stage_idx}_iter{i:02d}",
                config=config, flat_path=True,
            )
            det_correction_MHz = freq_cal.data["optimal_detuning_MHz"]
            meta_dict["IF_freq"] += int(det_correction_MHz * u.MHz)

            calibrated["amplitude"] = abs(amp_err) < conv_params["amplitude_accuracy"]
            calibrated["detuning"] = abs(det_correction_MHz) < conv_params["detuning_accuracy"]
            config = meta_dict_to_config_1Q(meta_dict)

            print(f"      Iter {i:02d} | amp_err={amp_err:+.3e}  "
                  f"det_correction={det_correction_MHz:+.4f} MHz  | calibrated={calibrated}")
            if all(calibrated.values()):
                print(f"    Stage {stage_idx} converged after {i + 1} iterations")
                break

    return config


# ─── Create parent folder ─────────────────────────────────────────────────────
_base_meta = make_meta_dict(QUBIT)[QUBIT]
outer_folder_base = _base_meta["global_params"]["outer_folder"]
_date_str = datetime.now().strftime("%Y_%m_%d")
_q_date_dir = os.path.join(outer_folder_base, QUBIT, f"{QUBIT}_{_date_str}")
os.makedirs(_q_date_dir, exist_ok=True)
parent_folder = os.path.join(
    _q_date_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_IQ_derivative_delay_sweep"
)
os.makedirs(parent_folder, exist_ok=True)
print(f"I/Q derivative-delay sweep folder: {parent_folder}")

sweep_start = time.time()
print(f"Sweep started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─── Loop over DRAG beta values ────────────────────────────────────────────────
for beta_idx, DRAG_BETA in enumerate(DRAG_BETA_VEC):
    beta_label = f"beta{DRAG_BETA:+.2f}_{beta_idx:02d}".replace(".", "p")
    beta_folder = os.path.join(parent_folder, beta_label)
    os.makedirs(beta_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"DRAG beta = {DRAG_BETA:.2f}  [{beta_idx + 1}/{len(DRAG_BETA_VEC)}]")
    print(f"Beta folder: {beta_folder}")
    print(f"{'='*60}")

    # ─── One-time calibration (UNPADDED), then freeze ─────────────────────────────
    meta_dict = make_meta_dict(QUBIT)[QUBIT]
    meta_dict["DRAG_beta"] = DRAG_BETA
    meta_dict["X90"]["length"] = GATE_LENGTH_NS
    meta_dict["Y90"]["length"] = GATE_LENGTH_NS
    meta_dict["global_params"]["outer_folder"] = beta_folder
    config = meta_dict_to_config_1Q(meta_dict)

    # 2-level SS calibration — retry up to 10 times until fidelity > 0.70
    max_retries_2L = 10
    two_level_success = False
    for retry_idx in range(max_retries_2L):
        SS = SingleShot1Q(
            QUBIT, experiment_dict["SS_Cal"]["shots"],
            meta_dict=meta_dict, outer_folder=beta_folder,
            path="cal", suffix="SS_Cal",
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
        raise RuntimeError(f"2-level SS calibration failed after {max_retries_2L} attempts")

    CALIB_PARAMS = SS.calib_params
    meta_dict["read_theta"] = SS.calib_params["read_theta"]
    config = meta_dict_to_config_1Q(meta_dict)

    # Gate calibration (chevron + AAE + FreqCal), all unpadded
    config = _run_calibration(
        QUBIT, meta_dict, config, CALIB_PARAMS, PULSE_TYPE,
        experiment_dict, optimizer_convergence_params,
        label="cal", save_path="cal",
    )
    print(f"  Final X90 amp={meta_dict['X90']['amp']:.6f} V  "
          f"IF_freq={meta_dict['IF_freq'] / u.MHz:.6f} MHz")

    # 3-level SS calibration — retry up to 30 times (unpadded config)
    max_retries_3L = 30
    three_level_success = False
    for retry_idx in range(max_retries_3L):
        SS3 = SingleShot1Q3Level(
            QUBIT, experiment_dict["SS_Cal_3L"]["shots"],
            meta_dict=meta_dict, outer_folder=beta_folder,
            path="cal", suffix="SS_Cal_3L", config=config, flat_path=True,
        )
        print(f"  3-level SS Cal — fidelity {SS3.max_F:.3f}")
        if SS3.max_F > 0.65:
            three_level_success = True
            break
        if retry_idx < max_retries_3L - 1:
            print("  Fidelity below 0.65, retrying...")
    if not three_level_success:
        raise RuntimeError(f"3-level SS calibration failed after {max_retries_3L} attempts")
    calib_3L = SS3.calib_params

    # Frozen, calibrated, UNPADDED base config — sweep points are built from this.
    base_config = meta_dict_to_config_1Q(meta_dict)

    # ─── Padding preview at extreme delays ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, d in zip(axes, [-PAD_EACH_SIDE_NS, 0, PAD_EACH_SIDE_NS]):
        cfg_prev = apply_derivative_delay(base_config, d)
        pwf = cfg_prev["pulses"][f"X90_pulse_{QUBIT}"]["waveforms"]
        ax.plot(cfg_prev["waveforms"][pwf["I"]]["samples"], label="I (main)", color="darkblue")
        ax.plot(cfg_prev["waveforms"][pwf["Q"]]["samples"], label="Q (derivative)", color="crimson")
        ax.set_title(f"X90, delay = {d:+d} ns")
        ax.set_xlabel("sample (ns)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{QUBIT} — padded X90: derivative offset relative to main envelope (β={DRAG_BETA:.2f})")
    plt.tight_layout()
    fig.savefig(os.path.join(beta_folder, f"{QUBIT}_padding_preview.png"), bbox_inches="tight")
    plt.close(fig)

    # ─── Delay sweep ──────────────────────────────────────────────────────────────
    # (delay_ns, L1_mean, L1_std, L1_fiterr, L2_mean, L2_std, rb_err_mean, rb_err_std)
    sweep_results = []

    for delay_ns in DELAY_VEC:
        delay_ns = int(delay_ns)
        d_label = f"delay{delay_ns:+d}ns".replace("+", "p").replace("-", "m")
        d_start = time.time()
        print(f"\n  delay = {delay_ns:+d} ns  |  {datetime.now().strftime('%H:%M:%S')}")

        cfg = apply_derivative_delay(base_config, delay_ns)

        # ── Leakage RB × NUM_AVERAGES ─────────────────────────────────────────────
        lrb_l1s, lrb_l1_errs, lrb_l2s = [], [], []
        for avg_idx in range(NUM_AVERAGES):
            rb = RB1Q_Leakage(
                QUBIT,
                experiment_dict["Leakage_RB"]["circuit_lengths"],
                experiment_dict["Leakage_RB"]["num_circuits"],
                experiment_dict["Leakage_RB"]["shots"],
                calib_3L,
                meta_dict=meta_dict,
                outer_folder=beta_folder,
                path=d_label,
                suffix=f"LRB_{d_label}_avg{avg_idx:02d}",
                config=cfg,
                flat_path=True,
            )
            try:
                L1, L1_err = rb.data["fit_data"]["state_2_L1_per_gate"]
                L2, _ = rb.data["fit_data"]["state_2_L2_per_gate"]
            except Exception as e:
                print(f"    LRB avg {avg_idx}: fit failed ({e}), recording NaN")
                L1, L1_err, L2 = np.nan, np.nan, np.nan
            lrb_l1s.append(L1)
            lrb_l1_errs.append(L1_err)
            lrb_l2s.append(L2)
            print(f"    LRB average {avg_idx + 1}/{NUM_AVERAGES}: L1={L1:.4e}, L2={L2:.4e}")

        L1_mean = np.nanmean(lrb_l1s)
        L1_std = np.nanstd(lrb_l1s)
        L1_fiterr = np.nanmean(lrb_l1_errs)
        L2_mean = np.nanmean(lrb_l2s)
        L2_std = np.nanstd(lrb_l2s)

        # ── Vanilla RB × NUM_AVERAGES ─────────────────────────────────────────────
        rb_fidelities = []
        for avg_idx in range(NUM_AVERAGES):
            rb1q = RB1Q(
                QUBIT,
                experiment_dict["RB1Q"]["circuit_lengths"],
                experiment_dict["RB1Q"]["num_circuits"],
                experiment_dict["RB1Q"]["shots"],
                CALIB_PARAMS,
                meta_dict=meta_dict,
                outer_folder=beta_folder,
                path=d_label,
                suffix=f"RB1Q_{d_label}_avg{avg_idx:02d}",
                config=cfg,
                simulate=experiment_dict["RB1Q"]["simulate"],
                flat_path=True,
            )
            _fid = rb1q.data.get("gate_fidelity", float("nan"))
            rb_fidelities.append(_fid)
            print(f"    RB average {avg_idx + 1}/{NUM_AVERAGES}: fidelity = {_fid:.6f}")

        rb_err_mean = 1.0 - np.nanmean(rb_fidelities)
        rb_err_std = np.nanstd(rb_fidelities)

        sweep_results.append(
            (delay_ns, L1_mean, L1_std, L1_fiterr, L2_mean, L2_std, rb_err_mean, rb_err_std)
        )
        print(f"  delay {delay_ns:+d} ns done in {time.time() - d_start:.0f}s — "
              f"L1={L1_mean:.4e} ± {L1_std:.4e}  RB_err={rb_err_mean:.4e} ± {rb_err_std:.4e}")

    # ─── Save + plot ──────────────────────────────────────────────────────────────
    delays = np.array([r[0] for r in sweep_results])
    L1_means = np.array([r[1] for r in sweep_results])
    L1_stds = np.array([r[2] for r in sweep_results])
    L1_fiterrs = np.array([r[3] for r in sweep_results])
    L2_means = np.array([r[4] for r in sweep_results])
    L2_stds = np.array([r[5] for r in sweep_results])
    rb_err_means = np.array([r[6] for r in sweep_results])
    rb_err_stds = np.array([r[7] for r in sweep_results])

    summary_path = os.path.join(beta_folder, f"{QUBIT}_error_vs_IQ_delay.h5")
    with h5py.File(summary_path, "w") as f:
        f.create_dataset("delay_ns", data=delays)
        f.create_dataset("L1_mean", data=L1_means)
        f.create_dataset("L1_std", data=L1_stds)
        f.create_dataset("L1_fiterr_mean", data=L1_fiterrs)
        f.create_dataset("L2_mean", data=L2_means)
        f.create_dataset("L2_std", data=L2_stds)
        f.create_dataset("RB_error_rate", data=rb_err_means)
        f.create_dataset("RB_error_rate_std", data=rb_err_stds)
        f.attrs["qubit"] = QUBIT
        f.attrs["num_averages"] = NUM_AVERAGES
        f.attrs["gate_length_ns"] = int(GATE_LENGTH_NS)
        f.attrs["padded_length_ns"] = int(TOTAL_LENGTH_NS)
        f.attrs["drag_beta"] = float(DRAG_BETA)
        f.attrs["X90_amp"] = float(meta_dict["X90"]["amp"])
        f.attrs["IF_freq_Hz"] = float(meta_dict["IF_freq"])
    print(f"\nSummary HDF5 saved to {summary_path}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(delays, rb_err_means, yerr=rb_err_stds, fmt="o-", capsize=4,
                label=f"Total error (RB, n={NUM_AVERAGES})")
    ax.errorbar(delays, L1_means, yerr=L1_stds, fmt="s-", capsize=4,
                label=f"Leakage error L1 (LRB, n={NUM_AVERAGES})")
    ax.axvline(0, color="grey", ls="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Derivative-term delay, Q_start − I_start (ns)   [+ = derivative late]")
    ax.set_ylabel("Error rate per Clifford")
    ax.set_title(f"{QUBIT} — RB error & leakage L1 vs I/Q (cosine/derivative) delay\n"
                 f"({GATE_LENGTH_NS} ns gate padded to {TOTAL_LENGTH_NS} ns, β={DRAG_BETA:.2f})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(beta_folder, f"{QUBIT}_error_vs_IQ_delay.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

print(f"\nTotal sweep time: {time.time() - sweep_start:.0f}s")
plt.ioff()
plt.show()