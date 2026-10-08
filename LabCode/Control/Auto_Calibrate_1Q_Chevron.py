import os
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import gaussian_filter

from LabCode.Config.make_meta_dict import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.PythonDrivers.control_8257 import set_LO_parameters
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.Rabi.m_Rabi_Chevron_SS import RabiChevronSS
from LabCode.Experiments.Gates.m_amplified_amplitude_error import AAE
from LabCode.Experiments.Gates.m_frequency_cal_XY import Frequency_Calibration_XY
from LabCode.Experiments.Gates.m_all_XY import AllXY

from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


PULSE_TYPE = "X90"
QUBIT = "q1"
CHEVRON_SMOOTH_SIGMA = 2  # Gaussian smoothing kernel size (pixels) for chevron peak-finding

optimizer_convergence_params = {
    "amplitude_accuracy": 0.00005,
    "detuning_accuracy": 0.03,  # MHz — matches the unit returned by optimal_detuning_MHz
    "max_iter": 4,
}

experiment_dict = {
    "SS_Cal": {
        "shots": 300,
        "number_pi_pulses": 1,
        "ground_threshold": 0.7,
        "object": SingleShot1Q,
    },
    "Chevron": {
        "shots": 50,
        "amp_min": 0.0,
        "amp_max": 0.45,
        "amp_points": 31,
        "freq_span": 11 * u.MHz,
        "freq_points": 31,
        "object": RabiChevronSS,
    },
    "Amplified_Amplitude_Error": {
        "shots": 300,
        "N_min": 1,
        "N_step": 1,
        "N_max_stages": [11, 51],  # N_max used at each calibration stage
        "object": AAE,
    },
    "Frequency_Calibration_XY": {
        "shots": 300,
        "pulse_sequence": [
            ("X90", "I"),
            ("Y90", "I"),
            ("X90", "Y90"),
            ("Y90", "X90"),
        ],
        "freq_span_stages": [3 * u.MHz, 1 * u.MHz],  # freq_span at each stage
        "freq_points": 51,
        "num_repeats": 1,
        "object": Frequency_Calibration_XY,
    },
}


# ─── Setup ────────────────────────────────────────────────────────────────────

calib_start = time.time()
print(f"Calibration started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

meta_dict = make_meta_dict(QUBIT)[QUBIT]
config = meta_dict_to_config_1Q(meta_dict)
LOs = [meta_dict["q_LO"], meta_dict["r_LO"]]
# set_LO_parameters(LOs)

# ─── Phase 0: 2-level single-shot calibration ─────────────────────────────────

SS = SingleShot1Q(
    QUBIT,
    experiment_dict["SS_Cal"]["shots"],
    meta_dict=meta_dict,
    outer_folder=meta_dict["global_params"]["outer_folder"],
    path=QUBIT,
    suffix="SS_Cal",
    repeats=experiment_dict["SS_Cal"]["number_pi_pulses"],
    config=config,
    confidence_threshold=experiment_dict["SS_Cal"]["ground_threshold"],
)
CALIB_PARAMS = SS.calib_params
meta_dict["read_theta"] = SS.calib_params["read_theta"]
config = meta_dict_to_config_1Q(meta_dict)
print(f"SS Cal complete — fidelity {SS.max_F:.3f}")

# ─── Phase 1: Rabi chevron bootstrap ─────────────────────────────────────────

chev = experiment_dict["Chevron"]
amp_vec = np.linspace(chev["amp_min"], chev["amp_max"], chev["amp_points"])
f_vec   = np.linspace(
    -chev["freq_span"] / 2,
     chev["freq_span"] / 2,
     chev["freq_points"],
    dtype=int,
)

chevron = RabiChevronSS(
    QUBIT,
    amp_vec,
    f_vec,
    chev["shots"],
    CALIB_PARAMS,
    num_pi_pulses=1,
    pulse_type=PULSE_TYPE,
    meta_dict=meta_dict,
    outer_folder=meta_dict["global_params"]["outer_folder"],
    path=QUBIT,
    suffix="Chevron_Bootstrap",
    config=config,
)

ss_array = chevron.data["ss_data"]                                  # (len(f_vec), len(amp_vec))
smoothed = gaussian_filter(ss_array, sigma=CHEVRON_SMOOTH_SIGMA)
max_idx  = np.unravel_index(np.argmax(smoothed), smoothed.shape)
opt_detuning_Hz  = f_vec[max_idx[0]]
opt_amp          = amp_vec[max_idx[1]]

print(f"Chevron max contrast at detuning={opt_detuning_Hz / u.MHz:.3f} MHz, "
      f"amp={opt_amp:.4f} V")

meta_dict["IF_freq"]    += int(opt_detuning_Hz)
meta_dict["X90"]["amp"] = opt_amp
meta_dict["Y90"]["amp"] = opt_amp
config = meta_dict_to_config_1Q(meta_dict)

# ─── Phase 2: Staged AAE + Frequency calibration ─────────────────────────────

max_iter = optimizer_convergence_params["max_iter"]

N_max_stages     = experiment_dict["Amplified_Amplitude_Error"]["N_max_stages"]
freq_span_stages = experiment_dict["Frequency_Calibration_XY"]["freq_span_stages"]

for stage_idx, (N_max, freq_span) in enumerate(zip(N_max_stages, freq_span_stages)):
    stage_start = time.time()
    print(f"\n── Stage {stage_idx} | AAE N_max={N_max}  freq_span={freq_span / u.MHz:.1f} MHz  |  {datetime.now().strftime('%H:%M:%S')} ──")
    prev = dict(amplitude=0.0, detuning=0.0)
    calibrated = dict(amplitude=False, detuning=False)

    for i in range(max_iter):

        # ── AAE: amplitude correction ─────────────────────────────────────────
        N_vec = np.arange(
            experiment_dict["Amplified_Amplitude_Error"]["N_min"],
            N_max,
            experiment_dict["Amplified_Amplitude_Error"]["N_step"],
        )
        aae = AAE(
            QUBIT,
            N_vec,
            experiment_dict["Amplified_Amplitude_Error"]["shots"],
            calib_params=CALIB_PARAMS,
            pulse_type=PULSE_TYPE,
            meta_dict=meta_dict,
            outer_folder=meta_dict["global_params"]["outer_folder"],
            path=QUBIT,
            suffix=f"AAE_s{stage_idx}_iter{i:02d}",
            config=config,
        )
        amp_err = aae.data["amplitude_error"]
        meta_dict["X90"]["amp"] += amp_err
        meta_dict["Y90"]["amp"] += amp_err
        config = meta_dict_to_config_1Q(meta_dict)

        # ── Frequency_Calibration_XY ──────────────────────────────────────────
        freq_vec = np.linspace(
            -freq_span / 2,
             freq_span / 2,
            experiment_dict["Frequency_Calibration_XY"]["freq_points"],
            dtype=int,
        )
        freq_cal = Frequency_Calibration_XY(
            QUBIT,
            freq_vec,
            experiment_dict["Frequency_Calibration_XY"]["pulse_sequence"],
            experiment_dict["Frequency_Calibration_XY"]["shots"],
            CALIB_PARAMS,
            num_repeats=experiment_dict["Frequency_Calibration_XY"]["num_repeats"],
            meta_dict=meta_dict,
            outer_folder=meta_dict["global_params"]["outer_folder"],
            path=QUBIT,
            suffix=f"FreqCal_s{stage_idx}_iter{i:02d}",
            config=config,
        )
        det_correction_MHz = freq_cal.data["optimal_detuning_MHz"]
        meta_dict["IF_freq"] += int(det_correction_MHz * u.MHz)

        # ── Convergence check ─────────────────────────────────────────────────
        calibrated["amplitude"] = (
            abs(amp_err) < optimizer_convergence_params["amplitude_accuracy"]
        )
        calibrated["detuning"] = (
            abs(det_correction_MHz) < optimizer_convergence_params["detuning_accuracy"]
        )

        prev["amplitude"] = amp_err
        prev["detuning"]  = det_correction_MHz

        print(
            f"  Iter {i:02d} | "
            f"amp_err={amp_err:+.3e}  "
            f"det_correction={det_correction_MHz:+.4f} MHz  | "
            f"calibrated={calibrated}"
        )

        config = meta_dict_to_config_1Q(meta_dict)

        if all(calibrated.values()):
            print(f"  Stage {stage_idx} converged after {i + 1} iterations  ({time.time() - stage_start:.0f}s)")
            break
    else:
        print(f"  Stage {stage_idx} reached max_iter  ({time.time() - stage_start:.0f}s)")

total_elapsed = time.time() - calib_start
print("\nCalibration complete")
print("-" * 40)
print(f"X90 amplitude : {meta_dict['X90']['amp']:.6f} V")
print(f"Y90 amplitude : {meta_dict['Y90']['amp']:.6f} V")
print(f"IF_freq       : {meta_dict['IF_freq'] / u.MHz:.6f} MHz")
print(f"Total elapsed : {total_elapsed:.0f}s")
print("-" * 40)

# ─── Phase 3: Diagnostic AllXY ───────────────────────────────────────────────

AllXY(
    QUBIT,
    1000,
    CALIB_PARAMS,
    num_pulses=1,
    meta_dict=meta_dict,
    outer_folder=meta_dict["global_params"]["outer_folder"],
    path=QUBIT,
    suffix="AllXY_final",
    config=config,
)
