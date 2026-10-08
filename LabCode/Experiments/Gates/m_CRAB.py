import os
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cma
from scipy.optimize import curve_fit
from qm.qua import *
from qm import QuantumMachinesManager

# Add LabCode to path
import sys
from pathlib import Path

# Get repo root (4 levels up: Gates → Experiments → LabCode → repo_root)
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))


from LabCode.Config.make_meta_dict_TATQ04_taper import make_meta_dict
from LabCode.Config.config_helpers import meta_dict_to_config_1Q
from LabCode.Experiments.Readout_Cal.m_single_shot_1Q import SingleShot1Q
from LabCode.Experiments.RB.m_RB_1Q_CRAB import RB1Q_CRAB
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def make_crab_waveform(tg, a_k, omega_k, phi_k, n_samples=None, normalize=False):
    """
    Returns a CRAB correction waveform as a sample array.
    omega_k must be in rad/ns (= 2*pi*f_MHz / 1000).
    n_samples: OPX requires multiples of 4, minimum 16.
    normalize: if True, rescales peak to 0.5 V (use for standalone pulse).
               if False (default), a_k directly sets the amplitude in V (use as additive correction).
    """
    if n_samples is None:
        n_samples = int(round(tg / 4)) * 4
        n_samples = max(n_samples, 16)

    t = np.arange(n_samples)
    pulse = np.zeros(n_samples)
    for k in range(len(a_k)):
        pulse += a_k[k] * np.cos(omega_k[k] * t + phi_k[k])

    if normalize:
        peak = np.max(np.abs(pulse))
        if peak > 0:
            pulse = pulse / peak * 0.5

    return pulse.tolist()


def add_x90_oc_to_config(config, q_name, crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y):
    """
    Adds X90_OC, mX90_OC, Y90_OC, mY90_OC operations to q_name in the config.

    X90 and Y90 receive INDEPENDENT corrections, allowing for asymmetric distortions.
    This is more generic than shared corrections as it accounts for:
    - Different pulse envelopes (X vs Y have different shapes)
    - Asymmetric leakage/nonlinearities
    - Gate-specific distortion signatures

    X90_OC : X90_I + crab_I_X,  X90_Q + crab_Q_X
    Y90_OC : Y90_I + crab_I_Y,  Y90_Q + crab_Q_Y  (independent correction)
    mX90/mY90 are the negatives of the above.

    CRAB corrections are rescaled independently using headroom for each gate.
    """
    # X90: I←X90_wf_, Q←X90_wf_der_
    # Y90: I←Y90_wf_der_, Q←Y90_wf_   (note reversed _/der_ convention vs X90)
    x90_I = np.array(
        config["waveforms"]["X90_wf_" + q_name]["samples"]
    )  # phys-I of X90
    x90_Q = np.array(
        config["waveforms"]["X90_wf_der_" + q_name]["samples"]
    )  # phys-Q of X90
    y90_phyI = np.array(
        config["waveforms"]["Y90_wf_der_" + q_name]["samples"]
    )  # phys-I of Y90
    y90_phyQ = np.array(
        config["waveforms"]["Y90_wf_" + q_name]["samples"]
    )  # phys-Q of Y90

    def _fit(corr, headroom, channel):
        corr = np.array(corr)
        peak = np.max(np.abs(corr))
        if peak > headroom:
            scale = headroom / peak
            print(
                f"  WARNING: CRAB-{channel} peak={peak:.4f} V > headroom={headroom:.4f} V — rescaling by {scale:.4f}."
            )
            corr = corr * scale
        return corr

    # X90 headroom (independent)
    headroom_I_X = 0.5 - np.max(np.abs(x90_I))
    headroom_Q_X = 0.5 - np.max(np.abs(x90_Q))
    crab_I_X = _fit(crab_wf_I_X, headroom_I_X, "I_X")
    crab_Q_X = _fit(crab_wf_Q_X, headroom_Q_X, "Q_X")

    # Y90 headroom (independent)
    headroom_I_Y = 0.5 - np.max(np.abs(y90_phyI))
    headroom_Q_Y = 0.5 - np.max(np.abs(y90_phyQ))
    crab_I_Y = _fit(crab_wf_I_Y, headroom_I_Y, "I_Y")
    crab_Q_Y = _fit(crab_wf_Q_Y, headroom_Q_Y, "Q_Y")

    # X90 family: correction added to physical channels
    x90_oc_phyI = x90_I + crab_I_X
    x90_oc_phyQ = x90_Q + crab_Q_X

    # Y90 family: independent correction on physical channels
    y90_oc_phyI = y90_phyI + crab_I_Y
    y90_oc_phyQ = y90_phyQ + crab_Q_Y

    # Store waveforms — preserving the naming convention of the base gates:
    #   X90: X90_OC_wf_     → phys-I,  X90_OC_wf_der_ → phys-Q
    #   Y90: Y90_OC_wf_der_ → phys-I,  Y90_OC_wf_     → phys-Q
    config["waveforms"]["X90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": x90_oc_phyI.tolist(),
    }
    config["waveforms"]["X90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": x90_oc_phyQ.tolist(),
    }
    config["waveforms"]["mX90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": (-x90_oc_phyI).tolist(),
    }
    config["waveforms"]["mX90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": (-x90_oc_phyQ).tolist(),
    }
    config["waveforms"]["Y90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": y90_oc_phyI.tolist(),
    }
    config["waveforms"]["Y90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": y90_oc_phyQ.tolist(),
    }
    config["waveforms"]["mY90_OC_wf_der_" + q_name] = {
        "type": "arbitrary",
        "samples": (-y90_oc_phyI).tolist(),
    }
    config["waveforms"]["mY90_OC_wf_" + q_name] = {
        "type": "arbitrary",
        "samples": (-y90_oc_phyQ).tolist(),
    }

    def _add_pulse(name, wf_I_key, wf_Q_key):
        config["pulses"][name] = {
            "operation": "control",
            "length": len(config["waveforms"][wf_I_key]["samples"]),
            "waveforms": {"I": wf_I_key, "Q": wf_Q_key},
            "digital_marker": "trig",
        }

    _add_pulse(
        "X90_OC_pulse_" + q_name, "X90_OC_wf_" + q_name, "X90_OC_wf_der_" + q_name
    )
    _add_pulse(
        "mX90_OC_pulse_" + q_name, "mX90_OC_wf_" + q_name, "mX90_OC_wf_der_" + q_name
    )
    _add_pulse(
        "Y90_OC_pulse_" + q_name, "Y90_OC_wf_der_" + q_name, "Y90_OC_wf_" + q_name
    )
    _add_pulse(
        "mY90_OC_pulse_" + q_name, "mY90_OC_wf_der_" + q_name, "mY90_OC_wf_" + q_name
    )

    config["elements"][q_name]["operations"]["X90_OC"] = "X90_OC_pulse_" + q_name
    config["elements"][q_name]["operations"]["mX90_OC"] = "mX90_OC_pulse_" + q_name
    config["elements"][q_name]["operations"]["Y90_OC"] = "Y90_OC_pulse_" + q_name
    config["elements"][q_name]["operations"]["mY90_OC"] = "mY90_OC_pulse_" + q_name

    return config


def run_rb_fidelity(
    a_k_I_X,
    a_k_Q_X,
    a_k_I_Y,
    a_k_Q_Y,
    omega_k,
    phi_k_I,
    phi_k_Q,
    qubit,
    tg,
    meta_dict,
    experiment_dict,
    calib_params,
    outer_folder,
    iteration_label,
    crab_dir=None,
):
    """
    Builds the CRAB-corrected config with independent X and Y corrections, runs RB1Q,
    returns (gate_fidelity, save_dir). If crab_dir is given, the RB figure is moved there.
    """
    # Fresh config each call so waveform edits don't accumulate
    config = meta_dict_to_config_1Q(meta_dict)

    crab_wf_I_X = make_crab_waveform(tg, a_k_I_X, omega_k, phi_k_I)
    crab_wf_Q_X = make_crab_waveform(tg, a_k_Q_X, omega_k, phi_k_Q)
    crab_wf_I_Y = make_crab_waveform(tg, a_k_I_Y, omega_k, phi_k_I)
    crab_wf_Q_Y = make_crab_waveform(tg, a_k_Q_Y, omega_k, phi_k_Q)
    config = add_x90_oc_to_config(config, qubit, crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y)

    rb = RB1Q_CRAB(
        qubit,
        experiment_dict["RB"]["circuit_lengths"],
        experiment_dict["RB"]["num_circuits"],
        experiment_dict["RB"]["shots"],
        calib_params,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=qubit,
        suffix=f"RB_OC_{iteration_label}",
        config=config,
    )

    if crab_dir is not None and os.path.isfile(rb.iname):
        shutil.move(rb.iname, os.path.join(crab_dir, os.path.basename(rb.iname)))

    return rb.gate_fidelity(), crab_dir if crab_dir is not None else os.path.dirname(
        rb.iname
    )


# ── Fast feedback: return probability with optional decay fitting ────────────

def run_rb_fast_feedback(
    a_k_I_X,
    a_k_Q_X,
    a_k_I_Y,
    a_k_Q_Y,
    omega_k,
    phi_k_I,
    phi_k_Q,
    qubit,
    tg,
    meta_dict,
    fast_circuit_depths,
    shots,
    num_circuits,
    calib_params,
    outer_folder,
    iteration_label,
):
    """
    Run RB at one or more circuit depths for fast feedback with independent X and Y corrections.
    - If len(fast_circuit_depths) == 1: return return_probability at that depth
    - If len(fast_circuit_depths) > 1: fit exponential decay, return fitted decay parameter p
    """
    config = meta_dict_to_config_1Q(meta_dict)
    crab_wf_I_X = make_crab_waveform(tg, a_k_I_X, omega_k, phi_k_I)
    crab_wf_Q_X = make_crab_waveform(tg, a_k_Q_X, omega_k, phi_k_Q)
    crab_wf_I_Y = make_crab_waveform(tg, a_k_I_Y, omega_k, phi_k_I)
    crab_wf_Q_Y = make_crab_waveform(tg, a_k_Q_Y, omega_k, phi_k_Q)
    config = add_x90_oc_to_config(config, qubit, crab_wf_I_X, crab_wf_Q_X, crab_wf_I_Y, crab_wf_Q_Y)

    rb = RB1Q_CRAB(
        qubit,
        fast_circuit_depths,  # Can be single or multiple depths
        num_circuits,
        shots,
        calib_params,
        meta_dict=meta_dict,
        outer_folder=outer_folder,
        path=qubit,
        suffix=f"RB_fast_{iteration_label}",
        config=config,
    )

    # Extract return probability data
    if hasattr(rb, 'data') and 'return_prob' in rb.data:
        return_prob_array = rb.data['return_prob']
        if not isinstance(return_prob_array, (list, tuple, np.ndarray)):
            return_prob_array = [return_prob_array]
        return_prob_array = np.array(return_prob_array)
    else:
        return_prob_array = np.full(len(fast_circuit_depths), 0.5)

    # Single depth: return the return probability directly
    if len(fast_circuit_depths) == 1:
        feedback_metric = np.mean(return_prob_array)
    # Multiple depths: fit exponential decay and return decay parameter p
    else:
        def fitRB(x, a, p, b):
            return a * p ** x + b

        try:
            popt, _ = curve_fit(
                fitRB,
                fast_circuit_depths,
                return_prob_array,
                [0.4, 0.99, 0.5],
                maxfev=5000,
            )
            feedback_metric = popt[1]  # Return the decay parameter p
        except Exception:
            # Fallback: use mean return probability if fit fails
            feedback_metric = np.mean(return_prob_array)

    # Delete intermediate RB files
    if hasattr(rb, 'iname') and rb.iname and os.path.isfile(rb.iname):
        os.remove(rb.iname)
    if hasattr(rb, 'fname') and rb.fname and os.path.isfile(rb.fname):
        os.remove(rb.fname)

    return feedback_metric


# ── CMA-ES evaluate wrapper ───────────────────────────────────────────────────

_eval_counter = [0]  # mutable counter accessible inside closure

LONG_RB_FIDELITY_THRESHOLD = 0.99999


def make_evaluate_fast(
    qubit,
    tg,
    x90_amp,
    omega_k,
    phi_k_I,
    phi_k_Q,
    meta_dict,
    fast_circuit_depths,
    shots,
    num_circuits,
    calib_params,
    outer_folder,
):
    """
    Fast evaluate function with independent X and Y corrections (4K parameters total).
    Parameter layout: [a_I_X, a_Q_X, a_I_Y, a_Q_Y] each of length K
    - If len(fast_circuit_depths) == 1: cost = 1 - return_probability
    - If len(fast_circuit_depths) > 1: cost = 1 - decay_parameter_p (from exponential fit)
    Minimizes cost to maximize return probability or decay parameter.
    """
    a_max = x90_amp / 30
    K = len(omega_k)

    def evaluate(x):
        a_k_I_X = a_max * np.tanh(x[:K])
        a_k_Q_X = a_max * np.tanh(x[K:2*K])
        a_k_I_Y = a_max * np.tanh(x[2*K:3*K])
        a_k_Q_Y = a_max * np.tanh(x[3*K:4*K])
        _eval_counter[0] += 1
        label = f"cma_iter_{_eval_counter[0]:04d}"

        feedback_metric = run_rb_fast_feedback(
            a_k_I_X,
            a_k_Q_X,
            a_k_I_Y,
            a_k_Q_Y,
            omega_k,
            phi_k_I,
            phi_k_Q,
            qubit,
            tg,
            meta_dict,
            fast_circuit_depths,
            shots,
            num_circuits,
            calib_params,
            outer_folder,
            label,
        )

        cost = 1.0 - feedback_metric
        fidelity = 1.0 - 0.5 * cost / 2.2083  # Approximate fidelity

        metric_name = "return_prob" if len(fast_circuit_depths) == 1 else "decay_p"
        print(f"[{label}]  {metric_name} = {feedback_metric:.6f}  →  cost = {cost:.6f}  (approx F = {fidelity:.6f})")

        return cost

    return evaluate


def make_evaluate(
    qubit,
    tg,
    x90_amp,
    omega_k,
    phi_k_I,
    phi_k_Q,
    meta_dict,
    experiment_dict,
    calib_params,
    outer_folder,
    crab_dir=None,
    long_rb_experiment_dict=None,
):
    """
    Returns an evaluate(x) function closed over the fixed CRAB basis.
    x  : CMA vector of length 4K — independent corrections for X and Y:
         [a_I_X, a_Q_X, a_I_Y, a_Q_Y] each of length K.
    a_k decoded as a_max * tanh(x) → stays within ±a_max per component.
    If fidelity exceeds LONG_RB_FIDELITY_THRESHOLD a long RB is run immediately.
    """
    a_max = x90_amp / 30
    K = len(omega_k)

    def evaluate(x):
        a_k_I_X = a_max * np.tanh(x[:K])
        a_k_Q_X = a_max * np.tanh(x[K:2*K])
        a_k_I_Y = a_max * np.tanh(x[2*K:3*K])
        a_k_Q_Y = a_max * np.tanh(x[3*K:4*K])
        _eval_counter[0] += 1
        label = f"cma_iter_{_eval_counter[0]:04d}"
        print(
            f"[{label}]  X: I [{a_k_I_X.min():.4f}, {a_k_I_X.max():.4f}]  Q [{a_k_Q_X.min():.4f}, {a_k_Q_X.max():.4f}] | "
            f"Y: I [{a_k_I_Y.min():.4f}, {a_k_I_Y.max():.4f}]  Q [{a_k_Q_Y.min():.4f}, {a_k_Q_Y.max():.4f}]"
        )

        fidelity, _ = run_rb_fidelity(
            a_k_I_X,
            a_k_Q_X,
            a_k_I_Y,
            a_k_Q_Y,
            omega_k,
            phi_k_I,
            phi_k_Q,
            qubit,
            tg,
            meta_dict,
            experiment_dict,
            calib_params,
            outer_folder,
            label,
            crab_dir=crab_dir,
        )

        if np.isnan(fidelity):
            print("  WARNING: RB returned NaN, substituting cost=1.0")
            return 1.0

        print(f"  gate_fidelity = {fidelity:.8f}  →  cost = {1 - fidelity:.8f}")

        if fidelity > LONG_RB_FIDELITY_THRESHOLD and long_rb_experiment_dict is not None:
            print(f"  *** fidelity {fidelity:.8f} > {LONG_RB_FIDELITY_THRESHOLD} — running long RB ***")
            long_fidelity, _ = run_rb_fidelity(
                a_k_I_X, a_k_Q_X, a_k_I_Y, a_k_Q_Y,
                omega_k, phi_k_I, phi_k_Q,
                qubit, tg, meta_dict, long_rb_experiment_dict, calib_params,
                outer_folder, f"{label}_long", crab_dir=crab_dir,
            )
            print(f"  Long RB gate_fidelity = {long_fidelity:.8f}")

        return 1.0 - fidelity

    return evaluate


### USAGE -----------------------------------------------------------------

if __name__ == "__main__":
    qubit = "q6"
    K = 12
    fast = True  # Set to True for fast feedback, False for full RB
    fast_circuit_depths = [64, 256, 1000, 3000]  # Single depth [500], or multiple [500, 1000] for exponential fit


    # --- load qubit parameters ---
    meta_dict = make_meta_dict(qubit)[qubit]
    tg = meta_dict["X90"]["length"]
    x90_amp = meta_dict["X90"]["amp"]
    outer_folder = meta_dict["global_params"]["outer_folder"]
    os.makedirs(outer_folder, exist_ok=True)
    today = datetime.date.today().strftime("%Y_%m_%d")
    crab_dir = os.path.join(outer_folder, qubit, f"{qubit}_{today}", "crab_cma")
    os.makedirs(crab_dir, exist_ok=True)

    # Create a timestamped subdirectory for this specific CRAB run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    crab_run_dir = os.path.join(crab_dir, run_timestamp)
    os.makedirs(crab_run_dir, exist_ok=True)
    print(f"Saving CRAB outputs to: {crab_run_dir}")
    config = meta_dict_to_config_1Q(meta_dict)
    calib_params = {}

    print(f"Qubit: {qubit}  |  tg={tg} ns  |  x90_amp={x90_amp} V")


    # ── CRAB basis (fixed across all iterations) ──────────────────────────────
    #f_k = np.random.uniform(0.0, 250, K)  # MHz
    f_k = np.linspace(0.01/tg * 1e3 , 3/tg * 1e3, K)
    omega_k = f_k * 2 * np.pi / 1000  # rad/ns
    phi_k_I = np.random.uniform(0, 2 * np.pi, K)
    phi_k_Q = np.random.uniform(0, 2 * np.pi, K)
    
    experiment_dict = {
        "SS_Cal": {
            "run": True,
            "shots": 1000,
        },
        "RB": {
            "run": True,
            "simulate": False,
            "shots": 100,  # 100,
            "num_circuits": 200,
            "circuit_lengths": [
                2,
                # 4,
                8,
                16,
                # 32,
                64,
                128,
                256,
                500,
                1000,
                # 1200,
                1500,
                2000,
                # 2500,
                3000,
                6000,
            ],
        },
    }

    

    # ── Single-shot calibration ───────────────────────────────────────────────
    if experiment_dict["SS_Cal"]["run"]:
        print("Running single shot calibration...")
        SS = SingleShot1Q(
            qubit,
            experiment_dict["SS_Cal"]["shots"],
            meta_dict=meta_dict,
            outer_folder=outer_folder,
            path=qubit,
            suffix="SS_Cal",
            config=config,
        )
        calib_params = SS.calib_params
        meta_dict["read_theta"] = SS.calib_params["read_theta"]
        config = meta_dict_to_config_1Q(meta_dict)
        print(SS.calib_params)

    


    # ── Long-RB experiment dict (used for baseline, checkpoints, and final) ─────
    long_rb_experiment_dict = {
        "RB": {
            "circuit_lengths": [
                2,
                #4,
                #8,
                16,
                #32,
                64,
                #128,
                256,
                500,
                1000,
                #1200,
                1500,
                2000,
                2500,
                3000,
                5000,
                7000,
            ],
            "num_circuits": 100, #experiment_dict["RB"]["num_circuits"],
            "shots": 100,#experiment_dict["RB"]["shots"],
        }
    }
### DEBUGGING
    ## Noise floor analysis ---------------------------------
    # noise_samples = []
    # for i in range(5):
    #     f, _ = run_rb_fidelity(
    #         np.zeros(K), np.zeros(K),
    #         omega_k, phi_k_I, phi_k_Q,
    #         qubit, tg, meta_dict,
    #         long_rb_experiment_dict, calib_params,
    #         outer_folder, f"noise_probe_{i}"
    #     )
    #     noise_samples.append(1 - f)
        
    # noise_floor = np.std(noise_samples)
    # print(f"Empirical noise floor (1σ): {noise_floor:.2e}")

    ## Single tone sanity check ----------------------------------------
    # Single tone test — bypass CMA-ES entirely
    # a_max = x90_amp / 30
    # f_test = 400  # MHz, near anharmonicity
    # omega_test = np.array([f_test * 2 * np.pi / 1000])
    # a_test_Q = np.array([a_max * 0.5])  # half of a_max
    # a_test_I = np.array([0.0])
    # phi_test = np.array([0.0])

    # for a_scale in [0.0, 0.5, 1.0, 2.0, 3.0]:
    #     f, _ = run_rb_fidelity(
    #         a_test_I * a_scale,
    #         a_test_Q,
    #         omega_test,
    #         phi_test,
    #         phi_test,
    #         qubit,
    #         tg,
    #         meta_dict,
    #         experiment_dict, calib_params,
    #         outer_folder, "noise_probe"
    #     )
    #     print(f"a_scale={a_scale:.2f}  fidelity={f:.8f}")
    # ── Baseline RB (zero CRAB correction) ────────────────────────────────────
    baseline_fidelity = None
    baseline_feedback = None  # For fast mode: return_prob or decay parameter p

    if experiment_dict["RB"]["run"]:
        print("Running baseline RB (no correction)...")
        if fast and len(fast_circuit_depths) > 1:
            # Fast mode with multiple depths: compute baseline feedback metric
            baseline_feedback = run_rb_fast_feedback(
                np.zeros(K),
                np.zeros(K),
                np.zeros(K),
                np.zeros(K),
                omega_k,
                phi_k_I,
                phi_k_Q,
                qubit,
                tg,
                meta_dict,
                fast_circuit_depths,
                experiment_dict["RB"]["shots"],
                experiment_dict["RB"]["num_circuits"],
                calib_params,
                crab_run_dir,  # Save all CRAB data to timestamped subdirectory
                "baseline_fast",
            )
            print(f"Baseline fast feedback metric: {baseline_feedback:.6f}\n")
        else:
            # Full mode or single-depth fast: use long RB baseline
            baseline_fidelity, _ = run_rb_fidelity(
                np.zeros(K),
                np.zeros(K),
                np.zeros(K),
                np.zeros(K),
                omega_k,
                phi_k_I,
                phi_k_Q,
                qubit,
                tg,
                meta_dict,
                long_rb_experiment_dict,
                calib_params,
                crab_run_dir,  # Save all CRAB data to timestamped subdirectory
                "baseline",
                crab_dir=crab_run_dir,
            )
            print(f"Baseline gate fidelity: {baseline_fidelity:.8f}\n")

    def run_long_rb_checkpoint(iteration, label, is_fast=False):
        """Run long RB validation without creating convergence plot."""
        ck_x = es.result.xbest
        ck_a_k_I_X = (x90_amp / 30) * np.tanh(ck_x[:K])
        ck_a_k_Q_X = (x90_amp / 30) * np.tanh(ck_x[K:2*K])
        ck_a_k_I_Y = (x90_amp / 30) * np.tanh(ck_x[2*K:3*K])
        ck_a_k_Q_Y = (x90_amp / 30) * np.tanh(ck_x[3*K:4*K])

        # Long RB for validation
        print(f"\n[{label}] Running checkpoint long RB...")
        if is_fast:
            # Save long RB validation data every 10 iterations in fast mode too
            ck_fidelity, _ = run_rb_fidelity(
                ck_a_k_I_X,
                ck_a_k_Q_X,
                ck_a_k_I_Y,
                ck_a_k_Q_Y,
                omega_k,
                phi_k_I,
                phi_k_Q,
                qubit,
                tg,
                meta_dict,
                long_rb_experiment_dict,
                calib_params,
                crab_run_dir,  # Save all CRAB data to timestamped subdirectory
                label,
                crab_dir=crab_run_dir,  # Save long RB results even in fast mode
            )
        else:
            ck_fidelity, _ = run_rb_fidelity(
                ck_a_k_I_X,
                ck_a_k_Q_X,
                ck_a_k_I_Y,
                ck_a_k_Q_Y,
                omega_k,
                phi_k_I,
                phi_k_Q,
                qubit,
                tg,
                meta_dict,
                long_rb_experiment_dict,
                calib_params,
                crab_run_dir,  # Save all CRAB data to timestamped subdirectory
                label,
                crab_dir=crab_run_dir,
            )
        print(f"[{label}] gate_fidelity = {ck_fidelity:.8f}")
        return ck_fidelity

    # ── CMA-ES optimisation ───────────────────────────────────────────────────
    if fast:
        mode_str = f"single depth {fast_circuit_depths[0]}" if len(fast_circuit_depths) == 1 else f"exponential fit over {fast_circuit_depths}"
        print(f"\n*** FAST MODE: {mode_str} ***\n")
        evaluate = make_evaluate_fast(
            qubit,
            tg,
            x90_amp,
            omega_k,
            phi_k_I,
            phi_k_Q,
            meta_dict,
            fast_circuit_depths=fast_circuit_depths,
            shots=experiment_dict["RB"]["shots"],
            num_circuits=experiment_dict["RB"]["num_circuits"],
            calib_params=calib_params,
            outer_folder=crab_run_dir,  # Save all CRAB data to timestamped subdirectory
        )
    else:
        print("\n*** FULL MODE: Using RB fidelity with full circuit_lengths ***\n")
        evaluate = make_evaluate(
            qubit,
            tg,
            x90_amp,
            omega_k,
            phi_k_I,
            phi_k_Q,
            meta_dict,
            experiment_dict,
            calib_params,
            crab_run_dir,  # Save all CRAB data to timestamped subdirectory
            crab_dir=crab_run_dir,
            long_rb_experiment_dict=long_rb_experiment_dict,
        )

    es = cma.CMAEvolutionStrategy(
        np.zeros(4 * K),  # 4K: [I_X, Q_X, I_Y, Q_Y] each of length K (independent X and Y)
        0.2,
        {
            "maxiter": 500,
            "popsize": int(4 + 3*np.log(4 * K)),  # 4 + floor(3 * log(N))
            "tolfun": 1e-5,  # stop when best cost changes by less than this much between gens
            "tolx": 1e-6,
            "verbose": 3,
        },
    )

    history = []
    sigma_history = []
    iteration = 0

    # Create convergence plot that will be updated each iteration
    fig, ax = plt.subplots(figsize=(10, 6))

    while not es.stop():
        solutions = es.ask()
        costs = [evaluate(x) for x in solutions]
        es.tell(solutions, costs)
        es.disp()
        history.append(min(costs))
        sigma_history.append(es.sigma)
        iteration += 1
        print(f"\n{'─' * 60}")
        print(
            f"  History — best cost per generation (overall best={es.result.fbest:.6e})"
        )
        for i, (h, sig) in enumerate(zip(history, sigma_history)):
            marker = " ◄" if h == min(history) else ""
            print(f"  Gen {i + 1:3d}: {h:.6e}  (fidelity={1.0 - h:.8f})  σ={sig:.4f}{marker}")
        print(f"{'─' * 60}\n")

        # Update convergence plot every iteration
        ax.clear()
        ax.plot(history, marker="o", markersize=3)

        if fast:
            # Fast mode: y-axis is cost (1 - return_prob or 1 - decay_p)
            baseline_cost = 1.0 - baseline_feedback if baseline_feedback is not None else 0.5
            ax.axhline(
                baseline_cost, color="gray", linestyle="--", label="baseline (no correction)"
            )
            ax.set_ylabel("Best cost (1 − feedback metric)")
        else:
            # Full mode: y-axis is 1 - fidelity
            ax.axhline(
                1 - baseline_fidelity, color="gray", linestyle="--", label="baseline"
            )
            ax.set_ylabel("Best (1 − fidelity)")

        ax.set_xlabel("CMA-ES iteration")
        ax.set_yscale("log")
        ax.set_title(f"CRAB-OC optimisation — {qubit}  [iter {iteration}]")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(crab_run_dir, f"cma_convergence_{qubit}.png"))

        if iteration % 10 == 0:
            run_long_rb_checkpoint(iteration, f"iter{iteration:04d}", is_fast=fast)

    # Close the convergence plot
    plt.close(fig)

    # ── Results ───────────────────────────────────────────────────────────────
    best_x = es.result.xbest
    best_a_k_I_X = (x90_amp / 30) * np.tanh(best_x[:K])
    best_a_k_Q_X = (x90_amp / 30) * np.tanh(best_x[K:2*K])
    best_a_k_I_Y = (x90_amp / 30) * np.tanh(best_x[2*K:3*K])
    best_a_k_Q_Y = (x90_amp / 30) * np.tanh(best_x[3*K:4*K])
    best_fid = 1.0 - es.result.fbest

    print(f"\nBaseline fidelity : {baseline_fidelity:.8f}")
    print(f"Optimised fidelity: {best_fid:.8f}")
    print(f"Improvement       : {(best_fid - baseline_fidelity) * 100:+.3f} pp")

    # ── Final long RB validation ────────────────────────────────────────────────
    run_long_rb_checkpoint(iteration, "final", is_fast=fast)

    # ── Save best parameters ──────────────────────────────────────────────────
    np.savez(
        os.path.join(crab_run_dir, f"best_crab_params_{qubit}.npz"),
        a_k_I_X=best_a_k_I_X,
        a_k_Q_X=best_a_k_Q_X,
        a_k_I_Y=best_a_k_I_Y,
        a_k_Q_Y=best_a_k_Q_Y,
        omega_k=omega_k,
        phi_k_I=phi_k_I,
        phi_k_Q=phi_k_Q,
        baseline_fidelity=baseline_fidelity,
        best_fidelity=best_fid,
        history=history,
    )
    print(f"Parameters saved to {crab_run_dir}/best_crab_params_{qubit}.npz")

    ### Testing --------------------------------------------------------
    # K_test = 10

    # meta_dict = make_meta_dict(qubit)[qubit]
    # tg = meta_dict["X90"]["length"]
    # config = meta_dict_to_config_1Q(meta_dict)
    # x90_amp = meta_dict["X90"]["amp"]

    # f_test    = np.array([13.6])              # MHz
    # omega_test = f_test * 2 * np.pi / 1000   # rad/ns
    # a_k_I = np.array([0.01])
    # a_k_Q = np.array([0.0])

    # f_test = np.random.uniform(0, 300, K_test)
    # omega_test = f_test * 2 * np.pi / 1000

    # phi_k_I = np.random.uniform(0, 2 * np.pi, K_test)
    # phi_k_Q = np.random.uniform(0, 2 * np.pi, K_test)

    # print("tg is", tg, "and x90_amp is", x90_amp)

    # crab_wf_I = make_crab_waveform(tg, a_k_I, omega_test, phi_k_I)
    # crab_wf_Q = make_crab_waveform(tg, a_k_Q, omega_test, phi_k_Q)
    # config = add_x90_oc_to_config(config, qubit, crab_wf_I, crab_wf_Q)

    # with program() as prog:
    #     with infinite_loop_():
    #         # reset_frame(qubit)
    #         play("X90_OC", qubit)

    # qmm = QuantumMachinesManager(host="192.168.0.134")
    # qm = qmm.open_qm(config)

    # qmm.clear_all_job_results()

    # job = qm.execute(prog)
